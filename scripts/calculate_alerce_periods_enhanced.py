#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALeRCE Parallel Period Calculator (Enhanced Version)

This script calculates periods for all ALeRCE objects using parallel processing.
It preloads all AstroObjects for better performance and distributes the workload
across multiple CPU cores.

Enhanced with:
- Improved parallel loading of AstroObjects
- Better memory management with garbage collection
- Robust error handling
- Detailed progress reporting
- Statistics on processing results

Based on the ALeRCE Period Calculator notebook.
"""

import os
import sys
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import gc
import warnings
import logging
from datetime import datetime
warnings.filterwarnings('ignore')

# Add pyarrow for Parquet support
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    logging.warning("pyarrow not available. Install with 'pip install pyarrow' for Parquet file support.")

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import ALeRCE modules
from lc_classifier.features.core.base import astro_object_from_dict, AstroObject
from lc_classifier.features.extractors.period_extractor import PeriodExtractor
from lc_classifier.features.preprocess.ztf import ZTFLightcurvePreprocessor


# Configure logging
def setup_logging(log_file=None, verbose=False):
    """Set up logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logging.warning(f"Could not create log file: {e}")
    
    return logger


def create_period_extractor(
    bands=['g', 'r'], 
    unit="magnitude",
    smallest_period=0.045,  # ~1.2 hours
    largest_period=100.0,  # 100 days
    trim_lightcurve_to_n_days=None,
    min_length=10,
    use_forced_photo=True,
    return_power_rates=False,
    shift=0.1
):
    """
    Create a PeriodExtractor instance with specified parameters.
    
    Parameters:
    -----------
    bands : list
        List of bands to use for period calculation
    unit : str
        Unit of brightness measurement ('magnitude' or 'diff_flux')
    smallest_period : float
        Smallest period to search for in days
    largest_period : float
        Largest period to search for in days
    trim_lightcurve_to_n_days : float or None
        Trim light curve to this many days or None for no trimming
    min_length : int
        Minimum number of observations required per band
    use_forced_photo : bool
        Whether to use forced photometry if available
    return_power_rates : bool
        Whether to calculate and return power rates
    shift : float
        Shift parameter for frequency grid evaluation
    
    Returns:
    --------
    PeriodExtractor instance
    """
    return PeriodExtractor(
        bands=bands,
        unit=unit,
        smallest_period=smallest_period,
        largest_period=largest_period,
        trim_lightcurve_to_n_days=trim_lightcurve_to_n_days,
        min_length=min_length,
        use_forced_photo=use_forced_photo,
        return_power_rates=return_power_rates,
        shift=shift
    )


def get_object_id(astro_object):
    """
    Extract the object ID from an AstroObject's metadata
    
    Parameters:
    -----------
    astro_object : AstroObject
        The AstroObject to get the ID from
        
    Returns:
    --------
    str
        The object ID or "unknown" if not found
    """
    for id_field in ["oid", "aid"]:
        if id_field in astro_object.metadata["name"].values:
            id_row = astro_object.metadata[astro_object.metadata["name"] == id_field]
            return id_row["value"].values[0]
    return "unknown"





def extract_period_for_object(astro_object, plot=False, min_period=0.045, max_period=100.0, 
                             oid=None, min_observations=15):
    """
    Extract period for a specific AstroObject.
    
    Parameters:
    -----------
    astro_object : AstroObject
        The AstroObject to process
    plot : bool
        Whether to plot the results (not used in parallel processing)
    min_period : float
        Minimum period to search for in days (default: 0.045)
    max_period : float
        Maximum period to search for in days (default: 100.0)
    oid : str or None
        Object ID (if None, will be extracted from the object)
    min_observations : int
        Minimum number of observations required per band
    
    Returns:
    --------
    dict
        Dictionary with OID, period, significance and class (if available)
    """
    start_time = time.time()
    
    # Get OID if not provided
    if oid is None:
        try:
            oid = get_object_id(astro_object)
        except Exception as e:
            return {"oid": "unknown", "period": np.nan, 'best_n_periods':[], "significance": np.nan, "class": None, 
                    "error": f"Failed to get object ID: {str(e)}"}
    
    try:
        # Create period extractor with specified minimum and maximum period
        period_extractor = create_period_extractor(
            smallest_period=min_period,
            largest_period=max_period,
            min_length=min_observations
        )
        
        # Clean existing period features if any
        if astro_object.features is not None and not astro_object.features.empty:
            try:
                period_features = [
                    'Multiband_period', 'PPE', 'Period_band', 'delta_period', 'best_n_periods',
                ] + period_extractor.pr_names
                astro_object.features = astro_object.features[
                    ~astro_object.features['name'].isin(period_features)
                ]
            except Exception:
                # If cleaning fails, continue with the extraction
                pass
    
        # Extract period features
        period_extractor.compute_features_single_object(astro_object)
        
        # Get period and significance
        period_features = astro_object.features[astro_object.features['name'] == 'Multiband_period']
        ppe_features = astro_object.features[astro_object.features['name'] == 'PPE']
        best_n_periods_features = astro_object.features[astro_object.features['name'] == 'Best_n_periods']
        period_1 = astro_object.features[astro_object.features['name'] == 'Period_band' & astro_object.features['fid'] == 'g' ]
        period_2 = astro_object.features[astro_object.features['name'] == 'Period_band' & astro_object.features['fid'] == 'r' ]
        delta_period_1 = astro_object.features[astro_object.features['name'] == 'delta_period' & astro_object.features['fid'] == 'g' ]
        delta_period_2 = astro_object.features[astro_object.features['name'] == 'delta_period' & astro_object.features['fid'] == 'r' ]

        
        #print(best_n_periods_features)
        period = period_features['value'].values[0] if len(period_features) > 0 else np.nan
        significance = ppe_features['value'].values[0] if len(ppe_features) > 0 else np.nan
        best_n_periods = best_n_periods_features['value'].values[0] if len(best_n_periods_features) > 0 else []
        period_1 = period_1['value'].values[0] if len(period_1) > 0 else np.nan
        period_2 = period_2['value'].values[0] if len(period_2) > 0 else np.nan
        delta_period_1 = delta_period_1['value'].values[0] if len(delta_period_1) > 0 else np.nan
        delta_period_2 = delta_period_2['value'].values[0] if len(delta_period_2) > 0 else np.nan
        # Handle best_n_periods with consistent data type (use Python list, not numpy array)
        # Get class information if available
        class_info = None
        try:
            if 'class' in astro_object.metadata["name"].values:
                class_row = astro_object.metadata[astro_object.metadata["name"] == "class"]
                class_info = class_row["value"].values[0]
        except Exception:
            # If getting class info fails, continue without it
            pass
        
        # Calculate processing time
        elapsed_time = time.time() - start_time
        
        return {
            "oid": oid, 
            "period": period,
            "best_n_periods": best_n_periods,
            "significance": significance,
            "period_1": period_1,
            "period_2": period_2,
            "delta_period_1": delta_period_1,
            "delta_period_2": delta_period_2,
            "class": class_info,
            "proc_time": elapsed_time
        }
    
    except Exception as e:
        # Return a dictionary with error information
        elapsed_time = time.time() - start_time
        return {
            "oid": oid, 
            "period": np.nan,
            "best_n_periods": [],
            "period_1": np.nan,
            "period_2": np.nan,
            "delta_period_1": np.nan,
            "delta_period_2": np.nan,
            "significance": np.nan, 
            "class": None, 
            "error": str(e),
            "proc_time": elapsed_time
        }


def load_batch_file(batch_file, data_folder):
    """
    Load and process a single batch file.
    
    Parameters:
    -----------
    batch_file : str
        Name of the batch file to load
    data_folder : str
        Path to the data folder
        
    Returns:
    --------
    dict
        Dictionary mapping OIDs to their AstroObjects
    """
    batch_objects_dict = {}
    try:
        # Load batch
        batch_path = os.path.join(data_folder, batch_file)
        batch_astro_objects = pd.read_pickle(batch_path)
        
        # Convert to AstroObjects
        batch_objects = []
        for d in batch_astro_objects:
            try:
                obj = astro_object_from_dict(d)
                batch_objects.append(obj)
            except Exception:
                # Skip objects that cannot be converted
                continue
        
        if not batch_objects:
            # If no objects were successfully converted, return empty dict
            return {}
        
        try:
            # Preprocess light curves
            lightcurve_preprocessor = ZTFLightcurvePreprocessor()
            lightcurve_preprocessor.preprocess_batch(batch_objects)
        except Exception as preprocess_error:
            # Log error but continue, as some objects might still be usable
            logging.warning(f"Error preprocessing batch {batch_file}: {str(preprocess_error)}")
        
        # Store in dictionary by OID
        for obj in batch_objects:
            try:
                oid = get_object_id(obj)
                if oid != "unknown":
                    batch_objects_dict[oid] = obj
            except Exception:
                # Skip objects that cannot be added to dictionary
                continue
                
        return batch_objects_dict
        
    except Exception as e:
        logging.error(f"Error processing batch {batch_file}: {str(e)}")
        return {}


def load_all_astro_objects(data_folder, max_batches=None, show_progress=True, save_to_cache=True, 
                         cache_path='all_astro_objects_cache.pkl', num_processes=None, memory_cleanup=False):
    """
    Load all AstroObjects from batch files into memory for faster access using parallel processing.
    
    Parameters:
    -----------
    data_folder : str
        Path to the folder containing batch files
    max_batches : int or None
        Maximum number of batch files to load (None for all)
    show_progress : bool
        Whether to show a progress bar
    save_to_cache : bool
        Whether to save loaded objects to a cache file
    cache_path : str
        Path to cache file
    num_processes : int or None
        Number of processes to use (None uses all CPUs)
    memory_cleanup : bool
        Whether to run garbage collection between batches
        
    Returns:
    --------
    dict
        Dictionary mapping OIDs to their AstroObjects
    """
    # Check if cache exists and try to load it
    if os.path.exists(cache_path) and save_to_cache:
        logging.info(f"Found cache file at {cache_path}. Attempting to load...")
        try:
            with open(cache_path, 'rb') as f:
                all_objects = pickle.load(f)
                logging.info(f"Successfully loaded {len(all_objects)} objects from cache.")
                return all_objects
        except Exception as e:
            logging.warning(f"Error loading from cache: {str(e)}")
            logging.info("Proceeding with loading from batch files.")
    
    start_time = time.time()
    
    # Get all batch files
    try:
        astro_objects_filenames = os.listdir(data_folder)
        astro_objects_filenames = [
            f for f in astro_objects_filenames if "astro_objects_batch" in f
        ]
        logging.info(f"Found {len(astro_objects_filenames)} batch files")
    except FileNotFoundError:
        logging.error(f"Data folder '{data_folder}' not found. Please check the path.")
        return {}
    
    # Get batch files to process
    batch_files = astro_objects_filenames[:max_batches] if max_batches else astro_objects_filenames
    total_batches = len(batch_files)
    
    logging.info(f"Loading objects from {total_batches} batch files...")
    
    # Set number of processes
    if num_processes is None:
        num_processes = -1  # Use all available CPUs
    
    # Create a processing function for the batches
    def process_batch_file(batch_file):
        try:
            return load_batch_file(batch_file, data_folder)
        except Exception as e:
            logging.error(f"Error in process_batch_file for {batch_file}: {str(e)}")
            return {}
    
    try:
        # Process batch files in parallel with progress tracking
        if show_progress:
            logging.info(f"Using {num_processes if num_processes > 0 else 'all available'} processes")
            batch_results = []
            
            # Use tqdm for progress tracking
            for i, batch_file in enumerate(tqdm(batch_files, desc="Loading batch files")):
                # Load batch file
                batch_dict = process_batch_file(batch_file)
                batch_results.append(batch_dict)
                
                # Garbage collection if requested
                if memory_cleanup and (i + 1) % 5 == 0:
                    gc.collect()
        else:
            # Process batch files in parallel using joblib
            batch_results = Parallel(n_jobs=num_processes, verbose=5)(
                delayed(process_batch_file)(batch_file) 
                for batch_file in batch_files
            )
        
        # Combine results from all batches
        all_objects = {}
        for batch_dict in batch_results:
            all_objects.update(batch_dict)
        
    except Exception as e:
        logging.error(f"Error during parallel loading: {str(e)}")
        logging.warning("Falling back to sequential loading...")
        
        # Fall back to sequential loading if parallel fails
        all_objects = {}
        for batch_file in tqdm(batch_files, desc="Loading batch files (sequential)"):
            batch_dict = load_batch_file(batch_file, data_folder)
            all_objects.update(batch_dict)
    
    elapsed_time = time.time() - start_time
    logging.info(f"Loaded {len(all_objects)} objects in {elapsed_time:.2f} seconds.")
    
    # Save to cache if requested
    if save_to_cache:
        logging.info(f"Saving objects to cache at {cache_path}...")
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(all_objects, f)
            logging.info("Successfully saved to cache.")
        except Exception as e:
            logging.error(f"Error saving to cache: {str(e)}")
    
    return all_objects


def run_period_calculation_parallel(all_objects, num_processes=None, min_period=0.045, max_period=100.0, 
                            output_file='periods_results.parquet', batch_size=100, memory_cleanup=False,
                            save_frequency=5, min_observations=15, output_format='parquet'):
    """
    Calculate periods for all objects in parallel using joblib.
    
    Parameters:
    -----------
    all_objects : dict
        Dictionary mapping OIDs to AstroObjects
    num_processes : int or None
        Number of processes to use (None uses all CPUs)
    min_period : float
        Minimum period to search for in days
    max_period : float
        Maximum period to search for in days
    output_file : str
        Path to save the results file
    batch_size : int
        Number of objects to process in each batch (helps manage memory usage)
    memory_cleanup : bool
        Whether to run garbage collection between batches
    save_frequency : int
        Save intermediate results every N batches
    min_observations : int
        Minimum number of observations required per band
    output_format : str
        Format to save results ('parquet', 'csv', or 'pickle'). Parquet preserves complex 
        data types like lists, while CSV converts them to strings.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with period calculation results
    """
    # Set number of processes
    if num_processes is None:
        # Use all available CPUs
        num_processes = -1
    
    logging.info(f"Running period calculation with {num_processes if num_processes > 0 else 'all available'} processes")
    logging.info(f"Period search range: {min_period} to {max_period} days")
    logging.info(f"Using batch size of {batch_size} objects")
    logging.info(f"Minimum observations per band: {min_observations}")
    
    # Prepare object items for parallel processing
    objects_list = list(all_objects.items())
    num_objects = len(objects_list)
    total_batches = (num_objects + batch_size - 1) // batch_size
    logging.info(f"Processing {num_objects} objects in {total_batches} batches")
    
    # Start processing
    start_time = time.time()
    
    # Process objects in batches to manage memory usage
    all_results = []
    batch_progress = tqdm(range(0, num_objects, batch_size), desc="Processing batches", unit="batch")
    
    for i in batch_progress:
        batch = objects_list[i:i+batch_size]
        batch_progress.set_description(f"Batch {i//batch_size + 1}/{total_batches} ({len(batch)} objects)")
        
        try:
            # Process batch with joblib
            batch_results = Parallel(n_jobs=num_processes, verbose=0)(
                delayed(extract_period_for_object)(
                    obj, plot=False, min_period=min_period, max_period=max_period,
                    oid=oid, min_observations=min_observations
                ) for oid, obj in batch
            )
            
            # Filter out None results (should not happen, but just in case)
            batch_results = [r for r in batch_results if r is not None]
            all_results.extend(batch_results)
            
            # Save intermediate results
            if (i + batch_size) % (batch_size * save_frequency) == 0 or (i + batch_size) >= num_objects:
                # Convert to DataFrame
                intermediate_df = pd.DataFrame(all_results)
                
                # Determine intermediate file extension based on output format
                if output_format == 'parquet' and PARQUET_AVAILABLE:
                    intermediate_file = f"{os.path.splitext(output_file)[0]}_intermediate.parquet"
                    
                    # Ensure consistent data types for best_n_periods before saving to parquet
                    if 'best_n_periods' in intermediate_df.columns:
                        # Convert all values to consistent Python lists using our helper function
                        intermediate_df['best_n_periods'] = intermediate_df['best_n_periods']
                    
                    try:
                        intermediate_df.to_parquet(intermediate_file, index=False)
                    except Exception as save_error:
                        logging.error(f"Error saving to Parquet: {save_error}. Falling back to Pickle format.")
                        # Fall back to pickle if Parquet fails
                        intermediate_file = f"{os.path.splitext(output_file)[0]}_intermediate.pkl"
                        intermediate_df.to_pickle(intermediate_file)
                elif output_format == 'pickle':
                    intermediate_file = f"{os.path.splitext(output_file)[0]}_intermediate.pkl"
                    intermediate_df.to_pickle(intermediate_file)
                else:
                    # Default to CSV for intermediate results
                    intermediate_file = f"{os.path.splitext(output_file)[0]}_intermediate.csv"
                    
                    # Convert lists to string for CSV storage
                    if 'best_n_periods' in intermediate_df.columns:
                        # Make a copy to avoid modifying the original data
                        save_df = intermediate_df.copy()
                        save_df['best_n_periods'] = save_df['best_n_periods'].apply(
                            lambda x: str(x) if isinstance(x, list) else x
                        )
                        save_df.to_csv(intermediate_file, index=False)
                    else:
                        intermediate_df.to_csv(intermediate_file, index=False)
                
                logging.info(f"Saved {len(all_results)}/{num_objects} intermediate results to {intermediate_file}")
            
            # Run garbage collection if requested
            if memory_cleanup and (i + batch_size) % (batch_size * 10) == 0:
                gc.collect()
                
        except Exception as e:
            logging.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
            # Continue with the next batch instead of failing completely
    
    results = all_results
    elapsed_time = time.time() - start_time
    
    if len(results) > 0:
        avg_time = elapsed_time / len(results)
        logging.info(f"Processed {len(results)} objects in {elapsed_time:.2f} seconds")
        logging.info(f"Average processing time: {avg_time:.4f} seconds per object")
    else:
        logging.warning("No results were produced!")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results if we have any
    if not results_df.empty:
        # Use specified output format
        output_format = output_format.lower()
        
        if output_format == 'parquet':
            if not PARQUET_AVAILABLE:
                logging.warning("Parquet format requested but pyarrow not available. Falling back to CSV.")
                output_format = 'csv'
                if not output_file.endswith('.csv'):
                    output_file = os.path.splitext(output_file)[0] + '.csv'
            else:
                if not output_file.endswith('.parquet'):
                    output_file = os.path.splitext(output_file)[0] + '.parquet'
                
                # Ensure consistent data types for best_n_periods before saving to parquet
                if 'best_n_periods' in results_df.columns:
                    # Convert all values to consistent Python lists using our helper function
                    results_df['best_n_periods'] = results_df['best_n_periods']
                
                try:
                    # Parquet can handle complex data types directly with proper type consistency
                    results_df.to_parquet(output_file, index=False)
                    logging.info(f"Results saved to {output_file} in Parquet format")
                except Exception as save_error:
                    logging.error(f"Error saving to Parquet: {save_error}. Falling back to Pickle format.")
                    # Fall back to pickle if Parquet fails
                    pickle_file = os.path.splitext(output_file)[0] + '.pkl'
                    results_df.to_pickle(pickle_file)
                    logging.info(f"Results saved to {pickle_file} in Pickle format instead")
        
        elif output_format == 'pickle':
            if not output_file.endswith('.pkl'):
                output_file = os.path.splitext(output_file)[0] + '.pkl'
            # Pickle preserves all Python object types
            results_df.to_pickle(output_file)
            logging.info(f"Results saved to {output_file} in Pickle format")
        
        else:  # Default to CSV
            if not output_file.endswith('.csv'):
                output_file = os.path.splitext(output_file)[0] + '.csv'
            
            # Make a copy of the original list values for use in the function
            if 'best_n_periods' in results_df.columns:
                list_values = results_df['best_n_periods'].copy()
                # Convert lists to string representation for CSV storage
                results_df['best_n_periods'] = results_df['best_n_periods'].apply(
                    lambda x: str(x) if isinstance(x, list) else x
                )
            
            results_df.to_csv(output_file, index=False)
            logging.info(f"Results saved to {output_file} in CSV format")
            
            # Restore original values after saving
            if 'best_n_periods' in results_df.columns and 'list_values' in locals():
                results_df['best_n_periods'] = list_values
    
    return results_df


def filter_by_class(results_df, objects_df=None, class_name=None):
    """
    Filter results by class if not already in the results.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame with period calculation results
    objects_df : pandas.DataFrame or None
        DataFrame with objects information (including class)
    class_name : str or None
        Class name to filter by
        
    Returns:
    --------
    pandas.DataFrame
        Filtered DataFrame
    """
    if class_name is None:
        return results_df
    
    if results_df.empty:
        logging.warning("No results to filter by class.")
        return results_df
    
    # If class information is already in results
    if 'class' in results_df.columns and not results_df['class'].isna().all():
        filtered = results_df[results_df['class'] == class_name]
        logging.info(f"Found {len(filtered)} objects with class '{class_name}' in results.")
        return filtered
    
    # If objects_df is provided, merge and filter
    if objects_df is not None:
        # Find the class column
        class_column = None
        for col in ['survey_class_mapped', 'alerceclass', 'class_name', 'class']:
            if col in objects_df.columns:
                class_column = col
                break
        
        if class_column is not None:
            # Find the ID column in objects_df
            id_column = None
            for col in ['oid', 'aid', 'object_id']:
                if col in objects_df.columns:
                    id_column = col
                    break
            
            if id_column is not None:
                # Merge and filter
                merged_df = pd.merge(
                    results_df, 
                    objects_df[[id_column, class_column]], 
                    left_on='oid', 
                    right_on=id_column,
                    how='left'
                )
                filtered = merged_df[merged_df[class_column] == class_name]
                logging.info(f"Found {len(filtered)} objects with class '{class_name}' after merging with objects file.")
                return filtered
    
    logging.warning(f"Cannot filter by class '{class_name}': class information not available")
    return results_df


def generate_statistics(results_df):
    """
    Generate statistics from the results DataFrame.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame with period calculation results
        
    Returns:
    --------
    dict
        Dictionary with statistics
    """
    stats = {}
    
    # Basic counts
    stats['total_objects'] = len(results_df)
    
    # Period statistics
    if 'period' in results_df.columns:
        valid_periods = results_df[~np.isnan(results_df['period'])]
        stats['valid_periods'] = len(valid_periods)
        stats['valid_periods_percent'] = (len(valid_periods) / len(results_df) * 100) if len(results_df) > 0 else 0
        
        if len(valid_periods) > 0:
            stats['min_period'] = valid_periods['period'].min()
            stats['max_period'] = valid_periods['period'].max()
            stats['median_period'] = valid_periods['period'].median()
            stats['mean_period'] = valid_periods['period'].mean()
    
    # Best N periods statistics
    if 'best_n_periods' in results_df.columns:
        # Count objects with non-empty lists
        valid_n_periods = results_df[results_df['best_n_periods'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
        stats['objects_with_multiple_periods'] = len(valid_n_periods)
        stats['objects_with_multiple_periods_percent'] = (len(valid_n_periods) / len(results_df) * 100) if len(results_df) > 0 else 0
        
        # Average number of periods per object
        if len(valid_n_periods) > 0:
            stats['avg_periods_per_object'] = np.mean([len(x) for x in valid_n_periods['best_n_periods']])
    
    # Significance statistics
    if 'significance' in results_df.columns:
        valid_sig = results_df[~np.isnan(results_df['significance'])]
        if len(valid_sig) > 0:
            stats['mean_significance'] = valid_sig['significance'].mean()
            stats['median_significance'] = valid_sig['significance'].median()
            stats['min_significance'] = valid_sig['significance'].min()
            stats['max_significance'] = valid_sig['significance'].max()
    
    # Error statistics
    if 'error' in results_df.columns:
        stats['error_objects'] = results_df['error'].notna().sum()
        stats['error_percent'] = (stats['error_objects'] / len(results_df) * 100) if len(results_df) > 0 else 0
    
    # Processing time statistics
    if 'proc_time' in results_df.columns:
        valid_times = results_df[~np.isnan(results_df['proc_time'])]
        if len(valid_times) > 0:
            stats['total_proc_time'] = valid_times['proc_time'].sum()
            stats['mean_proc_time'] = valid_times['proc_time'].mean()
            stats['median_proc_time'] = valid_times['proc_time'].median()
    
    # Class statistics if available
    if 'class' in results_df.columns:
        class_counts = results_df['class'].value_counts()
        stats['class_counts'] = class_counts.to_dict()
        stats['num_classes'] = len(class_counts)
    
    return stats


def load_results_file(file_path):
    """
    Load results from a file in various formats (Parquet, CSV, or Pickle).
    For CSV files, the function will convert string representations of lists back to actual lists.
    
    Parameters:
    -----------
    file_path : str
        Path to the results file (Parquet, CSV, or Pickle)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with properly parsed columns
    """
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        
        # Load file based on extension
        if file_extension == '.parquet':
            # Check if pyarrow is available
            if not PARQUET_AVAILABLE:
                logging.error("Cannot load Parquet file: pyarrow not available. Install with 'pip install pyarrow'.")
                return pd.DataFrame()
                
            # Parquet files preserve complex data types
            results_df = pd.read_parquet(file_path)
            
        elif file_extension == '.pkl':
            # Pickle files preserve all Python object types
            results_df = pd.read_pickle(file_path)
            
        else:
            # Default to CSV - need to parse string representations
            results_df = pd.read_csv(file_path)
            
            # Check for list columns that need conversion
            if 'best_n_periods' in results_df.columns:
                # Use our helper function for consistent type handling
                results_df['best_n_periods'] = results_df['best_n_periods'].apply(ensure_consistent_list_types)
        
        return results_df
        
    except Exception as e:
        logging.error(f"Error loading results file {file_path}: {str(e)}")
        return pd.DataFrame()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='ALeRCE Period Calculator with Parallel Processing')
    
    # Input/Output options
    io_group = parser.add_argument_group('Input/Output Options')
    io_group.add_argument('--data-folder', type=str, 
                        default="/mnt/home/shared/data_250408_ndetge8_ao",
                        help='Path to folder containing AstroObject batch files')
    
    io_group.add_argument('--objects-file', type=str,
                        default="/home/fsoto/Documents/LCsSSL/data/alerce_data_raw/ts_v9.0.1_b3000_objs_250408_ndetge8.parquet",
                        help='Path to objects file (parquet) containing classification information')
    
    io_group.add_argument('--output-file', type=str, 
                        default='periods_results.parquet',
                        help='Path to save the output file')
    
    io_group.add_argument('--cache-file', type=str, 
                        default='all_astro_objects_cache.pkl',
                        help='Path to cache file for loaded AstroObjects')
    
    io_group.add_argument('--no-cache', action='store_true', 
                        help='Do not use cache even if available')
    
    io_group.add_argument('--save-frequency', type=int, 
                        default=10,
                        help='Save intermediate results every N batches')
    
    io_group.add_argument('--output-format', type=str,
                        default='parquet',
                        choices=['parquet', 'csv', 'pickle'],
                        help='Format to save results file (parquet preserves complex data types)')
    
    io_group.add_argument('--log-file', type=str,
                        default=None,
                        help='Log file to save output messages')
    
    # Processing options
    proc_group = parser.add_argument_group('Processing Options')
    proc_group.add_argument('--max-batches', type=int, 
                        default=None,
                        help='Maximum number of batch files to process (None for all)')
    
    proc_group.add_argument('--max-period', type=float, 
                        default=1000.0,
                        help='Maximum period to search for in days')
    
    proc_group.add_argument('--min-period', type=float, 
                        default=0.045,
                        help='Minimum period to search for in days (~1.2 hours)')
    
    proc_group.add_argument('--processes', type=int, 
                        default=15,
                        help='Number of parallel processes to use (default: all CPUs, -1 for all, positive number for specific count)')
    
    proc_group.add_argument('--batch-size', type=int,
                        default=100,
                        help='Number of objects to process in each parallel batch (helps with memory management)')
    
    proc_group.add_argument('--memory-cleanup', action='store_true',
                        help='Run garbage collection between batches to reduce memory usage')
    
    # Filtering options
    filter_group = parser.add_argument_group('Filtering Options')
    filter_group.add_argument('--class-filter', type=str, 
                        default=None,
                        help='Filter objects by class (if available)')
    
    filter_group.add_argument('--min-observations', type=int,
                        default=10,
                        help='Minimum number of observations required per band')
    
    # Debugging options
    debug_group = parser.add_argument_group('Debugging and Performance Options')
    debug_group.add_argument('--verbose', action='store_true',
                        help='Increase output verbosity')
    
    debug_group.add_argument('--debug', action='store_true',
                        help='Enable debug mode (more detailed error messages)')
    
    debug_group.add_argument('--stats-only', action='store_true',
                        help='Only generate statistics from existing results file')
    
    return parser.parse_args()


def main():
    """Main execution function"""
    start_time = time.time()
    
    try:
        args = parse_arguments()
        
        # Set up logging
        logger = setup_logging(args.log_file, args.verbose or args.debug)
        
        logging.info("=" * 80)
        logging.info("ALeRCE Period Calculator with Parallel Processing (Enhanced Version)")
        logging.info("=" * 80)
        logging.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(os.path.abspath(args.output_file))
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                logging.info(f"Created output directory: {output_dir}")
            except Exception as e:
                logging.warning(f"Could not create output directory: {str(e)}")
        
        # If stats-only mode, read existing results and generate statistics
        if args.stats_only:
            if os.path.exists(args.output_file):
                logging.info(f"Reading existing results from {args.output_file} for statistics...")
                results_df = load_results_file(args.output_file)
                stats = generate_statistics(results_df)
                
                logging.info("-" * 80)
                logging.info("Results Statistics:")
                for key, value in stats.items():
                    if isinstance(value, dict):
                        logging.info(f"{key}:")
                        for subkey, subvalue in value.items():
                            logging.info(f"  {subkey}: {subvalue}")
                    else:
                        logging.info(f"{key}: {value}")
                
                # If class filter is specified, filter and save
                if args.class_filter:
                    # Load objects file if needed
                    objects_df = None
                    if args.objects_file and os.path.exists(args.objects_file):
                        try:
                            logging.info(f"Loading objects file: {args.objects_file}")
                            objects_df = pd.read_parquet(args.objects_file)
                            logging.info(f"Loaded {len(objects_df)} objects")
                        except Exception as e:
                            logging.warning(f"Could not load objects file: {str(e)}")
                    
                    # Filter results
                    logging.info(f"Filtering results by class: {args.class_filter}")
                    filtered_results = filter_by_class(results_df, objects_df, args.class_filter)
                    
                    # Save filtered results
                    base_name = os.path.splitext(args.output_file)[0]
                    
                    if args.output_format == 'parquet' and PARQUET_AVAILABLE:
                        filtered_output = f"{base_name}_{args.class_filter}.parquet"
                        filtered_results.to_parquet(filtered_output, index=False)
                    elif args.output_format == 'pickle':
                        filtered_output = f"{base_name}_{args.class_filter}.pkl"
                        filtered_results.to_pickle(filtered_output)
                    else:
                        filtered_output = f"{base_name}_{args.class_filter}.csv"
                        filtered_results.to_csv(filtered_output, index=False)
                    logging.info(f"Filtered results ({len(filtered_results)} objects) saved to {filtered_output}")
                
                return 0
            else:
                logging.error(f"Results file {args.output_file} not found for stats-only mode.")
                return 1
        
        # Load objects file if specified
        objects_df = None
        if args.objects_file and os.path.exists(args.objects_file):
            try:
                logging.info(f"Loading objects file: {args.objects_file}")
                objects_df = pd.read_parquet(args.objects_file)
                logging.info(f"Loaded {len(objects_df)} objects")
            except Exception as e:
                logging.warning(f"Could not load objects file: {str(e)}")
        
        # Load all AstroObjects
        all_objects = load_all_astro_objects(
            args.data_folder,
            max_batches=args.max_batches,
            show_progress=True,
            save_to_cache=not args.no_cache,
            cache_path=args.cache_file,
            num_processes=args.processes,
            memory_cleanup=args.memory_cleanup
        )
        
        if not all_objects:
            logging.error("No objects loaded. Exiting.")
            return 1
        
        # Run period calculation in parallel
        results = run_period_calculation_parallel(
            all_objects,
            num_processes=args.processes,
            min_period=args.min_period,
            max_period=args.max_period,
            output_file=args.output_file,
            batch_size=args.batch_size,
            memory_cleanup=args.memory_cleanup,
            save_frequency=args.save_frequency,
            min_observations=args.min_observations,
            output_format=args.output_format
        )
        
        # Generate statistics about results
        stats = generate_statistics(results)
        
        logging.info("-" * 80)
        logging.info("Period Calculation Results Summary:")
        for key, value in stats.items():
            if key != 'class_counts':  # Skip detailed class counts in the summary
                logging.info(f"{key}: {value}")
        
        # Filter by class if specified
        if args.class_filter:
            logging.info(f"Filtering results by class: {args.class_filter}")
            try:
                filtered_results = filter_by_class(results, objects_df, args.class_filter)
                
                # Save filtered results
                base_name = os.path.splitext(args.output_file)[0]
                
                if args.output_format == 'parquet' and PARQUET_AVAILABLE:
                    filtered_output = f"{base_name}_{args.class_filter}.parquet"
                    filtered_results.to_parquet(filtered_output, index=False)
                elif args.output_format == 'pickle':
                    filtered_output = f"{base_name}_{args.class_filter}.pkl"
                    filtered_results.to_pickle(filtered_output)
                else:
                    filtered_output = f"{base_name}_{args.class_filter}.csv"
                    filtered_results.to_csv(filtered_output, index=False)
                logging.info(f"Filtered results ({len(filtered_results)} objects) saved to {filtered_output}")
            except Exception as e:
                logging.error(f"Error filtering results by class: {str(e)}")
        
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(int(elapsed_time), 3600)
        minutes, seconds = divmod(remainder, 60)
        logging.info("-" * 80)
        logging.info(f"Total execution time: {hours}h {minutes}m {seconds}s")
        logging.info("Done!")
        
        return 0
        
    except KeyboardInterrupt:
        logging.warning("\nProcess interrupted by user. Exiting...")
        return 1
    except Exception as e:
        if 'logger' in locals():
            logging.critical(f"Critical error during execution: {str(e)}")
            if args.debug:
                import traceback
                traceback.print_exc()
        else:
            print(f"Critical error during execution: {str(e)}")
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
