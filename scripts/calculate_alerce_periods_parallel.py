#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALeRCE Parallel Period Calculator

This script calculates periods for all AleRCE objects using parallel processing.
It preloads all AstroObjects for better performance and distributes the workload
across multiple CPU cores.

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
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import ALeRCE modules
from lc_classifier.features.core.base import astro_object_from_dict, AstroObject
from lc_classifier.features.extractors.period_extractor import PeriodExtractor
from lc_classifier.features.preprocess.ztf import ZTFLightcurvePreprocessor


def create_period_extractor(
    bands=['g', 'r'], 
    unit="magnitude",
    smallest_period=0.045,  # ~1.2 hours
    largest_period=100.0,  # 100 days
    trim_lightcurve_to_n_days=None,
    min_length=15,
    use_forced_photo=True,
    return_power_rates=True,
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


def extract_period_for_object(astro_object, plot=False, max_period=100.0, oid=None):
    """
    Extract period for a specific AstroObject.
    
    Parameters:
    -----------
    astro_object : AstroObject
        The AstroObject to process
    plot : bool
        Whether to plot the results (not used in parallel processing)
    max_period : float
        Maximum period to search for in days (default: 100.0)
    oid : str or None
        Object ID (if None, will be extracted from the object)
    
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
            return {"oid": "unknown", "period": np.nan, "significance": np.nan, "class": None, 
                    "error": f"Failed to get object ID: {str(e)}"}
    
    try:
        # Create period extractor with specified maximum period
        period_extractor = create_period_extractor(largest_period=max_period)
        
        # Clean existing period features if any
        if astro_object.features is not None and not astro_object.features.empty:
            try:
                period_features = [
                    'Multiband_period', 'PPE', 'Period_band', 'delta_period'
                ] + period_extractor.pr_names
                astro_object.features = astro_object.features[
                    ~astro_object.features['name'].isin(period_features)
                ]
            except Exception as e:
                # If cleaning fails, continue with the extraction
                pass
    
        # Extract period features
        period_extractor.compute_features_single_object(astro_object)
        
        # Get period and significance
        period_features = astro_object.features[astro_object.features['name'] == 'Multiband_period']
        ppe_features = astro_object.features[astro_object.features['name'] == 'PPE']
        period_1 = astro_object.features[astro_object.features['name'] == 'Period_band' & astro_object.features['fid'] == 'g' ]
        period_2 = astro_object.features[astro_object.features['name'] == 'Period_band' & astro_object.features['fid'] == 'r' ]
        delta_period_1 = astro_object.features[astro_object.features['name'] == 'delta_period' & astro_object.features['fid'] == 'g' ]
        delta_period_2 = astro_object.features[astro_object.features['name'] == 'delta_period' & astro_object.features['fid'] == 'r' ]

        period = period_features['value'].values[0] if len(period_features) > 0 else np.nan
        significance = ppe_features['value'].values[0] if len(ppe_features) > 0 else np.nan
        period_1 = period_1['value'].values[0] if len(period_1) > 0 else np.nan
        period_2 = period_2['value'].values[0] if len(period_2) > 0 else np.nan
        delta_period_1 = delta_period_1['value'].values[0] if len(delta_period_1) > 0 else np.nan
        delta_period_2 = delta_period_2['value'].values[0] if len(delta_period_2) > 0 else np.nan
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
            "significance": np.nan,
            "period_1": np.nan,
            "period_2": np.nan,
            "delta_period_1": np.nan,
            "delta_period_2": np.nan,
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
            except Exception as obj_error:
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
            print(f"Error preprocessing batch {batch_file}: {str(preprocess_error)}")
        
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
        print(f"Error processing batch {batch_file}: {str(e)}")
        return {}


def load_all_astro_objects(data_folder, max_batches=None, show_progress=True, save_to_cache=True, 
                         cache_path='all_astro_objects_cache.pkl', num_processes=None):
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
        
    Returns:
    --------
    dict
        Dictionary mapping OIDs to their AstroObjects
    """
    # Check if cache exists and try to load it
    if os.path.exists(cache_path):
        print(f"Found cache file at {cache_path}. Attempting to load...")
        try:
            with open(cache_path, 'rb') as f:
                all_objects = pickle.load(f)
                print(f"Successfully loaded {len(all_objects)} objects from cache.")
                return all_objects
        except Exception as e:
            print(f"Error loading from cache: {str(e)}")
            print("Proceeding with loading from batch files.")
    
    start_time = time.time()
    
    # Get all batch files
    try:
        astro_objects_filenames = os.listdir(data_folder)
        astro_objects_filenames = [
            f for f in astro_objects_filenames if "astro_objects_batch" in f
        ]
        print(f"Found {len(astro_objects_filenames)} batch files")
    except FileNotFoundError:
        print(f"Data folder '{data_folder}' not found. Please check the path.")
        return {}
    
    # Get batch files to process
    batch_files = astro_objects_filenames[:max_batches] if max_batches else astro_objects_filenames
    total_batches = len(batch_files)
    
    print(f"Loading objects from {total_batches} batch files in parallel...")
    
    # Set number of processes
    if num_processes is None:
        num_processes = -1  # Use all available CPUs
        
    try:
        # Process batch files in parallel with progress tracking
        if show_progress:
            print(f"Using {num_processes if num_processes > 0 else 'all available'} processes")
            # We need to use a different approach since joblib with verbose doesn't work well with tqdm
            batch_results = []
            for i, batch_file in enumerate(tqdm(batch_files, desc="Loading batch files")):
                # Load batch file one at a time but process in parallel
                batch_dict = load_batch_file(batch_file, data_folder)
                batch_results.append(batch_dict)
                
                # Garbage collection every 10 batches to manage memory
                if (i + 1) % 10 == 0:
                    gc.collect()
        else:
            # Process batch files in parallel using joblib
            batch_results = Parallel(n_jobs=num_processes, verbose=5)(
                delayed(load_batch_file)(batch_file, data_folder) 
                for batch_file in batch_files
            )
        
        # Combine results from all batches
        all_objects = {}
        for batch_dict in batch_results:
            all_objects.update(batch_dict)
        
    except Exception as e:
        print(f"Error during parallel loading: {str(e)}")
        print("Falling back to sequential loading...")
        
        # Fall back to sequential loading if parallel fails
        all_objects = {}
        for batch_file in tqdm(batch_files, desc="Loading batch files (sequential)"):
            batch_dict = load_batch_file(batch_file, data_folder)
            all_objects.update(batch_dict)
    
    elapsed_time = time.time() - start_time
    print(f"Loaded {len(all_objects)} objects in {elapsed_time:.2f} seconds.")
    
    # Save to cache if requested
    if save_to_cache:
        print(f"Saving objects to cache at {cache_path}...")
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(all_objects, f)
            print("Successfully saved to cache.")
        except Exception as e:
            print(f"Error saving to cache: {str(e)}")
    
    return all_objects


# These functions are no longer needed with joblib approach
# Joblib automatically distributes the work across processes


def run_period_calculation_parallel(all_objects, num_processes=None, max_period=100.0, 
                            output_file='periods_results.csv', batch_size=100):
    """
    Calculate periods for all objects in parallel using joblib.
    
    Parameters:
    -----------
    all_objects : dict
        Dictionary mapping OIDs to AstroObjects
    num_processes : int or None
        Number of processes to use (None uses all CPUs)
    max_period : float
        Maximum period to search for in days
    output_file : str
        Path to save the results CSV
    batch_size : int
        Number of objects to process in each batch (helps manage memory usage)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with period calculation results
    """
    # Set number of processes
    if num_processes is None:
        # Use all available CPUs
        num_processes = -1
    
    print(f"Running period calculation with {num_processes if num_processes > 0 else 'all available'} processes")
    print(f"Maximum period set to {max_period} days")
    print(f"Using batch size of {batch_size} objects")
    
    # Prepare object items for parallel processing
    objects_list = list(all_objects.items())
    num_objects = len(objects_list)
    total_batches = (num_objects + batch_size - 1) // batch_size
    print(f"Processing {num_objects} objects in {total_batches} batches")
    
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
                delayed(extract_period_for_object)(obj, plot=False, max_period=max_period, oid=oid) 
                for oid, obj in batch
            )
            
            # Filter out None results (should not happen, but just in case)
            batch_results = [r for r in batch_results if r is not None]
            all_results.extend(batch_results)
            
            # Optional: Save intermediate results
            if (i + batch_size) % (batch_size * 5) == 0 or (i + batch_size) >= num_objects:
                intermediate_df = pd.DataFrame(all_results)
                intermediate_file = f"{os.path.splitext(output_file)[0]}_intermediate.csv"
                intermediate_df.to_csv(intermediate_file, index=False)
                print(f"Saved {len(all_results)}/{num_objects} intermediate results to {intermediate_file}")
                
            # Optional garbage collection to free memory
            if (i + batch_size) % (batch_size * 10) == 0:
                gc.collect()
                
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {str(e)}")
            # Continue with the next batch instead of failing completely
    
    results = all_results
    elapsed_time = time.time() - start_time
    print(f"Processed {len(results)} objects in {elapsed_time:.2f} seconds")
    print(f"Average processing time: {elapsed_time / len(results):.4f} seconds per object")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
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
    
    # If class information is already in results
    if 'class' in results_df.columns and not results_df['class'].isna().all():
        return results_df[results_df['class'] == class_name]
    
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
                return merged_df[merged_df[class_column] == class_name]
    
    print(f"Cannot filter by class '{class_name}': class information not available")
    return results_df


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='ALeRCE Period Calculator with Parallel Processing')
    
    parser.add_argument('--data-folder', type=str, 
                        default="/home/fsoto/Documents/LCsSSL/data/alerce_data_raw/data_250408_ndetge8_ao",
                        help='Path to folder containing AstroObject batch files')
    
    parser.add_argument('--objects-file', type=str,
                        default="/home/fsoto/Documents/LCsSSL/data/alerce_data_raw/ts_v9.0.1_b3000_objs_250408_ndetge8.parquet",
                        help='Path to objects file (parquet) containing classification information')
    
    parser.add_argument('--output-file', type=str, 
                        default='periods_results.csv',
                        help='Path to save the output CSV file')
    
    parser.add_argument('--cache-file', type=str, 
                        default='all_astro_objects_cache.pkl',
                        help='Path to cache file for loaded AstroObjects')
    
    parser.add_argument('--max-batches', type=int, 
                        default=None,
                        help='Maximum number of batch files to process (None for all)')
    
    parser.add_argument('--max-period', type=float, 
                        default=1000.0,
                        help='Maximum period to search for in days')
    
    parser.add_argument('--processes', type=int, 
                        default=None,
                        help='Number of parallel processes to use (default: all CPUs, -1 for all, positive number for specific count)')
    
    parser.add_argument('--class-filter', type=str, 
                        default=None,
                        help='Filter objects by class (if available)')
    
    parser.add_argument('--no-cache', action='store_true', 
                        help='Do not use cache even if available')
    
    parser.add_argument('--batch-size', type=int,
                        default=10,
                        help='Number of objects to process in each parallel batch (helps with memory management)')
    
    return parser.parse_args()


def main():
    """Main execution function"""
    start_time = time.time()
    
    try:
        args = parse_arguments()
        
        print("=" * 80)
        print("ALeRCE Period Calculator with Parallel Processing")
        print("=" * 80)
        print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(os.path.abspath(args.output_file))
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                print(f"Created output directory: {output_dir}")
            except Exception as e:
                print(f"Warning: Could not create output directory: {str(e)}")
        
        # Load objects file if specified
        objects_df = None
        if args.objects_file and os.path.exists(args.objects_file):
            try:
                print(f"Loading objects file: {args.objects_file}")
                objects_df = pd.read_parquet(args.objects_file)
                print(f"Loaded {len(objects_df)} objects")
            except Exception as e:
                print(f"Warning: Could not load objects file: {str(e)}")
        
        # Load all AstroObjects
        all_objects = load_all_astro_objects(
            args.data_folder,
            max_batches=args.max_batches,
            show_progress=True,
            save_to_cache=not args.no_cache,
            cache_path=args.cache_file,
            num_processes=args.processes
        )
        
        if not all_objects:
            print("No objects loaded. Exiting.")
            return 1
        
        # Run period calculation in parallel
        results = run_period_calculation_parallel(
            all_objects,
            num_processes=args.processes,
            max_period=args.max_period,
            output_file=args.output_file,
            batch_size=args.batch_size
        )
        
        # Generate statistics about results
        total_objects = len(results)
        valid_periods = results[~np.isnan(results['period'])].shape[0]
        error_objects = results['error'].notna().sum() if 'error' in results.columns else 0
        mean_significance = results['significance'].mean()
        
        print("-" * 80)
        print("Period Calculation Results Summary:")
        print(f"Total objects processed: {total_objects}")
        print(f"Objects with valid periods: {valid_periods} ({valid_periods/total_objects*100:.1f}%)")
        print(f"Objects with errors: {error_objects} ({error_objects/total_objects*100:.1f}%)")
        print(f"Mean significance: {mean_significance:.4f}")
        
        # Filter by class if specified
        if args.class_filter:
            print(f"Filtering results by class: {args.class_filter}")
            try:
                filtered_results = filter_by_class(
                    results,
                    objects_df=objects_df,
                    class_name=args.class_filter
                )
                
                # Save filtered results
                filtered_output = f"{os.path.splitext(args.output_file)[0]}_{args.class_filter}.csv"
                filtered_results.to_csv(filtered_output, index=False)
                print(f"Filtered results ({len(filtered_results)} objects) saved to {filtered_output}")
            except Exception as e:
                print(f"Error filtering results by class: {str(e)}")
        
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(int(elapsed_time), 3600)
        minutes, seconds = divmod(remainder, 60)
        print("-" * 80)
        print(f"Total execution time: {hours}h {minutes}m {seconds}s")
        print("Done!")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting...")
        return 1
    except Exception as e:
        print(f"Critical error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
