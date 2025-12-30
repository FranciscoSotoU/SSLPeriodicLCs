from os import path
import os
import numpy as np
import pandas as pd
import requests
from sqlalchemy import create_engine
from joblib import Parallel, delayed
from tqdm import tqdm
import logging
from lc_classifier.utils import (
    all_features_from_astro_objects,
    create_astro_object,
    EmptyLightcurveException,
    plot_astro_object
)
from lc_classifier.features.core.base import save_astro_objects_batch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def patch_xmatch_by_oid(oid: str):
    data = {
        "oid": [oid],
        "w1mpro": [np.nan],
        "w2mpro": [np.nan],
        "w3mpro": [np.nan],
        "w4mpro": [np.nan],
        "sgscore1": [np.nan],
        "sgmag1": [np.nan],
        "srmag1": [np.nan],
        "simag1": [np.nan],
        "szmag1": [np.nan],
        "distpsnr1": [np.nan],
    }
    return pd.DataFrame(data)

def dataframes_to_astro_object_list(
    detections,
    forced_photometry,
    xmatch,
    reference,
    features=None,
    data_origin="database",
    verbose=True,
):
    oids = detections["oid"].unique()
    
    # Pre-index all dataframes once for faster lookups
    detections_indexed = detections.set_index("oid").sort_index()
    forced_photometry_columns = forced_photometry.columns
    forced_photometry_indexed = forced_photometry.set_index("oid").sort_index()
    xmatch_indexed = xmatch.set_index("oid")
    reference_columns = reference.columns
    reference_indexed = reference.set_index("oid")
    
    astro_objects_list = []
    
    # Add progress bar for OID processing
    oids_iter = tqdm(oids, desc="Processing OIDs", disable=not verbose, leave=False)
    
    for oid in oids_iter:
        try:
            # Use pre-indexed dataframes for faster lookups
            try:
                xmatch_oid = xmatch_indexed.loc[[oid]].reset_index()
            except KeyError:
                xmatch_oid = patch_xmatch_by_oid(oid=oid).reset_index()

            if len(xmatch_oid) != 1:
                continue  # Skip invalid xmatch data
                
            xmatch_oid = xmatch_oid.iloc[0]
            
            # Use .get() with pre-indexed dataframes for faster access
            if oid in forced_photometry_indexed.index:
                forced_photometry_oid = forced_photometry_indexed.loc[[oid]].reset_index()
            else:
                forced_photometry_oid = pd.DataFrame(columns=forced_photometry_columns)

            if oid in reference_indexed.index:
                reference_oid = reference_indexed.loc[[oid]].reset_index()
            else:
                reference_oid = pd.DataFrame(columns=reference_columns)
                
            try:
                ao = create_astro_object(
                    data_origin=data_origin,
                    detections=detections_indexed.loc[[oid]].reset_index(),
                    forced_photometry=forced_photometry_oid,
                    xmatch=xmatch_oid,
                    reference=reference_oid,
                    non_detections=None,
                )
                if features is not None:
                    """add features from db"""
                    try:
                        ao.features = features.loc[features.oid == oid][
                            ["name", "value", "fid", "version"]
                        ]
                    except:
                        ao.features = None
            except EmptyLightcurveException:
                continue

            astro_objects_list.append(ao)
        except Exception:
            # Silent skip on errors to avoid spam
            continue
            
    return astro_objects_list

def extract_from_db(top_objects_batch, engine, fill=False):
     oids_list = top_objects_batch['oid'].tolist()
     oids_chunk = [f"'{oid}'" for oid in oids_list]

     # Query for detections
     query_detections = f"""
     SELECT * FROM detection
     WHERE oid in ({','.join(oids_chunk)}) and rb >= 0.55;
     """
     detections = pd.read_sql_query(query_detections, con=engine)
     
     # Query for xmatch
     query_xmatch = f"""
     SELECT oid, oid_catalog, dist FROM xmatch
     WHERE oid in ({','.join(oids_chunk)}) and catid='allwise';
     """
     xmatch = pd.read_sql_query(query_xmatch, con=engine)
     xmatch = xmatch.sort_values("dist").drop_duplicates("oid")
     oid_catalog = [f"'{oid}'" for oid in xmatch["oid_catalog"].values]
     
     if oid_catalog == []:
          return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
     
     # Query for WISE data
     query_wise = f"""
     SELECT oid_catalog, w1mpro, w2mpro, w3mpro, w4mpro FROM allwise
     WHERE oid_catalog in ({','.join(oid_catalog)});
     """
     wise = pd.read_sql_query(query_wise, con=engine).set_index("oid_catalog")
     wise = pd.merge(xmatch, wise, on="oid_catalog", how="outer")
     wise = wise[["oid", "w1mpro", "w2mpro", "w3mpro", "w4mpro"]].set_index("oid")
     null_oids = wise[wise[["w1mpro", "w2mpro", "w3mpro", "w4mpro"]].isnull().any(axis=1)].index.tolist()
     # null oids should also contain the oids that are not in the xmatch table
     # oids_chunk is a list of quoted oids, so we need to remove the quotes for comparison
     oids_unquoted = [oid.strip("'") for oid in oids_list]
     null_oids += [oid for oid in oids_unquoted if (oid not in wise.index and oid not in null_oids)]

     if null_oids and fill:
          objects = top_objects_batch[['oid', 'meanra', 'meandec']]
          objects = objects.set_index("oid")
          for oid in null_oids:
               ra, dec = objects.loc[oid, ['meanra', 'meandec']]
               response = requests.get(
                    f"https://catshtm.alerce.online/crossmatch?catalog=WISE&ra={ra}&dec={dec}"
               )
               if response.status_code != 200:
                    w1 = w2 = w3 = w4 = np.nan
               else:
                    data = response.json()
                    w1 = data.get('Mag_W1', {}).get('value', np.nan)
                    w2 = data.get('Mag_W2', {}).get('value', np.nan)
                    w3 = data.get('Mag_W3', {}).get('value', np.nan)
                    w4 = data.get('Mag_W4', {}).get('value', np.nan)
               wise.loc[oid, ['w1mpro', 'w2mpro', 'w3mpro', 'w4mpro']] = [w1, w2, w3, w4]
     wise_ = wise
     
     # Query for forced photometry
     query_forced_photometry = f"""
     SELECT * FROM forced_photometry
     WHERE oid in ({','.join(oids_chunk)}) and procstatus in ('0', '57');
     """
     query_reference = f"""
     SELECT oid, rfid, sharpnr, chinr FROM reference
     WHERE oid in ({','.join(oids_chunk)});
     """
     reference = pd.read_sql_query(query_reference, con=engine)
     reference = reference.drop_duplicates('rfid')

     forced_photometry = pd.read_sql_query(query_forced_photometry, con=engine)
     
     # Query for PS1 data
     query_ps = f"""
     SELECT oid, sgscore1, sgmag1, srmag1, simag1, szmag1, distpsnr1 FROM ps1_ztf
     WHERE oid in ({','.join(oids_chunk)});
     """
     ps = pd.read_sql_query(query_ps, con=engine)
     ps = ps.drop_duplicates("oid").set_index("oid")

     # Merge xmatch and PS1 data
     xmatch_ = pd.concat([wise_, ps], axis=1).reset_index()
     
     return detections, forced_photometry, xmatch_, reference


def process_batch(batch_info, db_params):
    """Process a single batch of objects in parallel."""
    batch_start, batch_objects = batch_info
    
    try:
        # Create engine for this worker with optimized settings
        engine = create_engine(
            f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}@{db_params['host']}/{db_params['dbname']}",
            pool_pre_ping=True,  # Check connections before use
            pool_recycle=3600,   # Recycle connections every hour
        )
        
        # Extract data from database - now passing the DataFrame
        detections, forced_photometry, xmatch, reference = extract_from_db(batch_objects, engine)
        
        if detections.empty:
            engine.dispose()
            return batch_start, []
        
        # Create astro objects
        ao_list = dataframes_to_astro_object_list(
            detections=detections,
            forced_photometry=forced_photometry,
            xmatch=xmatch,
            reference=reference,
            features=None,
            data_origin="database",
            verbose=False,  # Disable verbose for parallel processing
        )
        
        # Close the engine connection
        engine.dispose()
        
        return batch_start, ao_list
        
    except Exception as e:
        logger.error(f"Error processing batch {batch_start}: {e}")
        return batch_start, []


# Read parquet files
data_path_top = 'data/top_objects_4.parquet'
data_num = 4
top_objects_df = pd.read_parquet(data_path_top)
print(f"Loaded {len(top_objects_df)} objects from top_objects.parquet")
print(f"Columns available: {list(top_objects_df.columns)}")

URL = "https://raw.githubusercontent.com/alercebroker/usecases/master/alercereaduser_v4.json"
PARAMS = requests.get(URL).json()['params']
DB_PARAMs = {
    'user': PARAMS['user'],
    'password': PARAMS['password'],
    'host': PARAMS['host'],
    'dbname': PARAMS['dbname']
}

batch_size = 50
total_batches = (len(top_objects_df) + batch_size - 1) // batch_size
print(f"Processing {total_batches} batches of size {batch_size}")

# Create output directory for batches
output_dir = "data/astro_objects_batches"
os.makedirs(output_dir, exist_ok=True)
print(f"Saving batches to: {output_dir}")

# Prepare batch information for parallel processing
batch_info_list = []
for batch_start in range(0, len(top_objects_df), batch_size):
    batch_objects = top_objects_df.iloc[batch_start:batch_start + batch_size]
    batch_info_list.append((batch_start, batch_objects))

# Process batches in smaller groups for frequent saving
max_workers = min(8, total_batches)  # Reduced workers for better stability
save_frequency = 1  # Reduced to save more frequently
print(f"Using {max_workers} parallel workers, saving every {save_frequency} batches")

# Initialize counters
processed_count = 0
saved_count = 0
total_objects_saved = 0

# Process batches in small groups for frequent saving
group_iterator = range(0, len(batch_info_list), save_frequency)
with tqdm(total=len(batch_info_list), desc="Processing batches", unit="batch") as pbar:
    for group_start in group_iterator:
        group_end = min(group_start + save_frequency, len(batch_info_list))
        current_group = batch_info_list[group_start:group_end]
        
        print(f"Processing group {group_start//save_frequency + 1} ({len(current_group)} batches)")
        
        # Process current group in parallel with optimized backend
        group_results = Parallel(n_jobs=max_workers, backend='threading', verbose=0, batch_size=1)(
            delayed(process_batch)(batch_info, DB_PARAMs) 
            for batch_info in current_group
        )
    
        # Save results immediately after each group
        for batch_start, ao_list in group_results:
            processed_count += 1
            
            if ao_list:  # Save immediately when batch completes
                try:
                    save_astro_objects_batch(
                        astro_objects=ao_list,
                        filename=f'{output_dir}/astro_objects_batch_{data_num}_{batch_start}.pkl'
                    )
                    saved_count += 1
                    total_objects_saved += len(ao_list)
                    print(f"✓ Batch {batch_start}: Saved {len(ao_list)} objects")
                except Exception as e:
                    print(f"✗ Error saving batch {batch_start}: {e}")
            else:
                print(f"○ Batch {batch_start}: No objects to save")
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'Saved': saved_count,
                'Objects': total_objects_saved,
                'Rate': f"{saved_count/processed_count:.1%}" if processed_count > 0 else "0%"
            })
        
        # Print progress after each group
        print(f"Group completed: {processed_count}/{total_batches} batches processed, {saved_count} batches saved, {total_objects_saved} total objects")
        print("-" * 50)

print(f"Processing completed! Total: {processed_count} batches processed, {saved_count} batches saved, {total_objects_saved} objects saved")