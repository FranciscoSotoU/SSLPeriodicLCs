import logging
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
import os
import sys
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

from numpy import short
from tqdm import tqdm
import warnings
from numba.core.errors import NumbaWarning
from scipy.optimize import OptimizeWarning

import numpy as np
warnings.filterwarnings("ignore", category=np.RankWarning)

# Set environment variables for threading control
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Remove JAX CPU-only restriction to allow GPU usage
# os.environ["JAX_PLATFORM_NAME"] = "cpu"  # Comment out to enable GPU
import jax

from joblib import Parallel, delayed

folder = "/home/fsoto/Documents/LCsSSL/data/astro_objects_batches"
output_folder = folder + "_features"
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

astro_objects_filenames = os.listdir(folder)
astro_objects_filenames = [
    f for f in astro_objects_filenames if "astro_objects_batch" in f
]


def extract_features(
    batch_id, ao_filename, shorten_n_days=None, skip_if_output_exists=False
):
    import pandas as pd
    from lc_classifier.features.preprocess.ztf import (
        ZTFLightcurvePreprocessor,
        ShortenPreprocessor,
    )
    from lc_classifier.features.composites.ztf import ZTFFeatureExtractor
    from lc_classifier.features.core.base import astro_object_from_dict
    from lc_classifier.features.core.base import save_astro_objects_batch as save_batch

    # Extract data_num from the input filename (astro_objects_batch_1_0.pkl -> data_num = 1)
    data_num = int(ao_filename.split(".")[0].split("_")[-2])
    
    output_filename = os.path.join(
        output_folder, f"astro_objects_batch_{shorten_n_days}_{data_num}_{batch_id:04d}.pkl"
    )

    if skip_if_output_exists and os.path.exists(output_filename):
        return

    batch_astro_objects = pd.read_pickle(os.path.join(folder, ao_filename))
    batch_astro_objects = [astro_object_from_dict(d) for d in batch_astro_objects]

    lightcurve_preprocessor = ZTFLightcurvePreprocessor()
    lightcurve_preprocessor.preprocess_batch(batch_astro_objects,progress_bar=True)
    if shorten_n_days is not None:
        shorten_preprocessor = ShortenPreprocessor(shorten_n_days)
        shorten_preprocessor.preprocess_batch(batch_astro_objects)

    feature_extractor = ZTFFeatureExtractor()
    feature_extractor.compute_features_batch(batch_astro_objects, progress_bar=True)

    save_batch(batch_astro_objects, output_filename)


def patch_features(batch_id, shorten_n_days=None):
    import pandas as pd
    from typing import List

    from lc_classifier.features.core.base import (
        FeatureExtractorComposite,
        FeatureExtractor,
    )
    from lc_classifier.features.extractors.tde_extractor import TDETailExtractor
    from lc_classifier.features.core.base import astro_object_from_dict
    from lc_classifier.features.core.base import save_astro_objects_batch as save_batch


    # For patch_features, we need to determine data_num from existing files or use a default
    # Let's look for the pattern in the output folder
    data_num = 1  # Default to 1, but we can make this dynamic if needed
    
    # Use the same filename format as extract_features
    filename = os.path.join(
        output_folder, f"astro_objects_batch_{shorten_n_days}_{data_num}_{batch_id:04d}.pkl"
    )
    
    # Check if the file exists before trying to read it
    if not os.path.exists(filename):
        print(f"Warning: File {filename} does not exist, skipping...")
        return

    batch_astro_objects = pd.read_pickle(filename)
    batch_astro_objects = [astro_object_from_dict(d) for d in batch_astro_objects]
    # Assuming light curve is preprocessed and shortened

    # Delete old features to be patched
    features_to_be_patched = ["TDE_decay", "TDE_decay_chi"]

    for ao in batch_astro_objects:
        features = ao.features
        features = features[~features["name"].isin(features_to_be_patched)]
        ao.features = features

    class PatchExtractor(FeatureExtractorComposite):
        version = "1.0.0"

        def _instantiate_extractors(self) -> List[FeatureExtractor]:
            bands = list("gr")

            feature_extractors = [
                TDETailExtractor(bands),
            ]
            return feature_extractors

    feature_extractor = PatchExtractor()
    feature_extractor.compute_features_batch(batch_astro_objects, progress_bar=True)

    save_batch(batch_astro_objects, filename)


shorten_n_days = None  # Set to None if you want to skip shortening

# First run: Extract features
print("Running initial feature extraction...")
tasks = []
for ao_filename in tqdm(astro_objects_filenames, desc="Extracting features"):
    # Parse filename format: astro_objects_batch_1_0.pkl -> batch_id = 0
    batch_id = int(ao_filename.split(".")[0].split("_")[-1])
    tasks.append(delayed(extract_features)(batch_id, ao_filename, shorten_n_days))

Parallel(n_jobs=12, verbose=0, backend="loky")(tasks)

# Second run: Patch features
print("Running feature patching...")
tasks = []
for ao_filename in astro_objects_filenames:
    # Parse filename format: astro_objects_batch_1_0.pkl -> batch_id = 0
    batch_id = int(ao_filename.split(".")[0].split("_")[-1])
    tasks.append(delayed(patch_features)(batch_id, shorten_n_days))

Parallel(n_jobs=9, verbose=0, backend="loky")(tasks)
