"""
This script performs our CP detection experiments on real and simulated datasets.

Edoardo Caldarelli, ETH Zurich
"""
import numpy as np
import gpflow
from core.adaptive_regionalization_exact_linear import AdaptiveRegionalization as AdaptiveRegionalizationLinear
from core.adaptive_regionalization import AdaptiveRegionalization as AdaptiveRegionalization
import argparse
import tensorflow as tf
import json
import pathlib

seed = 374786
np.random.seed(seed)
tf.set_random_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="mean",
                    help="The time series to be processed.")
parser.add_argument("--batch_size", type=int, default=1,
                    help="The number of points in a batch.")
args = parser.parse_args()

dataset = args.dataset
batch_size = args.batch_size


delta = 0.6
n_feat = 30
n_min_windows = 12

jitter = 1e-4
custom_config = gpflow.settings.get_settings()
custom_config.numerics.jitter_level = jitter

custom_config = gpflow.settings.get_settings()
custom_config.numerics.jitter_level = jitter
custom_config.float_type = tf.float64

path_new = pathlib.Path(f"./regions_time_series/EXACT_LINEAR")

path_new.mkdir(parents=True, exist_ok=True)
x = np.loadtxt(f"../time_series/{dataset}/tsteps.csv").reshape((-1, 1)).astype(np.float)
y = np.loadtxt(f"../time_series/{dataset}/obs.csv").reshape((-1, 1)).astype(np.float)

n_batches = x.shape[0] / batch_size
min_window_size = 15

regionalization = AdaptiveRegionalization(domain_data=x,
                                          system_data=y,
                                          delta=delta,
                                          min_w_size=min_window_size,
                                          n_batches=n_batches,
                                          n_feat=n_feat,
                                          seed=seed) if dataset != "run_log" and dataset != "businv" else\
                  AdaptiveRegionalizationLinear(domain_data=x,
                                          system_data=y,
                                          delta=delta,
                                          min_w_size=min_window_size,
                                          n_batches=n_batches,
                                          seed=seed)
regionalization.regionalize()
regions = regionalization.closed_windows
np.save(f"{path_new}/regions_{dataset}_batch_size_{batch_size}_delta_{delta}_min_w_size_{min_window_size}.npy", regions)
