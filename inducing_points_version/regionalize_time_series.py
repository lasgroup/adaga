"""
This script performs our CP detection experiments on real and simulated datasets.

Edoardo Caldarelli, ETH Zurich
"""

import numpy as np
import gpflow
from core.adaptive_regionalization import AdaptiveRegionalization
import tensorflow as tf
import json
import pathlib
import argparse

seed = 374786
np.random.seed(seed)
tf.set_random_seed(seed)

delta = 0.6
num_inducing_points = 10
jitter = 1e-8
custom_config = gpflow.settings.get_settings()
custom_config.numerics.jitter_level = jitter

custom_config = gpflow.settings.get_settings()
custom_config.numerics.jitter_level = jitter
custom_config.float_type = tf.float64

path_new = pathlib.Path(f"./regions_time_series")

path_new.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="mean",
                    help="The time series to be processed.")
parser.add_argument("--batch_size", type=int, default=1,
                    help="The number of points in a batch.")
parser.add_argument("--kernel", type=str, default="RBF")
args = parser.parse_args()

dataset = args.dataset
batch_size = args.batch_size

x = np.loadtxt(f"../time_series/{dataset}/tsteps.csv").reshape((-1, 1)).astype(np.float)
y = np.loadtxt(f"../time_series/{dataset}/obs.csv").reshape((-1, 1)).astype(np.float)

min_window_size = 15
kernel = args.kernel
with gpflow.settings.temp_settings(custom_config):

    regionalization = AdaptiveRegionalization(domain_data=x,
                                              system_data=y,
                                              delta=delta,
                                              min_w_size=min_window_size,
                                              n_ind_pts=num_inducing_points,
                                              seed=seed,
                                              batch_size=batch_size,
                                              kern=kernel)
    regionalization.regionalize()
    regions = regionalization.closed_windows
np.save(f"{path_new}/regions_{dataset}_batch_size_{batch_size}_delta_{delta}_min_w_size_{min_window_size}.npy", regions)
