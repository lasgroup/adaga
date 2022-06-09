"""
Edoardo Caldarelli, ETH Zurich
"""

import numpy as np
import pathlib

for seed in range(0, 10):
    np.random.seed(seed)

    path_new = pathlib.Path(f"./time_series")

    path_new.mkdir(parents=True, exist_ok=True)

    t = np.arange(0.0, 30, 0.4).reshape([-1, 1])
    y_base = np.sin(0.5 * t).reshape([-1, 1])

    # No CPs

    noise = np.random.normal(0, 1e-1, (y_base.shape[0], 1))
    y_nocp = y_base + noise

    path_new = pathlib.Path(f"./time_series/no_cps_{seed}")

    path_new.mkdir(parents=True, exist_ok=True)

    np.savetxt(f"{path_new}/tsteps.csv", np.arange(0, np.shape(y_nocp)[0]))
    np.savetxt(f"{path_new}/obs.csv", y_nocp)


    # CP in mean
    size_1 = 20
    size_2 = 29
    size_3 = y_base.shape[0] - (size_1 + size_2)

    y_cp_mean = np.concatenate((y_base[:size_1, :], y_base[size_1:size_1 + size_2, :] + 2), axis=0)
    y_cp_mean = np.concatenate((y_cp_mean, y_base[size_1 + size_2:size_1 + size_2 + size_3, :] - 1), axis=0)
    y_cp_mean = np.concatenate((y_cp_mean, y_base[size_1 + size_2 + size_3:, :] + 0.5), axis=0)

    y_cp_mean += noise

    path_new = pathlib.Path(f"./time_series/mean_{seed}")

    path_new.mkdir(parents=True, exist_ok=True)

    np.savetxt(f"{path_new}/tsteps.csv", np.arange(0, np.shape(y_cp_mean)[0]))
    np.savetxt(f"{path_new}/obs.csv", y_cp_mean)

    # CP in variance

    size_1 = 23
    size_2 = 21
    size_3 = y_base.shape[0] - (size_1 + size_2)
    noise_1 = noise[:size_1, :]
    noise_2 = np.random.normal(0, 3e-1, (size_2, 1))
    noise_3 = np.random.normal(0, 0.8e-1, (size_3, 1))

    y_cp_variance = np.concatenate((y_base[:size_1, :] + noise_1, y_base[size_1:size_1 + size_2, :] + noise_2), axis=0)
    y_cp_variance = np.concatenate((y_cp_variance, y_base[size_1 + size_2:size_1 + size_2 + size_3, :] + noise_3), axis=0)

    path_new = pathlib.Path(f"./time_series/var_{seed}")

    path_new.mkdir(parents=True, exist_ok=True)

    np.savetxt(f"{path_new}/tsteps.csv", np.arange(0, np.shape(y_cp_variance)[0]))
    np.savetxt(f"{path_new}/obs.csv", y_cp_variance)

    # CP in periodicity (temporal correlation)

    size_1 = 27
    size_2 = 20
    size_3 = y_base.shape[0] - (size_1 + size_2)
    y_cp_periodicity = y_base[:size_1, :]
    y_new_per = np.sin(0.2 * t[:size_2 + 1] + y_cp_periodicity[-1, 0]).reshape([-1, 1])
    y_cp_periodicity = np.concatenate((y_cp_periodicity[:-1, :], y_new_per), axis=0)
    y_new_per = np.sin(0.6 * t[:size_3 + 1] + y_cp_periodicity[-1, 0]).reshape([-1, 1])
    y_cp_periodicity = np.concatenate((y_cp_periodicity[:-1, :], y_new_per), axis=0)

    y_cp_periodicity += noise

    path_new = pathlib.Path(f"./time_series/per_{seed}")

    path_new.mkdir(parents=True, exist_ok=True)

    np.savetxt(f"{path_new}/tsteps.csv", np.arange(0, np.shape(y_cp_periodicity)[0]))
    np.savetxt(f"{path_new}/obs.csv", y_cp_periodicity)