"""
This script extracts the data of our 6 real-world datasets to be used in our changepoint detection experiment.

Edoardo Caldarelli, ETH Zurich
"""
import numpy as np
import pathlib
import json

datasets = ["run_log", "businv", "gdp_iran", "gdp_argentina", "gdp_japan"]
for dataset in datasets:
    with open(f"./time_series/{dataset}/{dataset}.json") as f:
        dict = json.load(f)
    path_new = pathlib.Path(f"./time_series/{dataset}")
    path_new.mkdir(parents=True, exist_ok=True)
    x = np.array(dict["time"]["index"]).reshape((-1, 1))
    y = np.array(dict["series"][0]["raw"]).reshape((-1, 1)) if dataset != "run_log"\
        else np.array(dict["series"][1]["raw"]).reshape((-1, 1)) 
    np.savetxt(f"{path_new}/tsteps.csv", x, delimiter=",")
    np.savetxt(f"{path_new}/obs.csv", y, delimiter=",")


