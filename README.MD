# Structure

This project contains 2 versions of ADAGA, the algorithm for change point detection presented in "Adaptive Gaussian Process Change Point Detection", by Edoardo Caldarelli, Philippe Wenk, Stefan Bauer, and Andreas Krause (Proceedings of the 39th International Conference on Machine Learning, Baltimore, Maryland, USA, PMLR 162, 2022). The directory ```inducing_points_version``` contains the code implemented with the
inducing points, and ```qff_version``` contains the one implemented with quadrature
Fourier features (and the exact linear kernel). This demo uses pip version 20.2.4.

The directory 
```time_series``` contains the 6 real-world and 3 synthetic time series used in our experiments.

# Create the virtual environment

The virtual environment for running the project can be created by running these commands, in the main directory of the 
project ``code``. Firstly, we create the virtual environment via ```venv```:

```bash
python3 -m venv env python=3.7
```

Then we activate the environment:
```bash
source env/bin/activate
```
Now, we install the required packages:
```bash
pip install -r requirements.txt
```

# Sources

### Code
The implementation of GP regression with QFFs is based on code implemented by Emmanouil Angelis, 
ETH Zurich (Angelis et al., "SLEIPNIR: Deterministic and Provably Accurate Feature Expansion for Gaussian Process 
Regression with Derivatives", 2020).

### Datasets
The 6 real-world time series datatsets
were downloaded from https://github.com/alan-turing-institute/TCPD.
# Change point detection

If we want to reproduce our change point detection experiments, we can run the following steps.

Firstly, we must extract the 6 real-world datasets used in our experiment. To do so, 
we can run the script  ```extract_time_series.py```. We have to make sure that, at this step, we are in the ```code```directory:
```bash
python -m extract_time_series
```

Then, we can generate the 3 synthetic series, also used in our experiments, by using the script ```generate_simulated_cp_data.py```. Note that this command creates 10 different noisy realizations of each series. We have to make sure that we are still in the ```code``` directory:

```bash
python -m generate_simulated_cp_data
```

The extracted time series' values (```obs.csv```), along with the timesteps (```tsteps.csv```),
are saved, e.g., at the path ```./time_series/run_log``` for the Run log series, or ```./time_series/mean_0``` for the first noisy realization of synthetic series with 2 change points in the mean.

### Inducing points 

If the inducing points are used, we select the inducing point approximation, from the ```code```directory: 

```bash
cd inducing_points_version
```

We can now partition the desired series. For instance, 
```bash
python -m regionalize_time_series --dataset "run_log" --kernel "Linear"
```

regionalizes the Run Log time series with the parameters used in our experiments.

The real-world datasets available are ```run_log``` , ```businv``` , ```gdp_japan```, ```gdp_argentina```, ```gdp_iran```. 

The synthetic datasets available are ```mean``` (with change points in the mean), ```var```  (with change points in the noise variance), and ```per``` (with change points in the temporal correlation of the samples.

Valid kernels are ```Linear``` (for ```businv``` and ```run_log``` datasets), and ```RBF```, ```RQ```, ```Matern52```, ```Periodic``` (for the remaining datasets).

### QFFs (and exact linear kernel)

If the QFFs (or the exact linear kernel) are used, we select the QFF approximation, from the ```code```directory: 

```bash
cd qff_version
```

Then, we proceed as before. The command 

```bash
python -m regionalize_time_series --dataset "run_log" 
```

regionalizes the Run log time series with the parameters used in our experiments.

Note that, in our paper, the exact linear kernel is used with the ```businv``` and ```run_log``` datasets only. Conversely, all the other datasets are processed with the RBF kernel only (approximated with QFFs). Thus, in this case, we do not state the kernel to be used in the processing in the shell command.

### ADAGA's output

The information about the start and end of the regions is saved as a list of dictionaries at
```"inducing_points_version/regions_time_series"```, as ```.npy```files. Each element in the list corresponds to one region.

If the QFFs are used, the results' directory is ```"qff_version/regions_time_series"```.