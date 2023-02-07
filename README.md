# ICML Submission 'Hyperparameters in RL and How to Tune Them'

This repository contains both the code to reproduce our experiments (see 'sb3' for the experiments from Section 4 and 'sota' for Section 5) as well as the hydra sweeper implementations of PBT and DEHB we plan to release.
In order to run our experiments, please refer to the bash scripts in each directory, they either contain information on the sweeps we ran or how to start our tuning runs. 
We recommend using the conda environment provided for all of them. Install and load it via:

```bash
conda env create -f environment.yml
conda activate autorl
```

In case you're interested in testing out the sweepers, start in the examples directory for more lightweight experiments.
Assume that everything here is a *minimizer*! You can maximize instead by passing 'maximize=true' as a sweeper kwarg.

Open TODOs:
- Update/write docstrings
- explore if lazy imports can even work and if not how to deal with all the dependencies

## Usage in your own project
In your yaml-configuration file, set `hydra/sweeper` to the sweeper name, e.g. `DEHB`:
```yaml
defaults:
  - override hydra/sweeper: DEHB
```
You can also add `hydra/sweeper=<sweeper_name>` to your command line.
The sweepers will only be found if the `hydra_plugins` directory is in your PYTHONPATH. You can check if it's loaded by running your script with `--info plugins`.

## Hyperparameter Search Space
The definition of the hyperparameters is based on [ConfigSpace](https://github.com/automl/ConfigSpace/).
The syntax of the hyperparameters is according to ConfigSpace's json serialization.
Please see their [user guide](https://automl.github.io/ConfigSpace/master/User-Guide.html)
for more information on how to configure hyperparameters.

Your yaml-configuration file must adhere to following syntax:
```yaml
hydra:
  sweeper:
    ...
    search_space:
      hyperparameters:  # required
        hyperparameter_name_0:
          ...
        hyperparameter_name_1:
          ...
        ...

```
The configspace fields `conditions` and `forbiddens` aren't implemented for the solvers (except possibly DEHB?). Please don't use them, they'll be ignored.

Defining a uniform integer parameter is easy:
```yaml
n_neurons:
  type: uniform_int  # or have a float parameter by specifying 'uniform_float'
  lower: 8
  upper: 1024
  log: true  # optimize the hyperparameter in log space
  default_value: ${n_neurons}  # you can set your default value to the one normally used in your config
```
Same goes for categorical parameters:
```yaml
activation:
  type: categorical
  choices: [logistic, tanh, relu]
  default_value: ${activation}
```
