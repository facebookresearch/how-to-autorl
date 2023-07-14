# AutoRL Hydra Sweepers

This repository contains hydra sweeper versions of proven AutoRL tuning tools for pluag-and-play use. 
Currently included:
- [Differential Evolution Hyperband](https://arxiv.org/pdf/2105.09821.pdf)
- Standard [Population Based Training](https://arxiv.org/pdf/1711.09846.pdf) (with warmstarting option)
- [Population Based Bandits](https://arxiv.org/pdf/2002.02518.pdf) (with Mix/Multi versions and warmstarting option)
- [Bayesian-Generational Population Based Training](https://arxiv.org/pdf/2207.09405v1.pdf)

We recommend starting in the examples directory to see how the sweepers work.
Assume that everything here is a *minimizer*! You can maximize instead by passing 'maximize=true' as a sweeper kwarg.
For more background information, see [here](docs/index.md).

## Installation
We recommend creating a conda environment to install the sweeper in. Choose which you want to use and install the dependencies for these sweepers. For all available options, use 'all' or:
```bash
conda create -n autorl-sweepers python==3.9
conda activate autorl-sweepers
pip install -e .[dehb,pb2,bgt]
```

If you want to work on the code itself, you can also use:
```bash
make install-dev
```

## Examples
In ['examples'](examples) you can see example configurations and setups for all sweepers on Stable Baselines 3 agents.
To run an example with the sweeper, you need to set the '--multirun' flag:
```bash
python examples/dehb_for_pendulum_ppo.py -m
```

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

## Contribute
See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
This package is Apache 2.0 licensed, as found in the LICENSE file.