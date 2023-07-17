# Documentation for the Hydra AutoRL Sweepers

This documentation aims to provide basic information on the sweepers contained in this project as well as how to use them in your project. 
For in more information, we recommend the algorithms' original papers and our examples.

## Why Tune RL Hyperparameters?
RL has been shown to be fairly sensitive to its hyperparameters and many papers thus report running sweeps or grid searches across their hyperparameters [0](https://arxiv.org/pdf/1912.06680.pdf)[1](https://arxiv.org/pdf/2102.10330.pdf)[2](https://arxiv.org/pdf/2211.00539.pdf). 
This is pretty inefficient, however, as we know grid search does [not scale well at all](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf).
Therefore we can invest the compute budget (and often human time) better by using methods that have shown to be both efficient and well-performing on RL problems - like for example DEHB and PBT variations.
We show this in our [ICML'23 paper](https://arxiv.org/pdf/2306.01324.pdf).

## Our Tuning Algorithms
[DEHB](https://arxiv.org/pdf/2105.09821.pdf) is a simple but effective multi-fidelity tuning method based on [HyperBand](https://arxiv.org/pdf/1603.06560.pdf) and [Differential Evolution](https://link.springer.com/article/10.1023/a:1008202821328). 
Multi-fidelity means it will evaluate the algorithm at different budgets (most often different runtimes, discarding configurations that perform poorly already in the beginning of training) in order to be able to explore more hyperparameter configurations for less computing costs.
It will run several HyperBand brackets where the starting point is randomly sampled in the first bracket and then mutated for the next one using DE.

[PBT](https://arxiv.org/pdf/1902.01894.pdf) is stands for Population-Based Training where a population of agents is trained in parallel. 
Their configurations are updated during training after a configuration interval of steps.
There are several options of how to perform this update, we provide standard PBT (which simply modulates the current value up or down by a fixed value), [PB2](https://arxiv.org/pdf/2106.15883.pdf) in its Mix and Mult variations (where BO takes care of configuration selection) and [BG-PBT](https://arxiv.org/pdf/2207.09405.pdf) (also using BO, but with a specialized kernel and optional architecture search). 
There's also the option of using a number of full warmup runs at the start of training to pre-select the inital configurations from a larger set of runs.

Note that if you want to use the architecture search option of BG-PBT, you'll need to provide a distillation loop from one architecture to another.

## Installation
We recommend creating a conda environment to install the sweeper in:
```bash
conda create -n autorl-sweepers python==3.9
conda activate autorl-sweepers
pip install -e .[all] 
```

## Sweeper Usage in Hydra
A sweeper will automatically start a range of jobs according to some specification with a single command.
Sweepers are activated whenever the multirun flag is set:

```bash
# Standard run
python example_script.py

# Sweeper is used
python example_script.py --multirun
```

You can specify all options either in the yaml file or on the command line, the keywords are the same either way.
We recommend using the config files as much as possible, however, as it is usually easier to keep track of the full settings this way.

For examples of config files, please see our 'examples' directory. 
Important to note are the launcher and sweeper overrides at the top of the file that will decide where jobs are launched (locally or on a cluster) as well as which sweeper is used.

For our sweepers, there are some common arguments that are important to think about:
1. search_space: this should point to a search space config, e.g. in the form of another yaml file, that specifies types and limits of the hyperparameters you want to tune
2. resume: if a path to a tuning checkpoint is provided, the checkpoint will be loaded and tuning will continue from there
3. n_jobs: this is the limit of how many jobs to launch at a time. This is important if you want to e.g. only ever want to use 5 GPUs at a time for tuning.
4. budget_variable: the name of the argument that control how long your agent is training (e.g. num_steps). This is used to control the fidelity in each sweeper.
5. dehb_kwargs/pbt_kwargs: here the sweepers themselved can be configured.

## The Sweepers on Slurm Clusters
You can run all sweepers locally and sequentially using the submitit local launcher, but to run everything efficiently, you'll want to parallelize the runs.
The easiest way to do this is to use the submitit slurm launcher for hydra which will then automatically submit job arrays to the slurm cluster.

To optimize scheduling, we recommend providing an estimate of the runtime of a full budget run to the sweeper. 
The sweeper will then scale down the requested time and this will help your jobs be scheduled quicker.
You can do this in the sweeper kwargs by setting 'slurm=True' and 'slurm_timeout' to the timeout value in minutes. For DEHB, this would look like this:
```bash
python <your_script> --multirun +hydra.sweeper.dehb_kwargs.slurm=true +hydra.sweeper.dehb_kwargs.slurm_timeout=60
```

## Integrations
We support logging to [Weights & Biases](https://wandb.ai) as well as [DeepCave](https://github.com/automl/DeepCAVE). 
Both offer a way to monitor tuning, though W&B does this online and DeepCave is usually run locally.
Additionally, DeepCave offers plugins specifically for AutoML rundata analysis that might be helpful, e.g. hyperparameter importance.

To enable W&B logging, you need to provide your project name and optionally any tags you want to include in you run in the sweeper kwargs. As an example with DEHB:
```bash
python <your_script> --multirun +hydra.sweeper.dehb_kwargs.wandb_project=<project-name> +hydra.sweeper.dehb_kwargs.wandb_entity=<entity> +hydra.sweeper.dehb_kwargs.wandb_tag=[<tag>] 
```

DeepCave is activated, e.g. for PBT:
```bash
python <your_script> --multirun +hydra.sweeper.pbt_kwargs.deepcave=true
```
The DeepCave logs can be found in the sweep directory.

## A Typical HPO Workflow

To start with, you'll want to define all the important details about your setting:
- which environments do you want to tune on?
- how many seeds will you use?
- how much budget do you have?
- which hyperparameters do you want to tune?

You should also already define a corresponding test setting to see how your hyperparameters perform in practice. 
The you'll start the optimization for **all** methods you want to report (unless you're sure your baselines are already well tuned using a similar budget). Once you have the results, test them on the test setting.

Often you'll want some additional insights into the hyperparameters after the fact since you might want to measure hyperparameter sensitivity, importance of dependencies between hyperparameters. The easiest way of doing this, is to use DeepCave logging in the sweepers and then check out its insights.

## Best Practices

This is an abbreviated list with the most important points to remember. Our [checklist](../checklist.pdf) provides a more complete overview for reporting purposes.

**1. Tune all hyperparameters that could be relevant**
RL is a highly dynamic and quite complex optimization process where many hyperparameters can influence the result. 
If you're not sure which ones are important, err on the side of including more rather than less in your search space.
If you want to know more about your hyperparameter importances, log your tuning runs using the DeepCave reporting and use the hyperparameter importance analysis plugin in DeepCave to gain more insights.

**2. Tune across multiple seeds, then report across different test seeds**
Tuning can absolutely overfit to the random seed and the performance differences between tuning and test seeds can be immense. 
If you're training a model for research purposes, usually we want to gain general performance insights, so we recommend reporting on separate test seeds.

**3. Document your tuning and reporting process**
This includes budgets, exact seeds, search spaces, etc. so your experiments can be repoduced by others. If you're using our sweepers, you would ideally provide the sweep config and the final config together with the commands to run them (that include tuning and test seeds).