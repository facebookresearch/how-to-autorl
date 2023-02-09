## Tuning Brax

We tune a PPO agent on the brax environments 'ant', 'humanoid' and 'halfcheetah'.
To start tuning, use the provided bash scripts with the environment name, number of timesteps and in the case of DEHB minimum budget (we used 1/100 of the total steps). 
An example of our settings:
```bash
sbatch launch_bgt.sh ant 30000000
sbatch launch_rs.sh humanoid 50000000
sbatch launch_dehb.sh halfcheetah 100000000 1000000
```
SMAC is an exception, you won't need to provide any arguments, just run the corresponding script.

Once you have the tuning result, add each configuration as a configuration file in the 'configs' directory and run it across our 10 test seeds as follows:
```bash
sh evaluate_incumbent.sh <your-config-name>
```