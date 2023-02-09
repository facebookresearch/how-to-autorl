## Tuning IDAAC

We tune IDAAC on ProcGen. The 'launch' scripts will start 3 tuning runs across 5 seeds each using slurm if you provide the name of the game and clf size (4 for most games, see the 'idaac/hyperparams.py' file for details). To e.g. launch PB2 for climber, run:
```bash
sbatch launch_pb2.sh climber 64
```
SMAC is an exception, you won't need to provide any arguments, just run the corresponding script.

Once you have the tuning result, add each configuration as a configuration file in the 'configs' directory and run it across our 10 test seeds as follows:
```bash
sh evaluate_incumbent.sh <your-config-name>
```