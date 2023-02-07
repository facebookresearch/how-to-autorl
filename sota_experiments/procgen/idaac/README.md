# IDAAC: Invariant Decoupled Advantage Actor-Critic
This is a PyTorch implementation of the methods proposed in

[**Decoupling Value and Policy for Generalization in Reinforcement Learning**](https://arxiv.org/abs/2102.10330) by 

Roberta Raileanu and Rob Fergus.


# Citation
If you use this code in your own work, please cite our paper:
```
@article{Raileanu2021DecouplingVA,
  title={Decoupling Value and Policy for Generalization in Reinforcement Learning},
  author={Roberta Raileanu and R. Fergus},
  journal={ArXiv},
  year={2021},
  volume={abs/2102.10330}
}
```


# Requirements
To install all the required dependencies: 
```
conda create -n idaac python=3.7
conda activate idaac

cd idaac
pip install -r requirements.txt

pip install procgen

git clone https://github.com/openai/baselines.git
cd baselines 
python setup.py install 
```


# Instructions 
This repo provides instructions for training IDAAC, DAAC, and PPO on the Procgen benchmark. 

## Train IDAAC on CoinRun
```
python train.py --env_name coinrun --algo idaac
```

## Train DAAC on CoinRun
```
python train.py --env_name coinrun --algo daac
```

## Train PPO on CoinRun
```
python train.py --env_name coinrun --algo ppo --ppo_epoch 3
```

Note: The default code uses the same set of hyperparameters (HPs) for all environments, which are the best ones overall. 
In our studies, we've found some of the games can further benefit from slightly different HPs, so we provide those as well. To use the best hyperparameters for each environment, use the flag `--use_best_hps`. 


# Overview of DAAC and IDAAC

![IDAAC Overview](/figures/idaac_overview.png)


# Procgen Results 
**IDAAC** achieves state-of-the-art performance on the [Procgen benchmark](https://openai.com/blog/procgen-benchmark/) (easy mode), significantly improving the agent's generalization ability over standard RL methods such as PPO.  

Test Results on Procgen

![Procgen Test Results](/figures/idaac_procgen_test.png)


# Acknowledgements
This code was based on an open sourced [PyTorch implementation of PPO](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail).

