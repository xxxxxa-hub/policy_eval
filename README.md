# Off-policy Evaluation Library

This library was originally written to implement the continuous control
experiments in `Statistical Bootstrapping for Uncertainty Estimation in
Off-Policy Evaluation' by Ilya Kostrikov and Ofir Nachum. Beyond implementing a
bootstrapped version of FQE and MB, which was the focus of that paper, it also
includes a generic implementation of DualDICE as well as weighted per-step
importance sampling and doubly robust estimators.

Paper available on arXiv [here](https://arxiv.org/abs/2007.13609).

If you use this codebase for your research, please cite the paper:

```
@article{kostrikov2020statistical,
         title={Statistical Bootstrapping for Uncertainty Estimation in Off-Policy Evaluation},
         author={Ilya Kostrikov and Ofir Nachum},
         journal={arXiv preprint arXiv:2007.13609},
         year={2020},
}
```

The code allows for generating offline datasets (as used for the original paper)
as well as using D4RL datasets and target policies, as used for the benchmarking
paper `Benchmarks for Deep Off-Policy Evaluation' by Fu, et al.

## Installation
d3rlpy can be installed by cloning the repository as follows:
```
git clone https://github.com/xxxxxa-hub/policy_eval.git
cd policy_eval
pip install -e .
```

## Basic Commands
In this repository, we mainly use Importance Sampling and Model-based method for off-policy evaluation. For Importance Sampling, we need to do behavior cloning on offline dataset. Similarly, we need to fit the dynamics model for Model-based method. For each task, the behavior or dynamics model should be pretrained once and can be used for evaluation for multiple times. The pretraining is done as follows:
```
CUDA_VISIBLE_DEVICES=0 python -m policy_eval.train --env_name=Pendulum-replay --save_dir=/path/to/dir --target_policy_std=0.0 --seed=0 --algo=iw --noise_scale=0.0 --lr=0.01 --lr_decay=0.8 
```

With the pretrained model, we do off-policy evaluation as follows:
```
CUDA_VISIBLE_DEVICES=0 python -m policy_eval.eval --env_name=Pendulum-replay --save_dir=/path/to/dir --d4rl_policy_filename=/path/to/policy.pkl --target_policy_std=0.0 --seed=0 --algo=iw --noise_scale=0.0 --lr=0.003 --lr_decay=0.8 
```