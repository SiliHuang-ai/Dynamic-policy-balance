# Dynamic policy balance for multi-agent reinforcement learning
Double blind in AAAI2022

This repository implements DPB. The implementation in this repositorory is used in the work "Dynamic policy balance for multi-agent reinforcement learning".
This repository is heavily based on https://github.com/marlbenchmark/on-policy and https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail.

## Environments supported:

- [StarCraftII (SMAC)](https://github.com/oxwhirl/smac)

## 1. Usage

All core code is located within the Dynamic_policy_balance folder.

* The envs/ subfolder contains environment wrapper implementations for the SMAC.

* The utils/config.py file contains relevant hyperparameter and env settings.

## Run an experiment
```shell
python3 main.py
```

## 2. Installation
Here we give an example installation on CUDA == 10.1. For non-GPU & other CUDA version installation, please refer to the [PyTorch website](https://pytorch.org/get-started/locally/).

``` Bash
# create conda environment
conda create -n marl python==3.6.1
conda activate marl
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

Even though we provide requirement.txt, it may have redundancy. We recommend that the user try to install other required packages by running the code and finding which required package hasn't installed yet.

### 2.1 Install StarCraftII [4.10](http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip)

   

``` Bash
unzip SC2.4.10.zip
# password is iagreetotheeula
echo "export SC2PATH=~/StarCraftII/" > ~/.bashrc
```

* download SMAC Maps, and move it to `~/StarCraftII/Maps/`.

* To use a stableid, copy `stableid.json` from https://github.com/Blizzard/s2client-proto.git to `~/StarCraftII/`.


