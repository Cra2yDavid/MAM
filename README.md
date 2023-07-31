# [IEEE Transactions on Power Systems] Transmission Interface Power Flow Adjustment: A Deep Reinforcement Learning Approach based on Multi-task Attribution Map

[![License: Apache](https://img.shields.io/badge/License-Apache-blue.svg)](LICENSE)

Official codebase for paper [Transmission Interface Power Flow Adjustment: A Deep Reinforcement Learning Approach based on Multi-task Attribution Map](https://ieeexplore.ieee.org/abstract/document/10192091). This codebase is based on the open-source [Tianshou](https://github.com/thu-ml/tianshou) and [PandaPower](https://github.com/e2nIEE/pandapower) framework and please refer to those repo for more documentation. Baseline methods include [Soft-Module](https://github.com/RchalYang/Soft-Module) and traditional full-connected neural network.

A novel approach named as [ASF](https://github.com/Cra2yDavid/ASF) is recently proposed to solve the same task as an enhancement to MAM. 


## Overview

**TLDR:** This work is the first dedicated attempt towards learning multiple transmission interface power flow adjustment tasks jointly, a highly practical problem yet largely overlooked by existing literature in the field of the power system. We design a novel deep reinforcement learning (DRL) method based on multi-task attribution map (MAM) to handle multiple adjustment tasks jointly, where MAM enables the DRL agent to selectively integrate the node features into a compact task-adaptive representation for the final adjustment policy. Simulations are conducted on the IEEE 118-bus system, a realistic 300-bus system in China, and a very large European 9241-bus system, demonstrating that the proposed method brings remarkable improvements to the existing methods. Moreover, we verify the interpretability of the learnable MAM in different operation scenarios.


![image](https://github.com/Cra2yDavid/MAM/blob/main/framework.png)

## Prerequisites

### Install dependencies
* Python 3.8.13 or higher
* dgl 1.1 or higher
* Pytorch 1.13
* Pandapower 2.11
* gym 0.23
* tianshou 0.4.11


## Usage

Please follow the instructions below to replicate the results in the paper. Note that the model of China realistic 300-bus system is not available due to confidentiality policies of SGCC.

```bash
# IEEE 118-bus System under S10 (Single 10-Interface) task
python train.py --case='case118' --task='S10' --method='MAM' --model='Attention'
```
![image](https://github.com/Cra2yDavid/MAM/blob/main/exp.png)


## Citation

If you find this work useful for your research, please cite our paper:

```
@article{liu2023MAM,
  author={Liu, Shunyu and Luo, Wei and Zhou, Yanzhen and Chen, Kaixuan and Zhang, Quan and Xu, Huating and Guo, Qinglai and Song, Mingli},
  journal={IEEE Transactions on Power Systems}, 
  title={Transmission Interface Power Flow Adjustment: A Deep Reinforcement Learning Approach Based on Multi-Task Attribution Map}, 
  year={2023},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/TPWRS.2023.3298007}
}
```

## Contact

Please feel free to contact me via email (<liushunyu@zju.edu.cn>, <davidluo@zju.edu.cn>) if you are interested in my research :)
