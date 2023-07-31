# [IEEE Transactions on Power Systems] Transmission Interface Power Flow Adjustment: A Deep Reinforcement Learning Approach based on Multi-task Attribution Map

[![License: Apache](https://img.shields.io/badge/License-Apache-blue.svg)](LICENSE)

Official codebase for paper [Transmission Interface Power Flow Adjustment: A Deep Reinforcement Learning Approach based on Multi-task Attribution Map]([url](https://ieeexplore.ieee.org/abstract/document/10192091)https://ieeexplore.ieee.org/abstract/document/10192091). This codebase is based on the open-source [tianshou](https://github.com/oxwhirl/pymarl) framework and please refer to that repo for more documentation.


## Overview

**TLDR:** This work is therefore the first dedicated attempt towards learning multiple transmission interface power flow adjustment tasks jointly, a highly practical problem yet largely overlooked by existing literature in the field of the power system. We design a novel deep reinforcement learning (DRL) method based on multi-task attribution map (MAM) to handle multiple adjustment tasks jointly, where MAM enables the DRL agent to selectively integrate the node features into a compact task-adaptive representation for the final adjustment policy. Simulations are conducted on the IEEE 118-bus system, a realistic 300-bus system in China, and a very large European 9241-bus system, demonstrating that the proposed method brings remarkable improvements to the existing methods. Moreover, we verify the interpretability of the learnable MAM in different operation scenarios.


![image](https://github.com/Cra2yDavid/MAM/blob/main/framework.png)

## Prerequisites

#### Install dependencies

See `requirment.txt` file for more information about how to install the dependencies.


## Usage

Please follow the instructions below to replicate the results in the paper.

```bash
# IEEE 118-bus System
python train.py
```



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
