# RL_Transmission_Control

Open source code for paper: "Distributed Transmission Control for Wireless Networks using Multi-Agent Reinforcement Learning
" https://arxiv.org/abs/2205.06800

## Problem Description
A multi-agent reinfocement learning (MARL) problem where agents decide if and when to transmit in a highly abstracted wireless network setting. A threshold, _k_, is defined such that only _k_ or fewer agents can transmit successfully on the same time step. Given the level of abstraction, our environment and approach may be applied to other cooperative MARL problems where only a limited number of agents can take the same action on the same step without incurring a reward penalty.

## Description of Files
* `custom_env.py` is the custom environment built in OpenAI Gym
* `agent.py` is where agents are defined and actions are taken
* `argparse_agent.py` allows for command line arguments and can be used with `driver.sh` for automating multiple experiments
* `DQN.py` contains the code for Deep Q-Network algorithm 
* `ReplayMemory.py` contains the code for the experience replay memory for the DQN agents
* `CSMA_agent.py` contains the code for the CSMA algorithms used for benchmarking
