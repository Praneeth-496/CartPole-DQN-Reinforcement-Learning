# CartPole DQN Reinforcement Learning

This repository contains the implementation of Deep Q-Networks (DQN) for solving the CartPole-v1 environment from OpenAI Gymnasium. The project explores different DQN configurations and conducts experiments to analyze their performance.

## Project Overview

This project implements and evaluates four different DQN configurations:
- **Naive**: Basic Q-learning without replay buffer or target network
- **Only Target Network (only_tn)**: Q-learning with a target network but no replay buffer
- **Only Experience Replay (only_er)**: Q-learning with a replay buffer but no target network
- **Target Network + Experience Replay (tn_er)**: Full DQN implementation with both target network and replay buffer

The code includes comprehensive experiments, grid search for hyperparameter optimization, and ablation studies to analyze the impact of different components and parameters.

## Features

- Implementation of DQN with various configurations
- Vectorized environments for faster training
- Hyperparameter grid search
- Ablation studies on exploration parameters
- Comprehensive visualization of learning curves and performance metrics
- Detailed logging of training progress

## Requirements

The project requires the following Python packages:
```
gymnasium>=0.26.0
torch>=2.0.0
numpy>=1.21.0
matplotlib>=3.4.0
pandas>=1.3.0
```

You can install all dependencies using:
```bash
pip install -r requirements.txt
```

## Usage

To run the main experiments:

```bash
python main.py
```

This will execute:
1. Baseline experiments comparing all four DQN configurations
2. Grid search over learning rates, update ratios, and network sizes
3. Ablation study on exploration parameters
4. Q-learning curve plotting

## Code Structure

- `main.py`: Main script containing all implementations and experiments
- `QNetwork`: Neural network class for approximating Q-values
- `ReplayBuffer`: Experience replay buffer implementation
- `train_cartpole_dqn`: Main training function with configurable parameters
- Various experiment functions for different analyses

## Results

The experiments generate several plots in the `plots/` directory:
- Learning curves comparing different DQN configurations
- Final performance comparison bar charts
- Grid search learning curves and performance metrics
- Ablation study results on exploration parameters

## Author

Praneeth Dathu

## License

This project is available under the MIT License.

