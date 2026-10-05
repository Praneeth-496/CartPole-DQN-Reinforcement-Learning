# CartPole DQN Reinforcement Learning

A PyTorch reinforcement learning project investigating how experience replay and target networks affect neural Q-learning on CartPole-v1.

The repository includes four configurations, multi-seed experiments, hyperparameter grid search, and an exploration-decay ablation study.

## Project Overview

The project explores three questions:

1. How do experience replay and target networks affect learning?
2. How sensitive is DQN to learning rate, network size, and update frequency?
3. How does the exploration schedule influence training performance?

## DQN Configurations

| Configuration | Experience replay | Target network | Description |
|---------------|-------------------|----------------|-------------|
| `naive` | No | No | Neural Q-learning from individual transitions |
| `only_tn` | No | Yes | Individual-transition updates with a separate target network |
| `only_er` | Yes | No | Replay-based updates using the online network for targets |
| `tn_er` | Yes | Yes | DQN with both replay and a target network |

The `naive` configuration uses a neural network rather than a tabular Q-learning implementation.

## Implementation

### Q-Network

A configurable multilayer perceptron maps the four-dimensional CartPole observation to Q-values for its two actions.

Default architecture:

- Input: 4 state variables.
- Hidden layers: 128 and 128 units with ReLU.
- Output: 2 Q-values.
- Optimizer: Adam.
- Loss: mean squared temporal-difference error.

### Experience Replay

A bounded replay buffer stores transitions:

```text
(state, action, reward, next_state, done)
```

Replay-enabled configurations sample random minibatches for training.

### Target Network

Target-network configurations periodically copy the online network's weights into a separate network used to calculate learning targets.

### Exploration

Actions follow an epsilon-greedy policy. Exploration decreases exponentially:

```text
epsilon = max(epsilon_end, epsilon_start × exp(-total_steps / epsilon_decay))
```

### Vectorized Environments

Training uses Gymnasium's `SyncVectorEnv` with batched network inference. These environments execute synchronously; they are not separate worker processes.

The code selects CUDA when available and otherwise uses CPU. The function-based trainer also uses automatic mixed precision on CUDA.

## Default Training Settings

| Parameter | Value |
|-----------|-------|
| Environment | `CartPole-v1` |
| Timesteps per run | 100,000 |
| Number of environments | 16 |
| Hidden layers | `(128, 128)` |
| Learning rate | 0.0005 |
| Discount factor | 0.99 |
| Replay capacity | 10,000 transitions |
| Replay batch size | 256 |
| Replay updates per vector step | 5 |
| Target synchronization interval | Approximately 1,000 transitions |
| Initial epsilon | 1.0 |
| Minimum epsilon | 0.05 |
| Epsilon decay scale | 10,000 |
| Experiment seeds | 0, 1, 2, 3, 4 |

For configurations without replay, the code performs one update per collected transition. The `update_ratio` setting applies only to replay-based training.

## Experiments

### Configuration Comparison

`run_all_configurations()` evaluates all four configurations across five seeds and generates:

- Smoothed reward curves with variability bands.
- A final-performance comparison chart.

### Hyperparameter Grid Search

`run_grid_search()` evaluates the `tn_er` configuration using:

| Parameter | Values |
|-----------|--------|
| Learning rate | 0.0001, 0.0005, 0.001 |
| Replay update ratio | 1, 5, 10 |
| Hidden layers | `(64, 64)`, `(128, 128)`, `(256, 256)` |
| Seeds | 0, 1, 2, 3, 4 |

This produces 27 configurations and 135 training runs.

### Exploration Ablation

`run_ablation_study_exploration()` compares epsilon decay scales of:

- 5,000
- 10,000
- 15,000

Each setting is evaluated across five seeds using `tn_er`.

### Neural Q-Learning Curve

`run_q_learning_plot()` runs the `naive` configuration across five seeds with 24 environments and plots a smoothed cumulative-average reward against environment steps.

## Repository Contents

| File | Description |
|------|-------------|
| `main.py` | Networks, replay buffer, training routines, experiments, and plotting |
| `requirements.txt` | Python dependencies |
| `Praneeth_RL_assignment_1_report.pdf` | Project report |
| `README.md` | Project documentation |

## Installation

```bash
git clone https://github.com/Praneeth-496/CartPole-DQN-Reinforcement-Learning.git
cd CartPole-DQN-Reinforcement-Learning
python -m pip install -r requirements.txt
```

Dependencies include:

- Gymnasium
- PyTorch
- NumPy
- Matplotlib
- pandas

The requirements specify minimum versions rather than an exact tested environment. Review Gymnasium vector-environment behavior and PyTorch AMP compatibility when choosing versions.

## Usage

### Run All Experiments

```bash
python main.py
```

All four experiment calls are enabled in the current script.

The complete workflow schedules 175 training runs, totaling approximately 17.5 million environment transitions. Reduce the experiment scope for a shorter run.

### Run Only the Configuration Comparison

```bash
python -c "from main import run_all_configurations; run_all_configurations()"
```

### Run One Configuration

The training function can also be called directly:

```python
from main import train_cartpole_dqn

rewards, cumulative_mean_rewards, steps = train_cartpole_dqn(
    config="tn_er",
    total_timesteps=10_000,
    network_size=(128, 128),
    num_envs=4,
    seed=0,
    return_steps=True,
)

print("Completed episodes:", len(rewards))
print("Last episode reward:", rewards[-1] if rewards else None)
```

Review the implementation notes below before relying on experimental results.

## Generated Outputs

The script creates plots under `plots/` and writes progress messages to `training_log.txt`.

| Output | Description |
|--------|-------------|
| `learning_curve_configurations.png` | Four-configuration learning curves |
| `final_performance_configurations.png` | Configuration comparison |
| `grid_search_learning_curves.png` | Hyperparameter learning curves |
| `grid_search_final_performance.png` | Hyperparameter comparison |
| `learning_curve_ablation_exploration.png` | Exploration-decay learning curves |
| `final_performance_ablation_exploration.png` | Exploration-decay comparison |
| `q_learning_curve_5seeds_extended.png` | Neural Q-learning curve |

The grid-search function returns a results dictionary and pandas DataFrame. The script does not automatically export these results to CSV or save trained model checkpoints.

## Implementation Notes

- **Vector-environment resets need correction.** Both trainers manually reset individual environments while using `SyncVectorEnv`. The function-based trainer then overwrites the reset observations with `next_states`; the class-based trainer discards reset observations. Reset handling should follow the selected Gymnasium autoreset mode.

- **Termination and truncation are combined.** Both suppress bootstrapping in the learning target. Time-limit truncation should be handled separately when the task requires continued-value bootstrapping.

- **The episode-limit argument is unused.** `max_steps_per_episode` is accepted but is not applied when creating or stepping the environment.

- **Seeded runs are not fully deterministic.** Python, NumPy, PyTorch, and environment resets are seeded, but the action space used for random exploration is not explicitly seeded.

- **Final scores use truncated episode histories.** Runs are shortened to the minimum episode count across seeds before averaging. The resulting score is not the average of each seed's own final 100 episodes.

- **Evaluation uses training rewards.** There is no separate greedy-policy evaluation phase on fresh seeds.

- **The extra Q-learning plot uses cumulative averages.** Its displayed endpoint can also be extended to the requested step budget without an additional observation.

## Results and Reproducibility

The repository includes a project report but does not bundle raw training logs, generated result files, or model checkpoints.

No numerical performance claims are included here because the experiments have not been independently rerun. Correct the environment-reset handling and establish a consistent evaluation protocol before drawing conclusions about configuration superiority.

## Potential Improvements

- Correct vector-environment reset and truncation handling.
- Add evaluation episodes with exploration disabled.
- Save raw per-seed results and model checkpoints.
- Compare learning curves at matched environment-step counts.
- Pin dependency versions and record experiment settings.
- Add command-line options for selecting experiments.
- Explore Double DQN, Huber loss, and gradient clipping.

## Author

[Praneeth Dathu](https://github.com/Praneeth-496)
