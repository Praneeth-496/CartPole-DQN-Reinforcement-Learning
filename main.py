import os
import logging
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import time
import pandas as pd

# Create directories to save our plot images
os.makedirs("plots", exist_ok=True)
os.makedirs("plots/grid_curves", exist_ok=True)  # For saving grid search curves

# Set up logging so that messages show up on screen and get saved in a file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("training_log.txt"), logging.StreamHandler()]
)
logger = logging.getLogger()

# Choose GPU if available, else stick with CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
if device.type == "cuda":
    logger.info(f"GPU in use: {torch.cuda.get_device_name(0)}")

def smooth_curve(data, window=80):
    # Simple function to smooth data with moving average
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")

# QNetwork: A simple neural network to predict Q-values
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(128, 128)):
        super(QNetwork, self).__init__()
        layers = []
        input_dim = state_dim
        # Add hidden layers with ReLU activation – simple and effective
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        # Final layer outputs Q-values for each possible action
        layers.append(nn.Linear(input_dim, action_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

# ReplayBuffer: Stores our experience tuples for training later
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        # Save one experience tuple in the buffer
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        # Randomly select a batch of experiences
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

# Update network using one experience tuple (simple DQN update)
def update_from_single_transition(policy_net, target_net, optimizer, gamma, 
                                  state, action, reward, next_state, done, scaler=None):
    policy_net.train()
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
    action_tensor = torch.LongTensor([action]).to(device)
    reward_tensor = torch.FloatTensor([reward]).to(device)
    done_tensor = torch.FloatTensor([float(done)]).to(device)
    
    if scaler is not None:
        with torch.amp.autocast(device_type="cuda"):
            current_q = policy_net(state_tensor)[0, action_tensor]
            with torch.no_grad():
                next_q = (target_net if target_net is not None else policy_net)(next_state_tensor).max(1)[0]
            target_q = reward_tensor + gamma * next_q * (1.0 - done_tensor)
            loss = nn.MSELoss()(current_q, target_q)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        current_q = policy_net(state_tensor)[0, action_tensor]
        with torch.no_grad():
            next_q = (target_net if target_net is not None else policy_net)(next_state_tensor).max(1)[0]
        target_q = reward_tensor + gamma * next_q * (1.0 - done_tensor)
        loss = nn.MSELoss()(current_q, target_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Update network using a batch of experiences from replay buffer
def update_from_batch(policy_net, target_net, optimizer, gamma, batch, batch_size, scaler=None):
    states, actions, rewards, next_states, dones = batch
    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).to(device)
    
    if scaler is not None:
        with torch.amp.autocast(device_type="cuda"):
            current_q = policy_net(states).gather(1, actions)
            with torch.no_grad():
                next_q = (target_net if target_net is not None else policy_net)(next_states).max(1)[0]
            target_q = rewards + gamma * next_q * (1 - dones)
            target_q = target_q.unsqueeze(1)
            loss = nn.MSELoss()(current_q, target_q)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        current_q = policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q = (target_net if target_net is not None else policy_net)(next_states).max(1)[0]
        target_q = rewards + gamma * next_q * (1 - dones)
        target_q = target_q.unsqueeze(1)
        loss = nn.MSELoss()(current_q, target_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Training function for CartPole using DQN – simple and effective
def train_cartpole_dqn(config, total_timesteps=100000, max_steps_per_episode=500, batch_size=256,
                       gamma=0.99, lr=5e-4, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=10000,
                       buffer_capacity=10000, target_update_freq=1000, seed=0, update_ratio=5,
                       network_size=(128,128), num_envs=16, return_steps=False):
    """
    Train DQN on CartPole-v1 for given timesteps.
    Configurations:
      - "naive": No replay buffer, no target network.
      - "only_tn": Only target network.
      - "only_er": Only replay buffer.
      - "tn_er": Both replay buffer and target network.
    If return_steps=True, returns environment steps at episode ends.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None

    # Create multiple environments to speed up training
    env = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(num_envs)])
    states, _ = env.reset(seed=seed)
    
    state_dim = env.single_observation_space.shape[0]
    action_dim = env.single_action_space.n

    # Initialize our policy network
    policy_net = QNetwork(state_dim, action_dim, hidden_sizes=network_size).to(device)
    # Create a target network if required by configuration
    if config in ["only_tn", "tn_er"]:
        target_net = QNetwork(state_dim, action_dim, hidden_sizes=network_size).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
    else:
        target_net = None

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    # Set up replay buffer if configuration needs it
    if config in ["only_er", "tn_er"]:
        replay_buffer = ReplayBuffer(buffer_capacity)
    else:
        replay_buffer = None

    episode_rewards = []
    current_ep_rewards = np.zeros(num_envs)
    learning_curve = []
    step_record = [] if return_steps else None

    total_steps = 0
    epsilon = epsilon_start
    episode_count = 0

    while total_steps < total_timesteps:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(states).to(device)
            q_values = policy_net(state_tensor)
            best_actions = torch.argmax(q_values, dim=1).cpu().numpy()
        actions = []
        for i in range(num_envs):
            # With probability epsilon, choose random action
            if np.random.rand() < epsilon:
                actions.append(env.single_action_space.sample())
            else:
                actions.append(int(best_actions[i]))
        actions = np.array(actions)

        next_states, rewards, terminated, truncated, infos = env.step(actions)
        dones = np.logical_or(terminated, truncated)
        current_ep_rewards += rewards

        for i in range(num_envs):
            if replay_buffer is not None:
                replay_buffer.push(states[i], actions[i], rewards[i], next_states[i], dones[i])
            else:
                update_from_single_transition(policy_net, target_net, optimizer, gamma,
                                              states[i], actions[i], rewards[i],
                                              next_states[i], dones[i], scaler=scaler)
        
        if replay_buffer is not None and len(replay_buffer) >= batch_size:
            for _ in range(update_ratio):
                batch = replay_buffer.sample(batch_size)
                update_from_batch(policy_net, target_net, optimizer, gamma, batch, batch_size, scaler=scaler)
        
        total_steps += num_envs
        
        # Update target network every now and then
        if target_net is not None and (total_steps % target_update_freq < num_envs):
            target_net.load_state_dict(policy_net.state_dict())
        
        # Decrease epsilon gradually
        epsilon = max(epsilon_end, epsilon_start * np.exp(-total_steps / epsilon_decay))
        
        for i in range(num_envs):
            if dones[i]:
                episode_rewards.append(current_ep_rewards[i])
                learning_curve.append(np.mean(episode_rewards))
                if return_steps:
                    step_record.append(total_steps)
                episode_count += 1
                if episode_count % 100 == 0:
                    logger.info(f"Episode {episode_count} | Reward: {current_ep_rewards[i]:.1f} | Epsilon: {epsilon:.3f}")
                new_state, _ = env.envs[i].reset(seed=seed)
                states[i] = new_state
                current_ep_rewards[i] = 0

        states = next_states

    env.close()
    if return_steps:
        return episode_rewards, learning_curve, step_record
    return episode_rewards, learning_curve

# DQN trainer class with replay buffer and target network, simple style
class DQNReplayTargetTrainer:
    def __init__(self, config, total_timesteps, max_steps_per_episode, batch_size, gamma, lr,
                 epsilon_start, epsilon_end, epsilon_decay, buffer_capacity, target_update_freq,
                 seed, update_ratio, network_size, num_envs):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.config = config
        self.total_timesteps = total_timesteps
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.buffer_capacity = buffer_capacity
        self.target_update_freq = target_update_freq
        self.seed = seed
        self.update_ratio = update_ratio
        self.network_size = network_size
        self.num_envs = num_envs

        # Create multiple environments for training
        self.env = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(num_envs)])
        self.states, _ = self.env.reset(seed=seed)
        state_dim = self.env.single_observation_space.shape[0]
        action_dim = self.env.single_action_space.n

        # Initialize policy network
        self.policy_net = QNetwork(state_dim, action_dim, hidden_sizes=network_size).to(device)
        if config in ["only_tn", "tn_er"]:
            self.target_net = QNetwork(state_dim, action_dim, hidden_sizes=network_size).to(device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
        else:
            self.target_net = None

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        if config in ["only_er", "tn_er"]:
            self.replay_buffer = ReplayBuffer(buffer_capacity)
        else:
            self.replay_buffer = None

    def train(self):
        episode_rewards = []
        current_ep_rewards = np.zeros(self.num_envs)
        total_steps = 0
        epsilon = self.epsilon_start
        episode_count = 0

        while total_steps < self.total_timesteps:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(self.states).to(device)
                q_values = self.policy_net(state_tensor)
                best_actions = torch.argmax(q_values, dim=1).cpu().numpy()
            actions = []
            for i in range(self.num_envs):
                # Random action with chance epsilon, simple hai
                if np.random.rand() < epsilon:
                    actions.append(self.env.single_action_space.sample())
                else:
                    actions.append(int(best_actions[i]))
            actions = np.array(actions)
            next_states, rewards, terminated, truncated, _ = self.env.step(actions)
            dones = np.logical_or(terminated, truncated)
            current_ep_rewards += rewards

            if self.replay_buffer is not None:
                for i in range(self.num_envs):
                    self.replay_buffer.push(self.states[i], actions[i], rewards[i], next_states[i], dones[i])
            else:
                for i in range(self.num_envs):
                    update_from_single_transition(self.policy_net, self.target_net, self.optimizer, self.gamma,
                                                  self.states[i], actions[i], rewards[i],
                                                  next_states[i], dones[i], scaler=None)

            if self.replay_buffer is not None and len(self.replay_buffer) >= self.batch_size:
                for _ in range(self.update_ratio):
                    batch = self.replay_buffer.sample(self.batch_size)
                    update_from_batch(self.policy_net, self.target_net, self.optimizer, self.gamma,
                                      batch, self.batch_size, scaler=None)

            total_steps += self.num_envs

            if self.target_net is not None and (total_steps % self.target_update_freq < self.num_envs):
                self.target_net.load_state_dict(self.policy_net.state_dict())

            epsilon = max(self.epsilon_end, self.epsilon_start * np.exp(-total_steps / self.epsilon_decay))

            for i in range(self.num_envs):
                if dones[i]:
                    episode_rewards.append(current_ep_rewards[i])
                    episode_count += 1
                    current_ep_rewards[i] = 0
                    self.env.envs[i].reset(seed=self.seed)

            self.states = next_states

        self.env.close()
        return np.array(episode_rewards)

# Experiment function: Run experiments for various configurations
def run_all_configurations():
    """
    Run experiments for 4 configurations:
      "naive", "only_tn", "only_er", "tn_er"
    And save:
      - Combined smoothed learning curve plot.
      - Bar chart for final performance.
    """
    configs = ["naive", "only_tn", "only_er", "tn_er"]
    total_timesteps = 100000
    seeds = [0, 1, 2, 3, 4]
    results = {}

    for config in configs:
        all_rewards = []
        logger.info(f"Running configuration: {config.upper()}")
        for seed in seeds:
            rewards, curve = train_cartpole_dqn(
                config=config,
                total_timesteps=total_timesteps,
                max_steps_per_episode=500,
                batch_size=256,
                gamma=0.99,
                lr=5e-4,
                epsilon_start=1.0,
                epsilon_end=0.05,
                epsilon_decay=10000,
                buffer_capacity=10000,
                target_update_freq=1000,
                seed=seed,
                update_ratio=5,
                network_size=(128,128),
                num_envs=16
            )
            all_rewards.append(np.array(rewards))
        min_length = min(len(r) for r in all_rewards)
        all_rewards = np.stack([r[:min_length] for r in all_rewards])
        avg_rewards = np.mean(all_rewards, axis=0)
        std_rewards = np.std(all_rewards, axis=0)
        final_perf = np.mean(avg_rewards[-100:])
        results[config] = {"avg": avg_rewards, "std": std_rewards, "final": final_perf}

    # Plot the smoothed learning curves
    plt.figure()
    for config in configs:
        smoothed_avg = smooth_curve(results[config]["avg"], window=80)
        x_axis = range(len(smoothed_avg))
        plt.plot(x_axis, smoothed_avg, label=config.upper())
        smoothed_std = smooth_curve(results[config]["std"], window=80)
        plt.fill_between(x_axis, smoothed_avg - smoothed_std, smoothed_avg + smoothed_std, alpha=0.2)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward (over seeds)")
    plt.title("DQN Configurations on CartPole-v1 (Smoothed Learning Curves)")
    plt.legend()
    baseline_curve_path = os.path.join("plots", "learning_curve_configurations.png")
    plt.savefig(baseline_curve_path)
    plt.show()
    plt.close()
    logger.info(f"Saved baseline learning curve plot: {baseline_curve_path}")

    # Plot bar chart for final performance
    plt.figure()
    final_values = [results[config]["final"] for config in configs]
    plt.bar(configs, final_values)
    plt.xlabel("Configuration")
    plt.ylabel("Final Performance (Avg of Last 100 Episodes)")
    plt.title("Final Performance Comparison")
    baseline_bar_path = os.path.join("plots", "final_performance_configurations.png")
    plt.savefig(baseline_bar_path)
    plt.show()
    plt.close()
    logger.info(f"Saved baseline final performance plot: {baseline_bar_path}")

# Function for grid search experiments over hyperparameters, simple style
def run_grid_search():
    """
    Grid search over:
      - Learning rate: [1e-4, 5e-4, 1e-3]
      - Update ratio: [1, 5, 10]
      - Network size: [(64,64), (128,128), (256,256)]
    Using 5 seeds. Save:
      - Combined learning curves plot.
      - Bar chart for final performance.
    """
    learning_rates = [1e-4, 5e-4, 1e-3]
    update_ratios = [1, 5, 10]
    network_sizes = [(64,64), (128,128), (256,256)]
    seeds = [0, 1, 2, 3, 4]
    
    grid_results = {}
    grid_learning_curves = {}
    
    for lr in learning_rates:
        for ur in update_ratios:
            for net_size in network_sizes:
                key = f"lr={lr}_ur={ur}_net={net_size[0]}x{net_size[1]}"
                all_rewards = []
                logger.info(f"Grid search - Combination: {key}")
                for seed in seeds:
                    rewards, _ = train_cartpole_dqn(
                        config="tn_er",
                        total_timesteps=100000,
                        max_steps_per_episode=500,
                        batch_size=256,
                        gamma=0.99,
                        lr=lr,
                        epsilon_start=1.0,
                        epsilon_end=0.05,
                        epsilon_decay=10000,
                        buffer_capacity=10000,
                        target_update_freq=1000,
                        seed=seed,
                        update_ratio=ur,
                        network_size=net_size,
                        num_envs=16
                    )
                    all_rewards.append(np.array(rewards))
                min_length = min(len(r) for r in all_rewards)
                all_rewards = np.stack([r[:min_length] for r in all_rewards])
                avg_rewards = np.mean(all_rewards, axis=0)
                final_perf = np.mean(avg_rewards[-100:])
                grid_results[key] = {"final_perf": final_perf}
                smoothed_curve = smooth_curve(avg_rewards, window=80)
                grid_learning_curves[key] = smoothed_curve
    
    # Plot grid search learning curves
    plt.figure(figsize=(12, 8))
    for key, curve in grid_learning_curves.items():
        plt.plot(range(len(curve)), curve, label=key)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Grid Search: Smoothed Learning Curves (All Combinations)")
    plt.legend(fontsize=6, loc="lower right")
    grid_curve_path = os.path.join("plots", "grid_search_learning_curves.png")
    plt.savefig(grid_curve_path)
    plt.show()
    plt.close()
    logger.info(f"Saved grid search learning curve plot: {grid_curve_path}")
    
    # Plot grid search bar chart for final performance
    plt.figure(figsize=(14, 6))
    keys = list(grid_results.keys())
    final_perfs = [grid_results[k]["final_perf"] for k in keys]
    plt.bar(keys, final_perfs)
    plt.xticks(rotation=90)
    plt.xlabel("Hyperparameter Combination")
    plt.ylabel("Final Performance (Avg Reward of Last 100 Episodes)")
    plt.title("Grid Search: Final Performance for Each Combination")
    plt.tight_layout()
    grid_bar_path = os.path.join("plots", "grid_search_final_performance.png")
    plt.savefig(grid_bar_path)
    plt.show()
    plt.close()
    logger.info(f"Saved grid search final performance plot: {grid_bar_path}")
    
    df = pd.DataFrame([{"Combination": k, "Final_Performance": grid_results[k]["final_perf"]} for k in keys])
    logger.info(f"Grid Search Results:\n{df}")
    return grid_results, df

# Ablation study for exploration (epsilon decay) – simple experiment
def run_ablation_study_exploration():
    """
    Ablation study on the exploration factor (epsilon decay) using "tn_er" configuration.
    Test epsilon_decay values: [5000, 10000, 15000] with 5 seeds.
    Save:
      - Combined learning curves plot.
      - Bar chart for final performance.
    """
    epsilon_decays = [5000, 10000, 15000]
    total_timesteps = 100000
    seeds = [0, 1, 2, 3, 4]
    results = {}

    for decay in epsilon_decays:
        all_rewards = []
        for seed in seeds:
            rewards, _ = train_cartpole_dqn(
                config="tn_er",
                total_timesteps=total_timesteps,
                max_steps_per_episode=500,
                batch_size=256,
                gamma=0.99,
                lr=5e-4,
                epsilon_start=1.0,
                epsilon_end=0.05,
                epsilon_decay=decay,
                buffer_capacity=10000,
                target_update_freq=1000,
                seed=seed,
                update_ratio=5,
                network_size=(128,128),
                num_envs=16
            )
            all_rewards.append(np.array(rewards))
        min_length = min(len(r) for r in all_rewards)
        all_rewards = np.stack([r[:min_length] for r in all_rewards])
        avg_rewards = np.mean(all_rewards, axis=0)
        final_perf = np.mean(avg_rewards[-100:])
        results[decay] = {"avg_rewards": avg_rewards, "final_perf": final_perf}

    # Plot learning curves for different epsilon decay values
    plt.figure()
    for decay, res in results.items():
        smoothed_avg = smooth_curve(res["avg_rewards"], window=80)
        x_axis = range(len(smoothed_avg))
        plt.plot(x_axis, smoothed_avg, label=f"Decay={decay}")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Exploration Ablation: Smoothed Learning Curves")
    plt.legend()
    exploration_curve_path = os.path.join("plots", "learning_curve_ablation_exploration.png")
    plt.savefig(exploration_curve_path)
    plt.show()
    plt.close()
    logger.info(f"Saved exploration learning curve plot: {exploration_curve_path}")

    # Plot bar chart for final performance in exploration ablation
    plt.figure()
    decay_labels = [f"Decay={decay}" for decay in results.keys()]
    final_perfs = [results[decay]["final_perf"] for decay in results.keys()]
    plt.bar(decay_labels, final_perfs)
    plt.xlabel("Epsilon Decay")
    plt.ylabel("Final Performance (Avg of Last 100 Episodes)")
    plt.title("Exploration Ablation: Final Performance")
    exploration_bar_path = os.path.join("plots", "final_performance_ablation_exploration.png")
    plt.savefig(exploration_bar_path)
    plt.show()
    plt.close()
    logger.info(f"Saved exploration final performance plot: {exploration_bar_path}")
    
    return results

# Q-Learning implementation and learning curve plotting – simple and clear
def run_q_learning_plot():
    """
    Run the 'naive' Q-learning configuration on CartPole-v1 for 5 seeds.
    Aggregates episode returns and corresponding environment steps.
    Extends x-axis to 100,000 steps if needed.
    Note: Y-axis represents total reward per episode, smoothed by moving average.
    """
    config = "naive"
    total_timesteps = 100000
    seeds = [0, 1, 2, 3, 4]
    num_envs = 24 

    all_step_records = []
    all_learning_curves = []

    # Run training for each seed
    for seed in seeds:
        # Training returns: episode_rewards, learning_curve, step_record
        _, learning_curve, step_record = train_cartpole_dqn(
            config=config,
            total_timesteps=total_timesteps,
            max_steps_per_episode=500,
            batch_size=256,
            gamma=0.99,
            lr=5e-4,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=10000,
            buffer_capacity=10000,
            target_update_freq=1000,
            seed=seed,
            update_ratio=5,
            network_size=(128,128),
            num_envs=num_envs,
            return_steps=True
        )
        all_learning_curves.append(np.array(learning_curve))
        all_step_records.append(np.array(step_record))

    # Align data by trimming to the minimum length among seeds
    min_length = min(len(rec) for rec in all_step_records)
    trimmed_steps = np.stack([rec[:min_length] for rec in all_step_records])
    trimmed_rewards = np.stack([lc[:min_length] for lc in all_learning_curves])

    # Compute the average over seeds
    avg_steps = np.mean(trimmed_steps, axis=0)
    avg_rewards = np.mean(trimmed_rewards, axis=0)

    # Smooth the averaged reward curve using moving average
    window = 80
    smoothed_rewards = smooth_curve(avg_rewards, window=window)
    smoothed_steps = avg_steps[window-1:]

    # Extend the curve to 100,000 steps if needed
    if smoothed_steps[-1] < total_timesteps:
        smoothed_steps = np.concatenate((smoothed_steps, [total_timesteps]))
        smoothed_rewards = np.concatenate((smoothed_rewards, [smoothed_rewards[-1]]))

    # Plot the learning curve: Reward vs. Environment Steps
    plt.figure()
    plt.plot(smoothed_steps, smoothed_rewards, label="Q-Learning Curve (5 seeds avg)")
    plt.xlabel("Environment Steps")
    plt.ylabel("Reward")
    plt.title("Q-Learning: Learning Curve on CartPole-v1")
    plt.legend()
    plt.grid(True)
    plt.xlim([0, total_timesteps])
    plt.show()
    
    #save the plot
    q_learning_curve_path = os.path.join("plots", "q_learning_curve_5seeds_extended.png")
    plt.savefig(q_learning_curve_path)
    logger.info(f"Saved Q-Learning learning curve plot: {q_learning_curve_path}")

# Main entry point
if __name__ == "__main__":
    overall_start_time = time.time()

    # Uncomment these lines to run the full experiments:
    run_all_configurations()          # Run baseline experiments (saves 2 images)
    run_grid_search()                 # Run grid search experiments (saves 2 images)
    run_ablation_study_exploration()  # Run exploration ablation experiments (saves 2 images)
    
    # For Q-Learning learning curve plotting, run:
    run_q_learning_plot()

    overall_end_time = time.time()
    total_seconds = overall_end_time - overall_start_time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60

    logger.info(f"Total runtime: {hours} hours, {minutes} minutes, {seconds:.2f} seconds")
