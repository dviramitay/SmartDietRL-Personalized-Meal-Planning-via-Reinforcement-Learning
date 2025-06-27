"""
reward_analysis.py

This script analyzes the average contribution of each reward component and tracks violation rates
for key nutritional constraints (protein, calories, satiety) over 5000 greedy policy episodes.
Requires a pre-trained Q-table.
"""

import pickle
from meal_env import MealEnv
from q_learning_agent import QLearningAgent

# --- Load Trained Q-Table ---
try:
    with open("q_table.pkl", "rb") as f:
        q_table = pickle.load(f)
    print("✅ Q-Table loaded successfully.")
except FileNotFoundError:
    raise Exception("❌ No Q-Table found. Please train the agent first.")

# --- Initialize Environment and Agent ---
env = MealEnv()
agent = QLearningAgent(
    state_size=6,
    action_size=len(env.get_available_actions()),
    epsilon=0.0  # Use greedy policy for evaluation
)
agent.q_table = q_table

# --- Track cumulative reward component contributions ---
reward_components_keys = [
    "protein_reward", "satiety_reward", "protein_density_bonus", "satiety_density_bonus",
    "calorie_range_bonus", "protein_goal_bonus", "satiety_bonus",
    "macro_balance_penalty", "fat_ratio_penalty", "fat_protein_ratio_penalty", "diversity_bonus"
]
components_sum = {k: 0.0 for k in reward_components_keys}

num_episodes = 5000

for _ in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        available_actions = env.get_available_actions()
        if not available_actions:
            break
        action = agent.select_action(state, available_actions)
        next_state, reward, done, _ = env.step(action)
        state = next_state

    # Extract and accumulate reward components
    components = env._calculate_reward_components()
    for key in reward_components_keys:
        components_sum[key] += components.get(key, 0.0)

# --- Compute averages and display ---
avg_components = {k: v / num_episodes for k, v in components_sum.items()}
print("\n🔍 Average Reward Component Contribution over 5000 episodes:")
for k, v in avg_components.items():
    print(f"{k}: {v:.2f}")

# --- Save results for external plotting ---
with open("reward_component_breakdown.pkl", "wb") as f:
    pickle.dump(avg_components, f)

# --- Track policy violations ---
protein_deficit_violations = 0
calorie_surplus_violations = 0
low_satiety_violations = 0

for _ in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        available_actions = env.get_available_actions()
        if not available_actions:
            break
        action = agent.select_action(state, available_actions)
        next_state, reward, done, _ = env.step(action)
        state = next_state

    # Count violations
    if env.total_protein < env.protein_target:
        protein_deficit_violations += 1
    if env.total_calories > env.calorie_target:
        calorie_surplus_violations += 1
    if env.satiety < 3:
        low_satiety_violations += 1

# --- Display violation rates ---
print(f"\n🔍 Reward Violation Rates over {num_episodes} episodes:")
print(f"Protein deficit: {protein_deficit_violations / num_episodes * 100:.1f}%")
print(f"Calorie surplus: {calorie_surplus_violations / num_episodes * 100:.1f}%")
print(f"Low satiety: {low_satiety_violations / num_episodes * 100:.1f}%")
