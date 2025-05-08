import pickle
import numpy as np
import matplotlib.pyplot as plt
from meal_env import MealEnv
from q_learning_agent import QLearningAgent
from meals_data import MEALS

# Load trained Q-table
try:
    with open("q_table.pkl", "rb") as f:
        q_table = pickle.load(f)
    print("✅ Q-Table loaded successfully.")
except FileNotFoundError:
    raise Exception("❌ No Q-Table found. Please train the agent first.")

# Setup
env = MealEnv()
agent = QLearningAgent(
    state_size=6,
    action_size=len(env.get_available_actions()),
    epsilon=0.0
)
agent.q_table = q_table

# Track cumulative contributions
components_sum = {
    "protein_reward": 0.0,
    "satiety_reward": 0.0,
    "protein_density_bonus": 0.0,
    "satiety_density_bonus": 0.0,
    "calorie_range_bonus": 0.0,
    "protein_goal_bonus": 0.0,
    "protein_deficit_penalty": 0.0,
    "calorie_surplus_penalty": 0.0,
    "fat_penalty": 0.0,
    "low_satiety_penalty": 0.0
}

num_episodes = 1000

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

    # Extract totals
    cals = env.total_calories
    prot = env.total_protein
    fat = env.total_fat
    sat = env.satiety

    components_sum["protein_reward"] += prot * 0.5
    components_sum["satiety_reward"] += sat * 0.5
    components_sum["protein_density_bonus"] += (prot / max(cals, 1)) * 10
    components_sum["satiety_density_bonus"] += (sat / max(cals, 1)) * 10
    components_sum["calorie_range_bonus"] += 15 if 1600 <= cals <= 1800 else 0
    components_sum["protein_goal_bonus"] += 10 if prot >= env.protein_target else 0
    components_sum["protein_deficit_penalty"] += -0.5 * max(0, env.protein_target - prot)
    components_sum["calorie_surplus_penalty"] += -0.2 * max(0, cals - env.calorie_target)
    components_sum["fat_penalty"] += -0.5 * fat
    components_sum["low_satiety_penalty"] += -5 if sat < 3 else 0

# ממוצעים
avg_components = {k: v / num_episodes for k, v in components_sum.items()}
print("\n🔍 Average Reward Component Contribution over 1000 episodes:")
for k, v in avg_components.items():
    print(f"{k}: {v:.2f}")

# Save to file for visualization
with open("reward_component_breakdown.pkl", "wb") as f:
    pickle.dump(avg_components, f)

# Visualization
plt.figure(figsize=(10, 6))
plt.bar(avg_components.keys(), avg_components.values(), color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.ylabel("Average Reward Contribution")
plt.title("Average Contribution of Reward Components")
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

################
# Counters for violations
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

    # Track violations
    if env.total_protein < env.protein_target:
        protein_deficit_violations += 1
    if env.total_calories > env.calorie_target:
        calorie_surplus_violations += 1
    if env.satiety < 3:
        low_satiety_violations += 1

# Display results
print("\n🔍 Reward Violation Rates over", num_episodes, "episodes:")
print(f"Protein deficit: {protein_deficit_violations / num_episodes * 100:.1f}%")
print(f"Calorie surplus: {calorie_surplus_violations / num_episodes * 100:.1f}%")
print(f"Low satiety: {low_satiety_violations / num_episodes * 100:.1f}%")
