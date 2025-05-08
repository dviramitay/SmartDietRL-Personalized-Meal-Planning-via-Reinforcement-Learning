import pickle
import matplotlib.pyplot as plt
import numpy as np
from meal_env import MealEnv
from q_learning_agent import QLearningAgent

# Load trained Q-table
try:
    with open("q_table.pkl", "rb") as f:
        q_table = pickle.load(f)
    print("\n✅ Q-Table loaded successfully.")
except FileNotFoundError:
    raise Exception("\n❌ No Q-Table found! Please run train_and_plot.py first.")

# Setup environment and agent
env = MealEnv()
agent = QLearningAgent(
    state_size=6,
    action_size=len(env.get_available_actions()),
    epsilon=0.0  # No exploration during evaluation
)
agent.q_table = q_table

# Run evaluation
num_episodes = 1000
sampled_proteins = []
sampled_calories = []

print("\n--- Running evaluation episodes (epsilon=0) ---")

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

    sampled_proteins.append(env.total_protein)
    sampled_calories.append(env.total_calories)

# Display results
avg_protein = np.mean(sampled_proteins)
avg_calories = np.mean(sampled_calories)

print("\n--- Evaluation Results ---")
print(f"✅ Average Protein: {avg_protein:.2f} g")
print(f"✅ Average Calories: {avg_calories:.2f} kcal")
"""
# Plot protein distribution
plt.figure(figsize=(10, 5))
plt.hist(sampled_proteins, bins=30, color='skyblue', edgecolor='black')
plt.title("Distribution of Daily Protein Intake")
plt.xlabel("Protein (g)")
plt.ylabel("Number of Episodes")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot calorie distribution
plt.figure(figsize=(10, 5))
plt.hist(sampled_calories, bins=30, color='lightgreen', edgecolor='black')
plt.title("Distribution of Daily Calorie Intake")
plt.xlabel("Calories")
plt.ylabel("Number of Episodes")
plt.grid(True)
plt.tight_layout()
plt.show()
"""