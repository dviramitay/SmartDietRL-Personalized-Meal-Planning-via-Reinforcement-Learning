"""
plot_macronutrient_charts.py

This script generates visualizations for:
- Macronutrient breakdown (calories and grams)
- Reward components contribution
- Reward distribution across day types (regular, low_carb, post_workout)

Used for analyzing learned Q-learning agent behavior and validating reward design.
"""

import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
from meal_env import MealEnv
from q_learning_agent import QLearningAgent


def plot_macronutrient_pie(calories, protein, carbs, fat):
    """Pie chart: Caloric contribution from each macronutrient."""
    kcal_from_protein = protein * 4
    kcal_from_carbs = carbs * 4
    kcal_from_fat = fat * 9
    total_kcal = kcal_from_protein + kcal_from_carbs + kcal_from_fat

    labels = ['Protein', 'Carbs', 'Fat']
    values = [kcal_from_protein, kcal_from_carbs, kcal_from_fat]
    percentages = [v / total_kcal * 100 for v in values]

    plt.figure(figsize=(7, 7))
    plt.pie(values,
            labels=[f"{l} ({p:.1f}%)" for l, p in zip(labels, percentages)],
            autopct='%1.1f%%', startangle=140)
    plt.title("Macronutrient Caloric Distribution (Pie Chart)")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def plot_macronutrient_grams_pie(protein, carbs, fat):
    """Pie chart: Distribution of macronutrients by grams."""
    total_grams = protein + carbs + fat
    labels = ['Protein', 'Carbs', 'Fat']
    values = [protein, carbs, fat]
    percentages = [v / total_grams * 100 for v in values]

    plt.figure(figsize=(7, 7))
    plt.pie(values,
            labels=[f"{l} ({p:.1f}%)" for l, p in zip(labels, percentages)],
            autopct='%1.1f%%', startangle=140)
    plt.title("Macronutrient Distribution by Grams (Pie Chart)")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def plot_macronutrient_bar(calories, protein, carbs, fat):
    """Bar chart: Absolute caloric values from macronutrients."""
    values = [protein * 4, carbs * 4, fat * 9]
    labels = ['Protein', 'Carbs', 'Fat']

    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color=['blue', 'orange', 'green'])
    plt.ylabel("Calories (kcal)")
    plt.title("Calories per Macronutrient (Bar Chart)")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()


def plot_reward_by_day_type(q_table_path="q_table.pkl", num_episodes=5000):
    """Visualize average and distribution of total rewards by day type."""
    try:
        with open(q_table_path, "rb") as f:
            q_table = pickle.load(f)
    except FileNotFoundError:
        print("❌ Q-table not found.")
        return

    env = MealEnv()
    agent = QLearningAgent(
        state_size=9,  # Includes day_type
        action_size=len(env.get_available_actions()),
        epsilon=0.0
    )
    agent.q_table = q_table

    rewards_by_day_type = {"regular": [], "low_carb": [], "post_workout": []}

    for _ in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            actions = env.get_available_actions()
            if not actions:
                break
            action = agent.select_action(state, actions)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward

        rewards_by_day_type[env.day_type].append(total_reward)

    # Average rewards
    avg_rewards = {k: np.mean(v) for k, v in rewards_by_day_type.items()}

    # Bar chart
    plt.figure(figsize=(8, 5))
    plt.bar(avg_rewards.keys(), avg_rewards.values(), color=["gray", "orange", "green"])
    plt.title("Average Total Reward by Day Type")
    plt.ylabel("Average Total Reward")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()

    # Boxplot
    plt.figure(figsize=(8, 5))
    plt.boxplot(
        [rewards_by_day_type[k] for k in ["regular", "low_carb", "post_workout"]],
        labels=["regular", "low_carb", "post_workout"]
    )
    plt.title("Reward Distribution by Day Type (Boxplot)")
    plt.ylabel("Total Reward")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    # Print values
    print("\n📊 Average Reward by Day Type:")
    for k, v in avg_rewards.items():
        print(f"{k}: {v:.2f}")


def plot_reward_component_bar():
    """Bar chart: Average contribution of each reward component."""
    try:
        with open("reward_component_breakdown.pkl", "rb") as f:
            reward_components = pickle.load(f)
    except FileNotFoundError:
        print("❌ reward_component_breakdown.pkl not found. Please run reward_analysis.py first.")
        return

    plt.figure(figsize=(10, 6))
    plt.bar(reward_components.keys(), reward_components.values(), color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Average Reward Contribution")
    plt.title("Average Contribution of Reward Components")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()


def load_top_meals_and_plot():
    """
    Load the summary of top meal combinations and generate:
    - Macronutrient pie & bar charts
    - Reward component breakdown
    - Reward by day type analysis
    """
    try:
        with open("top_meals_summary.pkl", "rb") as f:
            df_meals = pickle.load(f)
    except FileNotFoundError:
        print("❌ top_meals_summary.pkl not found. Please run train_and_plot.py first.")
        return

    summary_row = df_meals[df_meals['Meal'] == "Total (Summary)"]
    if summary_row.empty:
        print("❌ Summary row not found in the data.")
        return

    total = summary_row.iloc[0]
    calories = total['Calories']
    protein = total['Protein']
    carbs = total['Carbs']
    fat = total['Fat']

    #plot_macronutrient_pie(calories, protein, carbs, fat)
    #plot_macronutrient_bar(calories, protein, carbs, fat)
    plot_macronutrient_grams_pie(protein, carbs, fat)
    plot_reward_component_bar()
    plot_reward_by_day_type()


if __name__ == "__main__":
    load_top_meals_and_plot()
