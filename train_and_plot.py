"""
train_and_plot.py

This script trains a Q-Learning agent in the MealEnv environment to select meal combinations
that maximize a nutritional reward signal. It supports loading/saving Q-tables,
tracking performance, and visualizing learning progress and policy behavior.
"""

import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from meal_env import MealEnv
from q_learning_agent import QLearningAgent
from meals_data import MEALS

if __name__ == "__main__":
    # Initialize environment and agent
    env = MealEnv()
    agent = QLearningAgent(
        state_size=6,
        action_size=len(env.get_available_actions())
    )

    # Load existing Q-Table if available
    if os.path.exists('q_table.pkl'):
        with open('q_table.pkl', 'rb') as f:
            agent.q_table = pickle.load(f)
        print("\n✅ Loaded existing Q-Table from q_table.pkl.\n")
    else:
        print("\n⚡ No Q-Table found. Starting fresh.\n")

    # Training loop
    num_episodes = 5000
    rewards = []
    q_table_sums = []
    action_counter = Counter()
    action_by_step_counter = defaultdict(Counter)

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            available_actions = env.get_available_actions()
            if not available_actions:
                break
            action = agent.select_action(state, available_actions)
            step_index = int(env.current_step)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state

            total_reward += reward
            action_counter[action] += 1
            action_by_step_counter[step_index][action] += 1

        rewards.append(total_reward)

        if (episode + 1) % 100 == 0:
            q_table_sums.append(np.sum([np.sum(v) for v in agent.q_table.values()]))

        if (episode + 1) % 200 == 0:
            avg_recent = np.mean(rewards[-200:])
            print(f"Episode {episode + 1}/{num_episodes}, Average Reward (last 200): {avg_recent:.2f}")

    # Save Q-Table after training
    with open('q_table.pkl', 'wb') as f:
        pickle.dump(agent.q_table, f)
    print("\n✅ Q-Table saved to q_table.pkl.\n")

    # Plot TD error
    rolling_td = pd.Series(agent.td_errors).rolling(window=100).mean()
    plt.plot(agent.td_errors, label="TD Error")
    plt.plot(rolling_td, label="Rolling Avg (100)", linestyle="--")
    plt.title("TD Error over Time")
    plt.xlabel("Step")
    plt.ylabel("TD Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot total reward per episode
    rolling_avg = pd.Series(rewards).rolling(window=100).mean()
    plt.plot(rewards, label='Reward')
    plt.plot(rolling_avg, label='Rolling Avg (100)', linestyle='--')
    plt.title("Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Boxplot of reward distributions by training quarters
    quarters = [
        rewards[:num_episodes // 4],
        rewards[num_episodes // 4:num_episodes // 2],
        rewards[num_episodes // 2:3 * num_episodes // 4],
        rewards[3 * num_episodes // 4:]
    ]
    labels = ["Q1", "Q2", "Q3", "Q4"]
    plt.boxplot(quarters, labels=labels)
    plt.title("Reward Distribution by Training Phase")
    plt.xlabel("Training Phase")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    top_actions = [action_id for action_id, _ in action_counter.most_common(4)]

    meals_records = []
    for action_id in top_actions:
        meal = MEALS[action_id]
        meals_records.append({
            "Meal": meal['name'],
            "Calories": meal['calories'],
            "Protein": meal['protein'],
            "Carbs": meal['carbs'],
            "Fat": meal['fat'],
            "Satiety": meal['satiety']
        })

    df_meals = pd.DataFrame(meals_records)
    total_row = df_meals.sum(numeric_only=True)
    total_row['Meal'] = "Total (Summary)"
    df_meals = pd.concat([df_meals, pd.DataFrame([total_row])], ignore_index=True)

    print("\n📋 Most Chosen Meals Summary:")
    print(df_meals.to_string(index=False))

    with open("top_meals_summary.pkl", "wb") as f:
        pickle.dump(df_meals, f)

    # Print top 3 selected meals per meal step
    print("\nTop 3 meals per meal step:")
    top3_per_step = {}
    for step in range(4):
        print(f"\nStep {step + 1}:")
        top_meals = []
        for action_id, count in action_by_step_counter[step].most_common(3):
            meal_name = MEALS[action_id]['name']
            print(f"  {meal_name} (action {action_id}): {count} times")
            top_meals.append((meal_name, count))
        top3_per_step[step] = top_meals

    # Plot Q-table convergence
    plt.plot(q_table_sums)
    plt.title("Q-table Convergence Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Sum of Q-values")
    plt.grid(True)
    plt.show()

    # Analyze top-performing meal combinations
    combo_rewards = defaultdict(float)
    combo_counts = defaultdict(int)
    env = MealEnv()  # Fresh env for greedy policy testing
    agent.epsilon = 0.0  # Greedy policy only

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            available_actions = env.get_available_actions()
            if not available_actions:
                break
            action = agent.select_action(state, available_actions)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward

        combo_key = tuple(sorted([meal['name'] for meal in env.meals]))
        combo_rewards[combo_key] += total_reward
        combo_counts[combo_key] += 1

    # Calculate average rewards per combo
    avg_combo_rewards = {
        combo: combo_rewards[combo] / combo_counts[combo]
        for combo in combo_rewards
    }

    # Top 5 combos
    top_5_combos = sorted(avg_combo_rewards.items(), key=lambda x: x[1], reverse=True)[:5]
    top_combo = top_5_combos[0]

    print("\n🥇 Top Meal Combination by Maximum Reward:")
    print(f"Max Reward: {top_combo[1]:.2f}")
    print("Meals in Combo:")
    for meal_name in top_combo[0]:
        print(f"  {meal_name}")

    # Plot top 5 combos
    combo_labels = [",\n".join(combo) for combo, _ in top_5_combos]
    combo_scores = [score for _, score in top_5_combos]
    plt.figure(figsize=(12, 6))
    plt.barh(combo_labels[::-1], combo_scores[::-1], color='lightgreen')
    plt.xlabel("Average Total Reward")
    plt.title("Top 5 Meal Combinations by Average Reward")
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.show()
