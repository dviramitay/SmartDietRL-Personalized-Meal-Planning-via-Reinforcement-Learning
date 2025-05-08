import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd
from collections import Counter, defaultdict
from meal_env import MealEnv
from q_learning_agent import QLearningAgent
from meals_data import MEALS

if __name__ == "__main__":
    env = MealEnv()
    agent = QLearningAgent(
        state_size=6,
        action_size=len(env.get_available_actions())
    )

    # טעינת Q-Table אם קיים
    if os.path.exists('q_table.pkl'):
        with open('q_table.pkl', 'rb') as f:
            agent.q_table = pickle.load(f)
        print("\n✅ Loaded existing Q-Table from q_table.pkl.\n")
    else:
        print("\n⚡ No Q-Table found. Starting fresh.\n")

    # אימון
    num_episodes = 40000
    rewards = []
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
            print(f"Episode {episode + 1}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    # שמירת Q-Table בסיום
    with open('q_table.pkl', 'wb') as f:
        pickle.dump(agent.q_table, f)
    print("\n✅ Q-Table saved to q_table.pkl.\n")

    # גרף תגמולים + ממוצע נע
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

    # גרף פיזור תגמולים לפי רבעים
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

    # טבלת המנות הנבחרות ביותר
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
    print(df_meals)

    with open("top_meals_summary.pkl", "wb") as f:
        pickle.dump(df_meals, f)

    # הדפסת 3 מנות מובילות לכל שלב
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
