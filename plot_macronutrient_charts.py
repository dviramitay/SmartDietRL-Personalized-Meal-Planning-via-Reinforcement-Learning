import matplotlib.pyplot as plt
import pickle
import pandas as pd


def plot_macronutrient_pie(calories, protein, carbs, fat):
    """Plot a pie chart of macronutrient percentage distribution."""
    kcal_from_protein = protein * 4
    kcal_from_carbs = carbs * 4
    kcal_from_fat = fat * 9
    total_kcal = kcal_from_protein + kcal_from_carbs + kcal_from_fat

    labels = ['Protein', 'Carbs', 'Fat']
    values = [kcal_from_protein, kcal_from_carbs, kcal_from_fat]
    percentages = [v / total_kcal * 100 for v in values]

    plt.figure(figsize=(7, 7))
    plt.pie(values, labels=[f"{l} ({p:.1f}%)" for l, p in zip(labels, percentages)],
            autopct='%1.1f%%', startangle=140)
    plt.title("Macronutrient Caloric Distribution (Pie Chart)")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def plot_macronutrient_bar(calories, protein, carbs, fat):
    """Plot a bar chart comparing absolute caloric values from each macronutrient."""
    kcal_from_protein = protein * 4
    kcal_from_carbs = carbs * 4
    kcal_from_fat = fat * 9

    labels = ['Protein', 'Carbs', 'Fat']
    values = [kcal_from_protein, kcal_from_carbs, kcal_from_fat]

    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color=['blue', 'orange', 'green'])
    plt.ylabel("Calories (kcal)")
    plt.title("Calories per Macronutrient (Bar Chart)")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()


def plot_reward_component_bar():
    """Plot the average contribution of each reward component from reward_component_breakdown.pkl."""
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
    """Automatically load top meals from train_and_plot.pkl file and plot charts."""
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

    plot_macronutrient_pie(calories, protein, carbs, fat)
    plot_macronutrient_bar(calories, protein, carbs, fat)
    plot_reward_component_bar()


if __name__ == "__main__":
    load_top_meals_and_plot()
