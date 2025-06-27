# SmartDietRL-Personalized-Meal-Planning-via-Reinforcement-Learning
A reinforcement learning agent that builds personalized daily meal plans by optimizing nutritional goals such as protein intake, satiety, and calorie balance. The agent learns to select optimal meal combinations from a predefined dataset, using Q-Learning in a deterministic environment.

This project implements a reinforcement learning agent that learns to construct **daily meal plans** based on **nutritional optimization goals**.

The environment simulates real-world meal selection, rewarding the agent for hitting dietary targets such as:

* High protein intake
* Satisfying satiety levels
* Balanced macronutrient distribution
* Staying within calorie limits
* Meal diversity across the day

The agent uses the **Q-Learning** algorithm with an ε-greedy exploration strategy in a **deterministic environment**.

## Project Structure

* `meal_env.py` – The environment simulating a 4-meal day (breakfast, lunch, dinner, snack)
* `meals_data.py` – Dataset of \~20 meals with nutritional values
* `q_learning_agent.py` – The Q-Learning agent logic
* `train_and_plot.py` – Script to train the agent and visualize learning curves
* `sensitivity_analysis.py` – Compares different hyperparameter settings
* `plot_macronutrient_charts.py` – Contains saved graphs and results from training

## Getting Started

```bash
# Clone the repo
git clone https://github.com/yourusername/SmartDietRL-Personalized-Meal-Planning-via-Reinforcement-Learning.git
cd SmartDietRL-Personalized-Meal-Planning-via-Reinforcement-Learning

# Install dependencies (Python 3.7+)
pip install -r requirements.txt

# Run training
python train_and_plot.py
```

## Sample Output

The agent converges toward meal plans that are:

* Protein-rich
* Satiating
* Balanced across macronutrients
* Calorie-appropriate

## Future Work

* More meals and user-specific preferences
* Integration with real recommendation engines
