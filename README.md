# Meal Planning with Reinforcement Learning

This project simulates a real-life decision-making process: selecting a balanced daily meal plan using **Reinforcement Learning (Q-learning)**. The agent learns to choose 4 meals per day (breakfast, lunch, dinner, and snack) in order to meet defined nutritional goals.

---

## Project Goals

The agent is trained to select meals that satisfy the following dietary objectives:

- **Protein intake** ≥ 120 grams  
- **Daily calorie range** between 1600–1800 kcal  
- **Satiety score** ≥ 3  
- **Fat limit**: less than 30% of total daily calories  

This project shows how nutritional goals can be encoded into a reward function and optimized using reinforcement learning.

---

## Project Files Overview

| File | Description |
|------|-------------|
| `meals_data.py` | Defines a list of meals with nutritional values and time-of-day tags (breakfast, lunch, etc.) |
| `meal_env.py` | Defines the environment, including nutritional tracking, action space, and reward function |
| `q_learning_agent.py` | Implements the Q-learning agent, action selection, and learning logic |
| `train_and_plot.py` | Trains the agent and saves reward graphs, top meal selections, and the Q-table |
| `evaluate_agent.py` | Evaluates the trained agent across 1000 episodes |
| `reward_analysis.py` | Analyzes which reward components influence behavior and logs violations |
| `plot_macronutrient_charts.py` | Generates pie and bar charts of macronutrient distribution and reward contributions |
| `q_table.pkl` | The trained Q-table used for evaluation and visualization |
| `top_meals_summary.pkl` | Summary of the most frequently chosen meals by the agent |
| `reward_component_breakdown.pkl` | Stores average reward component values across 1000 episodes |

---

## ⚙️ How to Run

### 1. Train the Agent

python train_and_plot.py

### 2. Evaluate the Agent

python evaluate_agent.py

### 3. Analyze Reward Components

python reward_analysis.py

### 4. Visualize Macronutrient & Reward Distributions

python plot_macronutrient_charts.py

### Example Results
Average protein per day: ~115g

Average calories per day: ~1730 kcal

Most selected meals:
Omelette with cottage cheese
Grilled chicken with sweet potato
Lean beef burger
Protein yogurt

Average episode reward: ~3.85

### Future Improvements
Better fat penalization to reduce % of calories from fat
Expand to >4 meals per day
Multi-objective optimization (e.g., taste, price, sustainability)

