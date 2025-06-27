"""
meal_env.py

This module defines the MealEnv environment for reinforcement learning.
The agent selects meals throughout the day to maximize nutritional rewards.
The environment simulates daily constraints such as calorie/protein targets,
meal timing, and varying day types (e.g., regular, low-carb, post-workout).

The reward function encourages balanced, protein-rich, and satisfying meals,
while penalizing overconsumption or poor macronutrient ratios.
"""

import numpy as np
from meals_data import MEALS

class MealEnv:
    def __init__(self, max_steps=4, calorie_target=1800, protein_target=120):
        """
        Initializes the environment with nutritional targets and daily schedule.
        """
        self.max_steps = max_steps
        self.calorie_target = calorie_target
        self.protein_target = protein_target
        self.meal_schedule = ["breakfast", "lunch", "dinner", "snack"]
        self.reset()

    def reset(self):
        """
        Resets the environment for a new episode. Initializes totals and selects a day type.
        """
        self.total_calories = 0
        self.total_protein = 0
        self.total_carbs = 0
        self.total_fat = 0
        self.satiety = 0
        self.current_step = 0
        self.done = False
        self.chosen_actions = set()
        self.meals = []

        self.daily_menu = np.random.choice(len(MEALS), size=30, replace=False)
        self.day_type = np.random.choice(["regular", "low_carb", "post_workout"])

        return self._get_state()

    def _get_state(self):
        """
        Returns the current state vector of nutritional totals and step count.
        """
        return np.array([
            self.total_calories,
            self.total_protein,
            self.total_carbs,
            self.total_fat,
            self.satiety,
            self.current_step
        ], dtype=np.float32)

    def step(self, action):
        """
        Executes a meal choice (action), updates nutritional totals, and returns reward.
        """
        if self.done:
            raise Exception("Episode is done. Please reset the environment.")

        self.chosen_actions.add(action)
        meal = MEALS[action]
        self.meals.append(meal)

        self.total_calories += meal["calories"]
        self.total_protein += meal["protein"]
        self.total_carbs += meal["carbs"]
        self.total_fat += meal["fat"]
        self.satiety += meal["satiety"]
        self.current_step += 1

        reward = self._calculate_reward()

        if self.current_step >= self.max_steps or self.total_calories > self.calorie_target * 1.3:
            self.done = True

        return self._get_state(), reward, self.done, {}

    def _calculate_reward(self):
        """
        Sums the different reward components into a total scalar reward.
        """
        components = self._calculate_reward_components()
        return sum(components.values())

    def _calculate_reward_components(self):
        """
        Calculates and returns individual components of the reward function.
        """
        components = {}

        calories = self.total_calories
        protein = self.total_protein
        carbs = self.total_carbs
        fat = self.total_fat
        satiety = self.satiety
        protein_target = self.protein_target
        calorie_target = self.calorie_target

        components["protein_reward"] = protein * 0.3
        components["satiety_reward"] = satiety * 5
        components["protein_density_bonus"] = (protein / max(calories, 1)) * 50
        components["satiety_density_bonus"] = (satiety / max(calories, 1)) * 20
        components["calorie_range_bonus"] = 30 if 1500 <= calories <= calorie_target else -30
        components["protein_goal_bonus"] = 20 if protein >= protein_target else -20
        components["satiety_bonus"] = 15 if satiety >= 3 else -5

        if self.day_type == "low_carb":
            if carbs > 150:
                components["low_carb_penalty"] = -30
            else:
                components["low_carb_bonus"] = 10

        elif self.day_type == "post_workout":
            components["protein_reward"] += protein * 0.2
            if protein >= self.protein_target:
                components["post_workout_bonus"] = 25

        fat_pct = (fat * 9) / max(calories, 1)
        carb_pct = (carbs * 4) / max(calories, 1)
        protein_pct = (protein * 4) / max(calories, 1)
        balance_penalty = abs(fat_pct - 0.3) + abs(carb_pct - 0.4) + abs(protein_pct - 0.3)
        components["macro_balance_penalty"] = -balance_penalty * 20

        components["fat_ratio_penalty"] = -20 if fat_pct > 0.3 else 20
        components["fat_protein_ratio_penalty"] = -10 if (fat / max(protein, 1)) > 1 else 10

        unique_meals = len(set([m["name"] for m in self.meals]))
        components["diversity_bonus"] = unique_meals * 2

        return components

    def get_available_actions(self):
        """
        Returns the list of meals available at the current meal time,
        filtered by day menu and previously chosen actions.
        """
        current_meal_time = self.meal_schedule[min(self.current_step, len(self.meal_schedule) - 1)]

        return [
            i for i, meal in enumerate(MEALS)
            if (
                i in self.daily_menu and
                i not in self.chosen_actions and
                current_meal_time in meal["meal_times"]
            )
        ]
