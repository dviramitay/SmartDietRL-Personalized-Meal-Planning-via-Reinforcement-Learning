"""
meal_env.py

Defines the MealEnv class representing the environment in which the agent selects meals.
Handles state transitions, rewards calculation, and available action management.
"""

import numpy as np
from meals_data import MEALS

class MealEnv:
    """Environment simulating meal selections for a day, with nutritional tracking and rewards."""

    def __init__(self, max_steps=4, calorie_target=1800, protein_target=120):
        self.max_steps = max_steps
        self.calorie_target = calorie_target
        self.protein_target = protein_target
        self.meal_schedule = ["breakfast", "lunch", "dinner", "snack"]
        self.reset()

    def reset(self):
        self.total_calories = 0
        self.total_protein = 0
        self.total_carbs = 0
        self.total_fat = 0
        self.satiety = 0
        self.current_step = 0
        self.done = False
        self.chosen_actions = set()
        self.meals = []
        return self._get_state()

    def _get_state(self):
        return np.array([
            self.total_calories,
            self.total_protein,
            self.total_carbs,
            self.total_fat,
            self.satiety,
            self.current_step
        ], dtype=np.float32)

    def step(self, action):
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
        reward = 0

        calories = self.total_calories
        protein = self.total_protein
        fat = self.total_fat
        satiety = self.satiety
        protein_target = self.protein_target
        calorie_target = self.calorie_target

        # חיוביים
        reward += protein * 0.5  # תגמול על חלבון
        reward += satiety * 0.5  # תגמול על שובע
        reward += (protein / max(calories, 1)) * 10  # יחס חלבון לקלוריות
        reward += (satiety / max(calories, 1)) * 10  # יחס שובע לקלוריות

        if 1600 <= calories <= 1800:
            reward += 15  # תגמול על טווח קלורי תקין

        if protein >= protein_target:
            reward += 10  # תגמול נוסף אם הגיע ליעד חלבון

        # שליליים
        if protein < protein_target:
            reward -= 0.5 * (protein_target - protein)  # עונש על חוסר חלבון

        if calories > calorie_target:
            reward -= 0.2 * (calories - calorie_target)  # עונש על קלוריות עודפות

        reward -= 0.5 * fat  # עונש על שומן

        # עונש על אחוז שומן גבוה מ־30% מהקלוריות
        if calories > 0:
            fat_ratio = (fat * 9) / calories
            if fat_ratio > 0.3:
                reward -= (fat_ratio - 0.3) * 100  # ניתן לכוונן את מקדם העונש הזה

        if satiety < 3:
            reward -= 5  # עונש על תחושת שובע יומית נמוכה מדי

        return reward

    def get_available_actions(self):
        current_meal_time = self.meal_schedule[min(self.current_step, len(self.meal_schedule) - 1)]
        return [
            i for i, meal in enumerate(MEALS)
            if current_meal_time in meal["meal_times"] and i not in self.chosen_actions
        ]
