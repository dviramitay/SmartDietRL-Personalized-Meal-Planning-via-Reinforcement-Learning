import random
import numpy as np
from meals_data import MEALS

class MealEnv:
    def __init__(self, max_steps=4, calorie_target=1800, protein_target=120):
        self.max_steps = max_steps
        self.calorie_target = calorie_target
        self.protein_target = protein_target
        self.reset()

    def reset(self):
        self.total_calories = 0
        self.total_protein = 0
        self.total_carbs = 0
        self.total_fat = 0
        self.satiety = 0
        self.current_step = 0
        self.done = False
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

        meal = MEALS[action]

        # Update state
        self.total_calories += meal["calories"]
        self.total_protein += meal["protein"]
        self.total_carbs += meal["carbs"]
        self.total_fat += meal["fat"]
        self.satiety += meal["satiety"]
        self.current_step += 1

        # Calculate reward
        reward = self._calculate_reward()

        # Check if episode is done
        if self.current_step >= self.max_steps or self.total_calories > self.calorie_target * 1.3:
            self.done = True

        return self._get_state(), reward, self.done, {}

    def _calculate_reward(self):
        # Reward function with both penalties and positive encouragement
        protein_diff = abs(self.protein_target - self.total_protein)
        calorie_penalty = max(0, self.total_calories - self.calorie_target)

        # Bonus for being close to protein target
        protein_bonus = 0
        if protein_diff < 10:
            protein_bonus = (10 - protein_diff) * 0.3

        reward = (
            - protein_diff * 0.3
            - calorie_penalty * 0.1
            + protein_bonus
            + self.satiety * 0.3
        )
        return reward

    def get_available_actions(self):
        return list(range(len(MEALS)))
