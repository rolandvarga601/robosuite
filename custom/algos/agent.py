import numpy as np


class RandomAgent():
	def __init__(self, action_range):
		self.action_low, self.action_high = action_range

	def get_action(self, obs):
		action = np.random.uniform(self.action_low, self.action_high)
		return (action, {})

	def reset(self):
		pass
