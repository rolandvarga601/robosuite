import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TKAgg")


class MyLogger:
	def __init__(self):
		self.epoch_dict = dict()

	def store(self, **kwargs):
		for key, value in kwargs.items():
			if not(key in self.epoch_dict.keys()):
				self.epoch_dict[key] = []
			self.epoch_dict[key].append(value)

	def plot(self, signals):
		for i, key in zip(range(len(signals)), signals):
			plt.figure(num=i)
			plt.plot(list(map(abs, self.epoch_dict[key])))
			plt.title(label=key)
			plt.yscale('log')