import os
from collections import OrderedDict
import pickle
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TKAgg")
import numpy as np
from scipy.ndimage import uniform_filter1d

exps = OrderedDict()

exp_name = 'lb-underestimated'
exp = dict()
exp['save_dir'] = '/home/rvarga/Data/Delft/thesis/implementation/robosuite/runs/lift-mh-seed15-lb-underest-commonloss-200epochs'

with open(os.path.join(exp['save_dir'], 'epoch_dict.pkl'), 'rb') as f:
	exp['epoch_dict'] = pickle.load(f)

exps[exp_name] = exp

exp_name = 'lb'
exp = dict()
exp['save_dir'] = '/home/rvarga/Data/Delft/thesis/implementation/robosuite/runs/lift-20221120-095052'

with open(os.path.join(exp['save_dir'], 'epoch_dict.pkl'), 'rb') as f:
	exp['epoch_dict'] = pickle.load(f)

exps[exp_name] = exp


# exp_name = 'standard'
# exp = dict()
# exp['save_dir'] = '/home/rvarga/Data/Delft/thesis/implementation/robosuite/runs/lift-seed96-standard-200epochs'

# with open(os.path.join(exp['save_dir'], 'epoch_dict.pkl'), 'rb') as f:
# 	exp['epoch_dict'] = pickle.load(f)

# exps[exp_name] = exp

i=int(0)
# Plot LossQ
plt.figure(num=i)
for exp_name in exps.keys():
	plt.plot(exps[exp_name]['epoch_dict']['LossQ'])
plt.title(label='LossQ')
plt.yscale('log')
i += 1

# Plot LossPi
plt.figure(num=i)
for exp_name in exps.keys():
	plt.plot(list(map(abs, exp['epoch_dict']['LossPi'])))
plt.title(label='LossPi')
# plt.yscale('log')
i += 1

# # Plot Q values
# plt.figure(num=i)
# plt.plot(list(map(abs, exp['epoch_dict']['Q1ValsMax'])))
# plt.plot(list(map(abs, exp['epoch_dict']['Q1ValsMin'])))
# # plt.plot(list(map(abs, exp['epoch_dict']['Q1ValsMean'])))
# # plt.plot(list(map(abs, exp['epoch_dict']['Q2ValsMax'])))
# # plt.plot(list(map(abs, exp['epoch_dict']['Q2ValsMin'])))
# # plt.title(label='LossPi')
# # plt.yscale('log')
# i += 1

# # Plot Q values
# plt.figure(num=i)
# plt.plot(min(list(map(abs, exp['epoch_dict']['Q1ValsMax'])),list(map(abs, exp['epoch_dict']['Q2ValsMax']))))
# plt.plot(min(list(map(abs, exp['epoch_dict']['Q1ValsMin'])),list(map(abs, exp['epoch_dict']['Q2ValsMin']))))
# plt.plot(min(list(map(abs, exp['epoch_dict']['Q1ValsMean'])),list(map(abs, exp['epoch_dict']['Q2ValsMean']))))
# # plt.title(label='LossPi')
# # plt.yscale('log')
# i += 1


# # Plot Q values
# plt.figure(num=i)
# for exp_name in exps.keys():
# 	plt.plot(min(list(map(abs, exps[exp_name]['epoch_dict']['Q1ValsMean'])),list(map(abs, exps[exp_name]['epoch_dict']['Q2ValsMean']))))
# # plt.title(label='LossPi')
# # plt.yscale('log')
# i += 1

# Plot Q values
plt.figure(num=i)
for exp_name in exps.keys():
	signal = min(list(map(abs, exps[exp_name]['epoch_dict']['Q1ValsMax'])),list(map(abs, exps[exp_name]['epoch_dict']['Q2ValsMax'])))
	filtered_signal = uniform_filter1d(signal, size=10)
	plt.plot(filtered_signal)
# plt.title(label='LossPi')
# plt.yscale('log')
i += 1

# Plot episode returns
plt.figure(num=i)
for exp_name in exps.keys():
	signal = np.minimum(exps[exp_name]['epoch_dict']['EpRet'], list(np.array(exps[exp_name]['epoch_dict']['EpRet'], dtype=np.uint8)*0+20))
	filtered_signal = uniform_filter1d(signal, size=5)
	plt.plot(filtered_signal)
# plt.title(label='LossPi')
# plt.yscale('log')
i += 1

# Plot episode returns
plt.figure(num=i)
for exp_name in exps.keys():
	signal = np.minimum(exps[exp_name]['epoch_dict']['TestEpRet'], list(np.array(exps[exp_name]['epoch_dict']['TestEpRet'], dtype=np.uint8)*0+20))
	filtered_signal = uniform_filter1d(signal, size=3)
	plt.plot(filtered_signal)
# plt.title(label='LossPi')
# plt.yscale('log')
i += 1

plt.show()


print()