from matplotlib import pyplot as plt
import numpy as np
import sys
import os

if len(sys.argv) < 2:
    print ("Provide folder name")
    exit(1)
folder = sys.argv[1]
os.chdir(f"./{folder}/results")

# Loading median results
y1 = np.load('median_ppo_eval_20.npy')
y2 = np.load('median_a2c_eval_20.npy')
y3 = np.load('median_salmut_20.npy')
y4 = np.load('median_plan_eval_20.npy')
y5 = np.load('median_thres_eval_20.npy')

#Loading overload metrics
ov1 = np.load('overload_med_ppo_eval.npy')
ov2 = np.load('overload_med_a2c_eval.npy')
ov3 = np.load('overload_med_salmut.npy')
ov4 = np.load('overload_med_plan_eval.npy')
ov5 = np.load('overload_med_thres_eval.npy')

#Loading offload metrics

off1 = np.load('offload_med_ppo_eval.npy')
off2 = np.load('offload_med_a2c_eval.npy')
off3 = np.load('offload_med_salmut.npy')
off4 = np.load('offload_med_plan_eval.npy')
off5 = np.load('offload_med_thres_eval.npy')

x = [i*1000 for i in range(1000)]

fig, ax = plt.subplots()
ax.plot(x, y1[:,1], 'r-', label='PPO')
ax.fill_between(x, y1[:,0], y1[:,2], color='r', alpha=0.2)
ax.plot(x, y2[:,1], 'b-', label='A2C')
ax.fill_between(x, y2[:,0], y2[:,2], color='b', alpha=0.2)
ax.plot(x, y3[:,1], 'g-', label='SALMUT')
ax.fill_between(x, y3[:,0], y3[:,2], color='g', alpha=0.2)
ax.plot(x, y4[:,1], 'c-', label='Policy Iteration')
ax.fill_between(x, y4[:,0], y4[:,2], color='c', alpha=0.2)
ax.plot(x, y5[:,1], 'm-', label='Threshold=18')
ax.fill_between(x, y5[:,0], y5[:,2], color='m', alpha=0.2)
plt.xlabel('Timesteps')
plt.ylabel('Reward per 1000 timesteps')
fig.savefig(f"{folder}_reward.png")


fig, ax = plt.subplots()
ax.plot(x, ov1[:,1], 'r-', label='PPO')
ax.fill_between(x, ov1[:,0], ov1[:,2], color='r', alpha=0.2)
ax.plot(x, ov2[:,1], 'b-', label='A2C')
ax.fill_between(x, ov2[:,0], ov2[:,2], color='b', alpha=0.2)
ax.plot(x, ov3[:,1], 'g-', label='SALMUT')
ax.fill_between(x, ov3[:,0], ov3[:,2], color='g', alpha=0.2)
ax.plot(x, ov4[:,1], 'c--', label='Policy Iteration')
ax.fill_between(x, ov4[:,0], ov4[:,2], color='c', alpha=0.2)
ax.plot(x, ov5[:,1], 'm--', label='Threshold=18')
ax.fill_between(x, ov5[:,0], ov5[:,2], color='m', alpha=0.2)
plt.xlabel('Timesteps')
plt.ylabel('Overloaded State per 1000 timesteps')
fig.savefig(f"{folder}_overload.png")


fig, ax = plt.subplots()
ax.plot(x, off1[:,1], 'r-', label='PPO')
ax.fill_between(x, off1[:,0], off1[:,2], color='r', alpha=0.2)
ax.plot(x, off2[:,1], 'b-', label='A2C')
ax.fill_between(x, off2[:,0], off2[:,2], color='b', alpha=0.2)
ax.plot(x, off3[:,1], 'g-', label='SALMUT')
ax.fill_between(x, off3[:,0], off3[:,2], color='g', alpha=0.2)
ax.plot(x, off4[:,1], 'c--', label='Policy Iteration')
ax.fill_between(x, off4[:,0], off4[:,2], color='c', alpha=0.2)
ax.plot(x, off5[:,1], 'm--', label='Threshold=18')
ax.fill_between(x, off5[:,0], off5[:,2], color='m', alpha=0.2)
plt.xlabel('Timesteps')
plt.ylabel('Requests Offloaded per 1000 timesteps')
fig.savefig(f"{folder}_offload.png")
