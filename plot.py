from matplotlib import pyplot as plt
import numpy as np

y1 = np.load('salmut_mean_res_try_env_20.npy')
y2 = np.load('mean_a2c_res_try_20.npy')
y3 = np.load('BCQ_mean_res_try_env_20.npy')

plt.plot(y1, label='salmut')
plt.plot(y2, label='a2c')
plt.plot(y3, label='PPO')

plt.legend()
plt.show()
