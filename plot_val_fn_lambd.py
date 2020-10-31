from matplotlib import pyplot as plt
import numpy as np

y1 = np.load('value_fn_6.0.npy')
y2 = np.load('value_fn_9.0.npy')
y3 = np.load('value_fn_12.0.npy')
y4 = np.load('value_fn_15.0.npy')
y5 = np.load('value_fn_18.0.npy')

fig = plt.figure(figsize=(12, 8))
plt.plot(y1, color='b', label='lambda = 6.0')
plt.plot(y2, color='g', label='lambda = 9.0')
plt.plot(y3, color='r', label='lambda = 12.0')
plt.plot(y4, color='m', label='lambda = 15.0')
plt.plot(y5, color='c', label='lambda = 18.0')

plt.xlabel('State Encoding')
plt.ylabel('Value')
plt.legend()
fig.savefig('value_fn_vs_lambda.png')
