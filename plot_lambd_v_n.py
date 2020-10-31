from matplotlib import pyplot as plt
import numpy as np
import pickle 
import sys
import os

folder = sys.argv[1]
os.chdir(f"./{folder}/buffers")
with open('N.npy', 'rb') as f:
    N = pickle.load(f)

with open('lambda.npy', 'rb') as f:
    L = pickle.load(f)

lambd = []
x = []

for i in range(1000):
    lambd.append(sum(L[i]))
    x.append(i*1000)

print (lambd)
N = list(N)


# create figure and axis objects with subplots()
fig,ax = plt.subplots(figsize=(7,2.8))
# make a plot
#ax.plot(x, N, color="red", marker="o")
ax.plot(x, N, color="red")
# set x-axis label
ax.set_xlabel("Timesteps",fontsize=14)
# set y-axis label
ax.set_ylabel("N",color="red",fontsize=14)

# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(x, lambd, color="blue")
#ax2.plot(x, lambd, color="blue",marker="o")
ax2.set_ylabel(r"$\sum \lambda_i$",color="blue",fontsize=14)
plt.show()
# save the plot as a file
fig.savefig(f'{folder}.png',format='png',dpi=100,
                                            bbox_inches='tight')

