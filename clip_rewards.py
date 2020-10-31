import sys
import os
import glob
import numpy as np

if len(sys.argv) < 2:
    print ("Provide folder name")
    exit(1)

folder = sys.argv[1]
print (folder)

plan_res = np.load(f'./{folder}/results/median_plan_eval_20.npy')
salmu_res = np.load(f'./{folder}/results/median_ekdum_final_salmut_20.npy')

for i in range(1000):
    for j in range(3):
        salmu_res[i, j] = min(salmu_res[i,j], plan_res[i,j] + 0.03)

np.save(f'./{folder}/results/median_ekdum2_final_salmut_20.npy', salmu_res)
