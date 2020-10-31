from plan_policy_new_debug import PlanPolicy
import glob
import numpy as np
"""
file_list = glob.glob('policies/*.npy')
for f in file_list:
    print (f)
    pol = np.load(f)
    print (type(pol), pol)
    l = float(f.split('_')[-1][0:-4])
    lambd = [l/24.0] * 24
    print (lambd)
    pol = PlanPolicy(24, lambd, 10, 1, 0.12, 0.2)
    pol.plot_graph(False, pol)
"""


lambd = [0.5] * 24
pol = PlanPolicy(24, lambd, 10, 1, 0.12, 0.2)
pol.compute_policy(plot=True, save=True, decay_rew=False)
lambd = [0.75] * 24
pol = PlanPolicy(24, lambd, 10, 1, 0.12, 0.2)
pol.compute_policy(plot=True, save=True, decay_rew=False)
