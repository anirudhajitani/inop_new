from plan_policy3 import PlanPolicy

"""
for overload in range(10, 20, 5):
    for holding in range(10, 20, 1):
        for reward in range(0, 5, 1):
            for g in [0.95, 0.99]:
                pol = PlanPolicy(6, [2.0, 2.0, 2.0, 2.0, 2.0, 2.0], overload, 1, holding/100.0, reward/10.0, gamma=g)
                pol.compute_policy(plot=True)
"""

for overload in range(10, 15, 5):
    for offload in range(1, 4, 1):
        for holding in range(0, 30, 2):
            for reward in range(0, 30, 5):
                for g in [0.95]:
                    pol = PlanPolicy(6, [2.0, 2.0, 2.0, 2.0, 2.0, 2.0], overload, offload, holding/100.0, reward/10.0, gamma=g)
                    pol.compute_policy(plot=True, decay_rew=True)
