# -*- coding: utf-8 -*-


import random
import json
import gym
import time
from gym import spaces
import numpy as np
import statistics

class OffloadEnv(gym.Env):

    """A offloading environment for OpenAI gym"""

    def __init__(self, eval_, lambd, offload, overload, holding, reward, N, seed, env_name=None, folder=None, start_iter=0, step=0, add_stats=True):

        super(OffloadEnv, self).__init__()
        #self.action_space = 2
        self.action_space = spaces.Discrete(2)
        self.eval_ = eval_
        #self.observation_space = 2
        np.random.seed(seed)
        random.seed(seed)
        self.observation_high = np.array([20, 20])
        self.observation_low = np.array([0, 0])
        self.observation_space = spaces.Box(
            self.observation_low, self.observation_high)
        self.curr_buffer = 0
        self.max_buffer = 20
        self.cpu_util = 0.0
        self.start_loop = start_iter
        self.end_loop = start_iter + step
        self.start_time = time.time()
        self.current_step = 0
        self.add_stats = add_stats
        self.N = N
        self.c = 2
        self.l = lambd
        self.env_name = env_name
        self.folder = folder
        self.lambd = [lambd] * self.N
        self.mu = 6.0
        self.seed_count = 0
        self.offload_cost = offload
        self.overload_cost = overload
        self.holding_cost = holding
        self.reward = reward
        self.outer_seed_count = 0
        self.overload_count = 0
        self.overload_state = []
        self.med_overload = []
        self.overload_run = []
        self.offload_count = 0
        self.offload_state = []
        self.med_offload = []
        self.offload_run = []
    """
    For now we assume lambda also evolves at same time for all clients
    """
    def set_lambd(self, lambd):
        self.lambd = lambd

    def set_N(self, N, lambd):
        self.N = N
        self.lambd = lambd

    def get_lambd(self):
        return self.lambd

    def get_N(self):
        return self.N

    def get_cpu_util2(self):
        x = np.random.normal(4, 0.25, 1)[0]
        v = min(15.0 + (self.curr_buffer * x), 100.0)
        return int(v/5)

    def get_cpu_util(self, ev_type, action):
        if ev_type == 1:
            prob = 0.6
        elif action == 0:
            prob = 0.6
        else:
            prob = 0.6
        prob = 1.0 - prob
        val = np.random.binomial(n=1, p=prob, size=1)[0]
        if ev_type == 1:
            return max(self.cpu_util - val - 1, 0)
        elif action == 0:
            return min(self.cpu_util + val + 1, 20)
        else:
            return max(self.cpu_util - val, 0)

    def get_reward(self):
        if self.cpu_util >= 18:
            if self.eval_ == True:
                self.overload_count += 1
            return -self.overload_cost
        elif self.cpu_util >=6:
            return self.reward
        else:
            return 0.0

    def get_prob(self):
        if self.curr_buffer == 0:
            prob = 1.0
        elif self.curr_buffer <= self.c:
            prob = sum(self.lambd) / (sum(self.lambd) +
                                      self.curr_buffer * self.mu)
        else:
            prob = sum(self.lambd) / (sum(self.lambd) + self.c * self.mu)
        return 1.0 - prob

    def _next_observation(self):
        # return np.array([self.curr_buffer, self.get_cpu_util2()])
        return np.array([self.curr_buffer, self.cpu_util])

    def _take_action(self, action, event_type):
        if event_type == 1:
            self.curr_buffer = max(self.curr_buffer - 1, 0)
        elif action == 0 and event_type == 0:
            self.curr_buffer = min(self.curr_buffer + 1, 20)

    def step(self, action):
        self.current_step += 1
        # Execute one time step within the environment
        prob = self.get_prob()
        reward = 0.0
        #print ("Prob : ", prob, self.curr_buffer)
        event_type = np.random.binomial(n=1, p=prob, size=1)[0]
        #print ("Event", event_type)
        # Condition where max_buff len is reached
        if self.curr_buffer == self.max_buffer and event_type == 0 and action == 0:
            reward -= self.overload_cost
        self._take_action(action, event_type)
        #print("Buffer, CPU Util before = ", self.curr_buffer, self.cpu_util)
        #self.cpu_util = self.get_cpu_util2()
        self.cpu_util = self.get_cpu_util(event_type, action)
        reward += self.get_reward()
        if action == 1:
            #print ("Offload")
            if self.eval_ == True:
                self.offload_count += 1
            reward -= self.offload_cost
            if self.cpu_util < 3:
                reward -= 10
        reward -= self.holding_cost * \
            (self.curr_buffer - self.c) if self.curr_buffer - self.c > 0 else 0
        #print("Buffer, CPU Util, Reward, Action, Event = ",
        #      self.curr_buffer, self.cpu_util, reward, action, event_type)
        done = False
        """
        if self.current_step % 1000 == 0:
            #print ("Episode complete")
            done = True
            self.overload_state.append(self.overload_count)
            #print ("Overload count = ", self.overload_count)
        """
        obs = self._next_observation()
        return obs, reward, done, {}

    def reset(self):
        #print ("RESET CALLED")
        #self.buffer_size = 10
        if self.eval_ == True:
            #print ("SEED COUNT", self.seed_count)
            if self.seed_count > 0 and self.add_stats == True:
                self.overload_state.append(self.overload_count)
                self.offload_state.append(self.offload_count)
            if self.seed_count % 100 == 99 and self.add_stats == True:
                self.overload_run.append(statistics.mean(self.overload_state))
                #self.std_overload.append(statistics.stdev(self.overload_state))
                self.overload_state = []
                self.offload_run.append(statistics.mean(self.offload_state))
                #self.std_offload.append(statistics.stdev(self.offload_state))
                self.offload_state = []
                self.outer_seed_count = (self.outer_seed_count + 1) % 10
                #print ("AVG OVERLOAD", self.avg_overload, self.std_overload, self.env_name)
                #print ("AVG Offload", self.avg_offload, self.std_offload, self.env_name)
                if self.outer_seed_count % 10 == 0: 
                    self.med_overload.append(np.percentile(self.overload_run, [25,50,75]))
                    self.med_offload.append(np.percentile(self.offload_run, [25,50,75]))
                    print ("Overload, Offload", np.percentile(self.overload_run, [25,50,75]), 
                            np.percentile(self.offload_run, [25,50,75]))
                    #print ("SAVING OVERLOAD COUNT")
                    self.overload_run = []
                    self.offload_run = []
                    if self.start_loop != self.end_loop:
                        np.save(f"./{self.folder}/results/overload_med_{self.start_loop}_{self.end_loop}_{self.env_name}.npy", self.med_overload) 
                        #np.save(f"./{self.folder}/results/overload_std_{self.env_name}.npy", self.std_overload) 
                        np.save(f"./{self.folder}/results/offload_med_{self.start_loop}_{self.end_loop}_{self.env_name}.npy", self.med_offload) 
                        #np.save(f"./{self.folder}/results/offload_std_{self.env_name}.npy", self.std_offload) 
            np.random.seed(int(self.seed_count))
            random.seed(int(self.seed_count))
            self.seed_count = (self.seed_count + 1) % 100
        #self.curr_buffer = random.randrange(10, 20)
        self.curr_buffer = 10 
        self.overload_count = 0
        self.offload_count = 0
        self.cpu_util = 10
        #print("RESET ", self.outer_seed_count, self.seed_count, " State (Buff, CPU) ", self.curr_buffer, self.cpu_util)
        return self._next_observation()
