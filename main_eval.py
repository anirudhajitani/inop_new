import argparse
import copy
import importlib
import json
import os
import statistics
import numpy as np
import torch
import random
import discrete_BCQ
import structured_learning
import DQN
from NewOffloadEnv import OffloadEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
#from OffloadEnv2 import OffloadEnv
import utils
import pickle
from stable_baselines3.common.cmd_util import make_vec_env
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN, SAC
from stable_baselines3 import A2C, PPO
#from stable_baselines.common.vec_env import DummyVecEnv
#from stable_baselines import DQN
from stable_baselines3.common import results_plotter


def interact_with_environment(env, replay_buffer, is_atari, num_actions, state_dim, device, args, parameters):
    # For saving files
    setting = f"{args.env}_{args.seed}"
    buffer_name = f"{args.buffer_name}_{setting}"
    #setting = args.env + "_" + args.seed
    #buffer_name = args.buffer_name + "_" + setting

    # Initialize and load policy
    policy = DQN.DQN(
        is_atari,
        num_actions,
        state_dim,
        device,
        parameters["discount"],
        parameters["optimizer"],
        parameters["optimizer_parameters"],
        parameters["polyak_target_update"],
        parameters["target_update_freq"],
        parameters["tau"],
        parameters["initial_eps"],
        parameters["end_eps"],
        parameters["eps_decay_period"],
        parameters["eval_eps"],
    )

    if args.generate_buffer:
        policy.load(f"./models/behavioral_{setting}")

    evaluations = []

    state, done = env.reset(), False
    episode_start = True
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    low_noise_ep = np.random.uniform(0, 1) < args.low_noise_p

    # Interact with the environment for max_timesteps
    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        # If generating the buffer, episode is low noise with p=low_noise_p.
        # If policy is low noise, we take random actions with p=eval_eps.
        # If the policy is high noise, we take random actions with p=rand_action_p.
        if args.generate_buffer:
            if not low_noise_ep and np.random.uniform(0, 1) < args.rand_action_p - parameters["eval_eps"]:
                action = np.random.binomial(n=1, p=0.5, size=1)[0]
            else:
                action = policy.select_action(np.array(state), eval=True)

        if args.train_behavioral:
            if t < parameters["start_timesteps"]:
                action = np.random.binomial(n=1, p=0.5, size=1)[0]
            else:
                action = policy.select_action(np.array(state))

        # Perform action and log results
        next_state, reward, done, info = env.step(action)
        episode_reward += reward

        # Only consider "done" if episode terminates due to failure condition
        done_float = float(
            done) if episode_timesteps < env._max_episode_steps else 0

        # For atari, info[0] = clipped reward, info[1] = done_float
        if is_atari:
            reward = info[0]
            done_float = info[1]

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward,
                          done_float, done, episode_start)
        state = copy.copy(next_state)
        episode_start = False

        # Train agent after collecting sufficient data
        if args.train_behavioral and t >= parameters["start_timesteps"] and (t+1) % parameters["train_freq"] == 0:
            policy.train(replay_buffer)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_start = True
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            low_noise_ep = np.random.uniform(0, 1) < args.low_noise_p

        # Evaluate episode
        if args.train_behavioral and (t + 1) % parameters["eval_freq"] == 0:
            evaluations.append(eval_policy(policy, args.env, args.seed))
            np.save(f"./results/behavioral_{setting}", evaluations)
            policy.save(f"./models/behavioral_{setting}")

    # Save final policy
    if args.train_behavioral:
        policy.save(f"./models/behavioral_{setting}")

    # Save final buffer and performance
    else:
        evaluations.append(eval_policy(policy, args.env, args.seed))
        np.save(f"./results/buffer_performance_{setting}", evaluations)
        replay_buffer.save(f"./buffers/{buffer_name}")


# Trains BCQ offline
def train_BCQ(env, replay_buffer, is_atari, num_actions, state_dim, device, args, parameters):
    # For saving files
    setting = f"{args.env}_{args.seed}"
    buffer_name = f"{args.buffer_name}_{setting}"
    # buffer_name = f"{args.buffer_name}"
    algo = args.algo
    baseline = False
    # Initialize and load policy
    if algo == 0:
        policy = discrete_BCQ.discrete_BCQ(
            # policy = structured_learning.structured_learning(
            is_atari,
            num_actions,
            state_dim,
            device,
            args.BCQ_threshold,
            parameters["discount"],
            parameters["optimizer"],
            parameters["optimizer_parameters"],
            parameters["polyak_target_update"],
            parameters["target_update_freq"],
            parameters["tau"],
            parameters["initial_eps"],
            parameters["end_eps"],
            parameters["eps_decay_period"],
            parameters["eval_eps"]
        )
    elif algo == 1:
        baseline = True
    elif algo == 2:
        baseline = True

    # Load replay buffer
    replay_buffer.load(f"./buffers/{buffer_name}")
    # replay_buffer.load(f"./results/{buffer_name}")

    evaluations = []
    episode_num = 0
    done = True
    training_iters = 0
    while training_iters < args.max_timesteps:
        if baseline == False:
            for _ in range(int(parameters["eval_freq"])):
                policy.train(replay_buffer)
        evaluations.append(eval_policy(
            policy, args.env, args.seed, baseline, 100, args.threshold))
        np.save(f"./results/BCQ_{setting}", evaluations)

        training_iters += int(parameters["eval_freq"])
        print(f"Training iterations: {training_iters}")


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, type=0, eval_episodes=100, threshold_pol=7):
    #eval_env, _, _, _ = utils.make_env(env_name, atari_preprocessing)
    #eval_env.seed(seed + 100)
    global cpu_util
    global action_list
    eval_env = OffloadEnv()
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        for t in range(200):
            if type == 0:
                action = policy.select_action(np.array(state), eval=True)
            elif type == 1:
                if state[1] < threshold_pol:
                    action = 0
                else:
                    action = 1
            prev_state = state
            # cpu_util.append(state[1])
            # action_list.append(action)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
            print("Eval policy action reward",
                  prev_state, action, reward, state)

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    #cpu_npy = np.array(cpu_util)
    #act_npy = np.array(action_list)
    #np.save('./buffers/cpu_util.npy', cpu_npy)
    #np.save('./buffers/action.npy', act_npy)
    return avg_reward


if __name__ == "__main__":

    # Atari Specific
    atari_preprocessing = {
        "frame_skip": 4,
        "frame_size": 84,
        "state_history": 4,
        "done_on_life_loss": False,
        "reward_clipping": True,
        "max_episode_timesteps": 27e3
    }

    atari_parameters = {
        # Exploration
        "start_timesteps": 2e4,
        "initial_eps": 1,
        "end_eps": 1e-2,
        "eps_decay_period": 25e4,
        # Evaluation
        "eval_freq": 5e3,
        "eval_eps": 1e-3,
        # Learning
        "discount": 0.99,
        "buffer_size": 1e6,
        "batch_size": 32,
        "optimizer": "Adam",
        "optimizer_parameters": {
            "lr": 0.0000625,
            "eps": 0.00015
        },
        "train_freq": 4,
        "polyak_target_update": False,
        "target_update_freq": 8e3,
        "tau": 1
    }

    regular_parameters = {
        # Exploration
        "start_timesteps": 1e3,
        "initial_eps": 0.1,
        "end_eps": 0.1,
        "eps_decay_period": 1,
        # Evaluation
        "eval_freq": 1e4,
        "eval_eps": 0,
        # Learning
        "discount": 0.99,
        "buffer_size": 1e6,
        "batch_size": 1000,
        "optimizer": "Adam",
        "optimizer_parameters": {
            "lr": 3e-4
        },
        "train_freq": 1,
        "polyak_target_update": False,
        "target_update_freq": 1,
        "tau": 0.005
    }

    # Load parameters
    parser = argparse.ArgumentParser()
    # OpenAI gym environment name
    parser.add_argument("--env", default="PongNoFrameskip-v0")
    # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--seed", default=20, type=int)
    # Prepends name to filename
    parser.add_argument("--buffer_name", default="offload_0310")
    # Max time steps to run environment or train for
    parser.add_argument("--max_timesteps", default=1e6, type=int)
    # Threshold hyper-parameter for BCQ
    parser.add_argument("--BCQ_threshold", default=0.3, type=float)
    # Probability of a low noise episode when generating buffer
    parser.add_argument("--low_noise_p", default=0.2, type=float)
    # Probability of taking a random action when generating buffer, during non-low noise episode
    parser.add_argument("--rand_action_p", default=0.2, type=float)
    # If true, train behavioral policy
    parser.add_argument("--train_behavioral", action="store_true")
    # If true, generate buffer
    parser.add_argument("--generate_buffer", action="store_true")
    # Algo 0-PPO 1-A2C 2-SAC
    parser.add_argument("--algo", default=0, type=int)
    parser.add_argument("--baseline-threshold", default=18, type=int)
    parser.add_argument("--env_name", default="res_try_env")
    parser.add_argument("--logdir", default="res_try_log")
    parser.add_argument("--lambd", default=0.5, type=float)
    parser.add_argument("--lambd_high", default=0.75, type=float)
    parser.add_argument("--lambd_evolve", default=False, type=bool)
    parser.add_argument("--user_evolve", default=False, type=bool)
    parser.add_argument("--user_identical", default=True, type=bool)
    parser.add_argument("--folder", default='res_try_0')
    parser.add_argument("--train_iter", default=1e6, type=int)
    parser.add_argument("--eval_freq", default=1e3, type=int)
    parser.add_argument("--offload_cost", default=1.0, type=float)
    parser.add_argument("--overload_cost", default=10.0, type=float)
    parser.add_argument("--holding_cost", default=0.12, type=float)
    parser.add_argument("--reward", default=0.2, type=float)
    parser.add_argument("--N", default=24, type=int)
    args = parser.parse_args()

    print("---------------------------------------")
    if args.train_behavioral:
        print(
            f"Setting: Training behavioral, Env: {args.env}, Seed: {args.seed}")
    elif args.generate_buffer:
        print(
            f"Setting: Generating buffer, Env: {args.env}, Seed: {args.seed}")
    else:
        print(f"Setting: Training BCQ, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if args.train_behavioral and args.generate_buffer:
        print("Train_behavioral and generate_buffer cannot both be true.")
        exit()
    """
    if not os.path.exists(args.folder):
        os.makedirs(args.folder)
    
    if not os.path.exists(f"./{args.folder}/results"):
        os.makedirs(f"./{args.folder}/results")

    if not os.path.exists("./{args.folder}/models"):
        os.makedirs(f"./{args.folder}/models")

    if not os.path.exists("./{args.folder}/buffers"):
        os.makedirs(f"./{args.folder}/buffers")

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    """
    # Make env and determine properties
    #env, is_atari, state_dim, num_actions = utils.make_env(
    #    args.env, atari_preprocessing)
    #is_atari = False
    #state_dim = 4
    state_dim = 2
    num_actions = 2
    gamma = 0.95
    env = OffloadEnv(True, args.lambd, args.offload_cost, args.overload_cost, args.holding_cost, args.reward, args.N, args.seed, args.env_name, args.folder)
    #eval_env = OffloadEnv(True, args.lambd, args.mdp_evolve, args.user_evolve, args.user_identical, args.env_name)
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.algo == 3:
        model = structured_learning.structured_learning(False, num_actions, state_dim, device, args.BCQ_threshold) 
    #env_name = "offload_dqn_mdp_5"
    env_name = args.env_name
    setting = f"{env_name}_{args.seed}"
    buffer_name = f"{args.buffer_name}_{setting}"
    #env = DummyVecEnv([lambda: env])
    #log_dir = "./off_a2c_res_5/"
    #env = make_vec_env(lambda: env, n_envs=1)
    testing_eval_med = []
    #eval_env = make_vec_env(lambda: eval_env, n_envs=1)
    #callback = SaveOnBestTrainingRewardCallback(check_freq=10, log_dir=log_dir)
    loop_range = int(args.train_iter / args.eval_freq)
    with open (f"./{args.folder}/buffers/lambda.npy", "rb") as fp:
        lambd = pickle.load(fp)
    with open (f"./{args.folder}/buffers/N.npy", "rb") as fp:
        N = pickle.load(fp)
    
    for i in range(loop_range):
        print ("EVAL ", i)
        avg_reward_run = []
        for j in range(0, 10):
            print ("J = ", j)
            #lambd = np.load(f"./{args.folder}/buffers/lambda_{args.algo}_{j}.npy")
            #N = np.load(f"./{args.folder}/buffers/N_{args.algo}_{j}.npy")
            model_name = f"./{args.folder}/models/model_{args.algo}_{j}_{i}"
            #print ("Lambd N i ", lambd[i], N[i])
            env.set_N(int(N[i]), list(lambd[i]))
            if args.algo == 0:
                model = PPO.load(model_name, env)
            elif args.algo == 1:
                model = A2C.load(model_name, env)
            elif args.algo == 2:
                model = SAC.load(model_name, env)
            elif args.algo == 3:
                thres_vec = np.load(f"./{args.folder}/buffers/thresvec_{args.env_name}_{j}.npy")
                model.set_threshold_vec(thres_vec[i])
            avg_dis_reward = 0.0
            for k in range(100):
                env.seed(k)
                obs = env.reset()
                reward_traj = []
                dis_reward = 0.0
                for t in range(int(1e3)):
                    if args.algo == 3:
                        action = model.select_action(np.array(obs), eval_=True)
                    else:
                        action, _states = model.predict(obs)
                    obs, reward, done, info = env.step(action)
                    #print(obs, reward, done, info)
                    reward_traj.append(reward)
                #print ("Reward", avg_reward)
                for r in reward_traj[::-1]:
                    dis_reward = r + gamma * dis_reward
                avg_dis_reward += dis_reward
            avg_dis_reward = avg_dis_reward / 100
            print ("AVG REWARD", avg_reward)
            avg_reward_run.append(avg_dis_reward)
        quartiles = np.percentile(avg_reward_run, [25, 50, 75])
        print ("quartiles ", quartiles)
        testing_eval_med.append(quartiles)
        #testing_eval_std.append(avg_std)
        np.save(f"./{args.folder}/results/median_{setting}", testing_eval_med)
        #np.save(f"./{args.folder}/results/std_{setting}", testing_eval_std)

    """
    # Initialize buffer
    replay_buffer = utils.ReplayBuffer(
        state_dim, is_atari, atari_preprocessing, parameters["batch_size"], parameters["buffer_size"], device)

    if args.train_behavioral or args.generate_buffer:
        interact_with_environment(
            env, replay_buffer, is_atari, num_actions, state_dim, device, args, parameters)
    else:
        train_BCQ(env, replay_buffer, is_atari, num_actions,
                  state_dim, device, args, parameters)
    """
