import argparse
import copy
import importlib
import json
import os
import statistics
import time
import numpy as np
import torch
import random
import discrete_BCQ
import structured_learning
import DQN
from os import path
from NewOffloadEnv import OffloadEnv
import utils


def rephrase_buffer(replay_buffer):
    buffer_name = f"{args.buffer_name}"
    replay_buffer.load(f"./buffers/{buffer_name}")
    f_name = "./results/" + buffer_name + "_state.npy"
    new_buffer = utils.ReplayBuffer(
        state_dim, is_atari, atari_preprocessing, parameters["batch_size"], parameters["buffer_size"], device)
    if path.exists(f_name):
        print ("PATH EXISTS")
        new_buffer.load(f"./results/{buffer_name}")
        print (new_buffer.crt_size, new_buffer.ptr)
    for i in range(replay_buffer.crt_size - 1):
        new_buffer.add(replay_buffer.state[i], replay_buffer.action[i],
                       replay_buffer.state[i+1], replay_buffer.reward[i], 0, 0, 0)
    r = random.randrange(0, replay_buffer.crt_size - 1)
    new_buffer.add(replay_buffer.state[r], replay_buffer.action[r],
                   replay_buffer.state[r+1], replay_buffer.reward[r], 0, 0, 0)
    print (replay_buffer.crt_size, new_buffer.crt_size)
    new_buffer.save(f"./results/{buffer_name}")
    print ("Buffer saved")
    time.sleep(10)
    return new_buffer


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
                action = env.action_space.sample()
            else:
                action = policy.select_action(np.array(state), eval=True)

        if args.train_behavioral:
            if t < parameters["start_timesteps"]:
                action = env.action_space.sample()
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
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
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
    setting = f"{args.env_name}_{args.seed}"
    buffer_name = f"{args.buffer_name}_{setting}"
    #buffer_name = f"{args.buffer_name}"

    # Initialize and load policy
    # policy = discrete_BCQ.discrete_BCQ(
    policy = structured_learning.structured_learning(
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

    # Load replay buffer
    #replay_buffer.load(f"./buffers/{buffer_name}")

    testing_eval_mean = []
    testing_eval_std = []
    training_eval = []
    episode_num = 0
    done = True
    training_iters = 0
    avg_reward = 0.0
    while training_iters < args.train_iter:
        state = env.reset()
        for _ in range(int(args.eval_freq)):
            action = policy.select_action(state)
            prev_state = state
            state, reward, done, _ = env.step(action)
            avg_reward += reward
            if done:
                training_eval.append(avg_reward)
                avg_reward = 0.0
                np.save(f"./results/BCQ_thres_train_{setting}", training_eval)
            policy.train(prev_state, action, reward, state, args.env_name)
        rew, std = eval_policy(policy, args.env_name, args.seed)
        testing_eval_mean.append(rew)
        testing_eval_std.append(std)
        np.save(f"./results/BCQ_thres_mean_{setting}", testing_eval_mean)
        np.save(f"./results/BCQ_thres_std_{setting}", testing_eval_std)
        training_iters += int(args.eval_freq)
        print(f"Training iterations: {training_iters}")


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=100):
    #eval_env, _, _, _ = utils.make_env(env_name, atari_preprocessing)
    #eval_env.seed(seed + 100)

    eval_env = OffloadEnv(True, args.lambd, args.mdp_evolve, args.user_evolve, args.user_identical, env_name)
    avg_reward = 0.0
    avg_reward_run = []
    for i in range(10):
        #print ("SEED ", i*2)
        eval_env.seed(i)
        for j in range(eval_episodes):
            #print ("ROLLOUT", j)
            state, done = eval_env.reset(), False
            state, done = eval_env.reset(), False
            avg_reward = 0.0
            for t in range(1000):
                action = policy.select_action(np.array(state), eval=True)
                state, reward, done, _ = eval_env.step(action)
                avg_reward += reward
            #print ("AVG REWARD", avg_reward)
            #print ("Eval policy action ", state, action, reward)
            avg_reward_run.append(avg_reward)        
    avg_rew = statistics.mean(avg_reward_run)
    avg_std = statistics.stdev(avg_reward_run)
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_rew:.3f}")
    print(f"Evaluation over {eval_episodes} episodes: {avg_std:.3f}")
    print("---------------------------------------")
    return avg_rew, avg_std


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
        "eval_freq": 1e6,
        "eval_eps": 1e-3,
        # Learning
        "discount": 0.99,
        "buffer_size": 1e6,
        "small_buffer_size": 1e3,
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
        "eval_freq": 1e6,
        "eval_eps": 0,
        # Learning
        "discount": 0.99,
        "buffer_size": 1e6,
        "small_buffer_size": 1e6,
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
    parser.add_argument("--seed", default=0, type=int)
    # Prepends name to filename
    parser.add_argument("--buffer_name", default="offload_small2")
    # Max time steps to run environment or train for
    parser.add_argument("--max_timesteps", default=1e8, type=int)
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
    parser.add_argument("--algo", default=0, type=int)
    parser.add_argument("--baseline-threshold", default=7, type=int)
    parser.add_argument("--env_name", default="salmut_try_1")
    parser.add_argument("--log_dir", default="offload_res_1")
    parser.add_argument("--lambd", default=1.0, type=float)
    parser.add_argument("--mdp_evolve", default=False, type=bool)
    parser.add_argument("--user_evolve", default=False, type=bool)
    parser.add_argument("--user_identical", default=True, type=bool)
    parser.add_argument("--train_iter", default=1e8, type=int)
    parser.add_argument("--eval_freq", default=1e6, type=int)

    args = parser.parse_args()
    

    print("---------------------------------------")
    if args.train_behavioral:
        print(f"Setting: Training behavioral, Env: {args.env}, Seed: {args.seed}")
    elif args.generate_buffer:
        print(f"Setting: Generating buffer, Env: {args.env}, Seed: {args.seed}")
    else:
        print(f"Setting: Training BCQ, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if args.train_behavioral and args.generate_buffer:
        print("Train_behavioral and generate_buffer cannot both be true.")
        exit()

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if not os.path.exists("./models"):
        os.makedirs("./models")

    if not os.path.exists("./buffers"):
        os.makedirs("./buffers")

    # Make env and determine properties
    env, is_atari, state_dim, num_actions = utils.make_env(
        args.env, atari_preprocessing)
    is_atari = False
    #state_dim = 3
    state_dim = 2
    num_actions = 2
    parameters = atari_parameters if is_atari else regular_parameters
    env = OffloadEnv(False, args.lambd, args.mdp_evolve, args.user_evolve, args.user_identical, args.env_name)
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize buffer
    replay_buffer = utils.ReplayBuffer(
        state_dim, is_atari, atari_preprocessing, parameters["batch_size"], parameters["small_buffer_size"], device)
    #replay_buffer = rephrase_buffer(replay_buffer)
    if args.train_behavioral or args.generate_buffer:
        interact_with_environment(
            env, replay_buffer, is_atari, num_actions, state_dim, device, args, parameters)
    else:
        train_BCQ(env, replay_buffer, is_atari, num_actions,
                  state_dim, device, args, parameters)
