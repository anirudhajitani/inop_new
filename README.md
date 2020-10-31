# inop_new
First step
-----------
Generate lambda and N buffers

python main_train.py --help to list all the arguments to be passed

Some important arguments
lambd_evolve --> When True, user arrival rate changes over time
user_identical --> When False, all users have same different requests rate
user_evolve --> When True, no of users change

To generate lambda and N buffers
Use algo == 4
python main_train.py --algo 4 --folder scenario_1_lambd_2 --env_name salmut --logdir salmut_log --lambd 0.5 --lambd_evolve False --user_identical True --user_evolve False

To train the agents
algo == 0 (PPO) algo == 1 (A2C) algo == 3(SALMUT)
python main_train.py --algo 3 --folder scenario_1_lambd_2 --env_name salmut --logdir salmut_log --lambd 0.5 --lambd_evolve False --user_identical True --user_evolve False

To evaluate the agents
start_iter --> evaluation start step, step --> how many evaluation steps
python main_eval2.py --algo 3 --folder scenario_1_lambd_2 --env_name salmut --logdir salmut_log --lambd 0.5 --lambd_evolve False --user_identical True --user_evolve False --start_iter 0 --step 1000

To run Planning and Baseline MC returns
algo == 0 (Optimal solution) algo == 1 (baseline threshold)
python main_plan.py --algo 0 --folder scenario_1_lambd_2 --env_name salmut --logdir salmut_log --lambd 0.5 --lambd_evolve False --user_identical True --user_evolve False

To combine multiple results
python combine_results.py <scenario_folder> <env_name>

To plot results
python plot_results.py <scenario_folder>
