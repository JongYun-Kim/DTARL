# Ray!
# import ray
# import ray.rllib.algorithms.ppo as ppo
from ray.rllib.policy.policy import Policy
# from ray.rllib.algorithms import Algorithm
# from ray import serve
# RLLib Model
from ray.rllib.models import ModelCatalog
from model.my_rllib_models import MyCustomRLlibModel
# Environment
from env.envs import SingleStaticEnv, GreedyPolicy
from ray.tune.registry import register_env
# Utils
import numpy as np
import copy


# Register your custom model and environment
ModelCatalog.register_custom_model("custom_model", MyCustomRLlibModel)
# TODO: Is there a way to adjust the config of a restored policy?
#       If the model name is different, it will not be restored. (Carefully determine the model name in training XD)
register_env("test_env", lambda cfg: SingleStaticEnv(cfg))

base_path = "../../../ray_results/PPO/"
# algo_ckpt_file = "algorithm_state.pkl"
# policy_ckpt_file = "policy_state.pkl"

# Checkpoint for Algo class
# checkpoint = "PPO_TARL_single_static_env_6e376_00000_0_2023-06-04_05-58-22/checkpoint_004000/"
# checkpoint = base_path + checkpoint # + algo_ckpt_file
# algo = Algorithm.from_checkpoint(checkpoint)
# action = algo.compute_single_action(observation=obs, explore=True, policy_id="default_policy")

# Checkpoint for Policy class
checkpoint2 = "PPO_TARL_single_static_env_6e376_00000_0_2023-06-04_05-58-22/checkpoint_004000/policies/default_policy/"
checkpoint2 = base_path + checkpoint2
policy = Policy.from_checkpoint(checkpoint2)

# Get a greedy policy
greedy_policy = GreedyPolicy()

# Create an environment instance
num_task_max = 10
num_task_min = 1
env_config = {
    "num_task_max": num_task_max,
    "num_task_min": num_task_min,
}
env = SingleStaticEnv(env_config)

# Define experiment configs
num_experiments = 100000
do_render = False
reward_greedy = np.zeros(num_experiments)
# reward_algo = np.zeros(num_experiments)
reward_policy = np.zeros(num_experiments)

# Carry out experiments
worst_case_idx = 0
worst_obs = None
worst_actions = - np.ones(num_task_max, dtype=np.int32)
worst_reward_diff = - np.inf  # reward_diff = reward_greedy - reward_policy
worst_env = None
best_case_idx = 0
best_obs = None
best_actions = - np.ones(num_task_max, dtype=np.int32)
best_reward_diff = np.inf  # reward_diff = reward_greedy - reward_policy
best_env = None
for i in range(num_experiments):
    action_history_temp = - np.ones(10, dtype=np.int32)
    obs = env.reset()
    obs_preserve = copy.deepcopy(obs)
    env_preserve = copy.deepcopy(env)

    # Get results from the greedy policy
    greedy_policy.reset(initial_observation=obs)
    greedy_policy.calculate_policy()
    reward_greedy[i] = greedy_policy.get_total_travelled_distance()

    # Get results from the policy
    done = False
    time_step = 0
    while not done:
        if do_render:
            env.render()
            response = input("Press enter to continue, if not no rendering anymore: ")
            do_render = True if response == "" else False
        action = policy.compute_single_action(obs, explore=False)
        action_history_temp[time_step] = action[0]
        obs, reward, done, _ = env.step(action)
        time_step += 1
        reward_policy[i] += reward
    if do_render:
        env.render()
        response = input("Press enter to continue, if not no rendering anymore: ")
        do_render = True if response == "" else False

    # Compare results
    reward_diff = reward_greedy[i] - reward_policy[i]
    if reward_diff < best_reward_diff:
        best_reward_diff = reward_diff
        best_case_idx = i
        best_obs = obs_preserve
        best_env = copy.deepcopy(env_preserve)
        best_actions = action_history_temp
    if reward_diff > worst_reward_diff:
        worst_reward_diff = reward_diff
        worst_case_idx = i
        worst_obs = obs_preserve
        worst_env = copy.deepcopy(env_preserve)
        worst_actions = action_history_temp

    print("Experiment {} done".format(i))

# Render the (relatively!) best case
best_env_greedy = copy.deepcopy(best_env)
greedy_policy.reset(initial_observation=best_obs.copy())
greedy_policy.calculate_policy()
best_actions_greedy = greedy_policy.get_action_history()
best_env.render()
response = input("Press enter to continue, if not no rendering anymore: ")
do_render = True if response == "" else False
for action in best_actions:
    _, _, done, _ = best_env.step(action)
    if do_render:
        best_env.render()
        response = input("Press enter to continue, if not no rendering anymore: ")
        do_render = True if response == "" else False
    if done:
        break
best_env_greedy.render()
response = input("Press enter to continue, if not no rendering anymore: ")
do_render = True if response == "" else False
for action in best_actions_greedy:
    _, _, done, _ = best_env_greedy.step(action)
    if do_render:
        best_env_greedy.render()
        response = input("Press enter to continue, if not no rendering anymore: ")
        do_render = True if response == "" else False
    if done:
        break
print("Best case: {}".format(best_case_idx))
print("Best case reward diff: {}".format(best_reward_diff))
print("Best case greedy reward: {}".format(reward_greedy[best_case_idx]))
print("Best case policy reward: {}".format(reward_policy[best_case_idx]))
print("Best case actions of the trained policy: {}".format(best_actions))
print("Best case actions of greedy policy: {}".format(best_actions_greedy))
print("")

# Render the (relatively!) worst case
worst_env_greedy = copy.deepcopy(worst_env)
greedy_policy.reset(initial_observation=worst_obs.copy())
greedy_policy.calculate_policy()
worst_actions_greedy = greedy_policy.get_action_history()
worst_env.render()
response = input("Press enter to continue, if not no rendering anymore: ")
do_render = True if response == "" else False
for action in worst_actions:
    _, _, done, _ = worst_env.step(action)
    if do_render:
        worst_env.render()
        response = input("Press enter to continue, if not no rendering anymore: ")
        do_render = True if response == "" else False
    if done:
        break
worst_env_greedy.render()
response = input("Press enter to continue, if not no rendering anymore: ")
do_render = True if response == "" else False
for action in worst_actions_greedy:
    _, _, done, _ = worst_env_greedy.step(action)
    if do_render:
        worst_env_greedy.render()
        response = input("Press enter to continue, if not no rendering anymore: ")
        do_render = True if response == "" else False
    if done:
        break
print("Worst case: {}".format(worst_case_idx))
print("Worst case reward diff: {}".format(worst_reward_diff))
print("Worst case greedy reward: {}".format(reward_greedy[worst_case_idx]))
print("Worst case policy reward: {}".format(reward_policy[worst_case_idx]))
print("Worst case actions of the trained policy: {}".format(worst_actions))
print("Worst case actions of greedy policy: {}".format(worst_actions_greedy))
print("")

# Save the results
# np.savez_compressed("results.npz", reward_greedy=reward_greedy, reward_policy=reward_policy,
#                     best_case_idx=best_case_idx, best_reward_diff=best_reward_diff,
#                     best_actions=best_actions, best_actions_greedy=best_actions_greedy,
#                     worst_case_idx=worst_case_idx, worst_reward_diff=worst_reward_diff,
#                     worst_actions=worst_actions, worst_actions_greedy=worst_actions_greedy)

print("Special thanks to Copilot!")
print("Done!")
