# Ray!
# import ray
# import ray.rllib.algorithms.ppo as ppo
from ray.rllib.policy.policy import Policy
# from ray.rllib.algorithms import Algorithm
# from ray import serve
# RLLib Model
from ray.rllib.models import ModelCatalog
from ray.rllib.models.preprocessors import get_preprocessor
from model.my_rllib_models import MyCustomRLlibModel, MyMLPModel
# Environment
from env.envs import SingleStaticEnv, GreedyPolicy
from ray.tune.registry import register_env
# Utils
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
import copy


# Register your custom model and environment
ModelCatalog.register_custom_model("custom_model", MyCustomRLlibModel)
ModelCatalog.register_custom_model("custom_nn_model", MyMLPModel)
# TODO: Is there a way to adjust the config of a restored policy?
#       If the model name is different, it will not be restored. (Carefully determine the model name in training XD)
register_env("test_env", lambda cfg: SingleStaticEnv(cfg))

base_path = "../../../ray_results/PPO/"

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
# Get preprocessor for MLP model (may cause error with transformer models;)
preprocessor_cls = get_preprocessor(env.observation_space)
preprocessor = preprocessor_cls(env.observation_space)

# Define experiment configs
num_experiments = 100000
require_flatten = True
do_render = False
test_render = False
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
    obs_test = copy.deepcopy(obs)
    env_test = copy.deepcopy(env)

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
        if require_flatten:
            obs = preprocessor.transform(obs)
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
    if test_render:  # render policy behavior when it is worse than the greedy
        # if reward_diff > 0.03:  # Policy is worse than greedy
        # if np.abs(reward_diff) < 0.00001:  # Almost the same
        if reward_diff < -0.1:  # Policy is better than greedy
            print(f"reward_diff = {reward_diff}")
            done = False
            time_step = 0
            while not done:
                temp_obs = copy.deepcopy(obs_test)
                if require_flatten:
                    temp_obs = preprocessor.transform(temp_obs)
                    temp_obs = torch.from_numpy(temp_obs).to(policy.device).unsqueeze(0)
                else:
                    for key in temp_obs:
                        # Switch each dict key numpy array to torch tensor and move to the device
                        temp_obs[key] = torch.from_numpy(temp_obs[key]).to(policy.device)
                        temp_obs[key] = temp_obs[key].unsqueeze(0)  # add batch dimension
                logits, _ = policy.model({
                    "obs": temp_obs,
                    "is_training": False,
                }, [], None)
                probs = F.softmax(logits, dim=1)
                action_probs = probs[0].cpu().detach().numpy()
                if require_flatten:
                    obs_test = preprocessor.transform(obs_test)
                action_test = policy.compute_single_action(obs_test, explore=False)
                env_test.render(action_probs=action_probs)
                response = input("Press enter to continue, if not no rendering anymore: ")
                obs_test, _, done, _ = env_test.step(action_test)
                time_step += 1
            env_test.render()
            response = input("Press enter to continue, if not no rendering anymore: ")

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

# Calculate reward_diffs
reward_diffs = reward_policy - reward_greedy

# Plot histogram
plt.figure(figsize=(10, 5))
plt.hist(reward_diffs, bins=100, edgecolor='black')
plt.xlabel("Reward Difference (Policy - Greedy)")
plt.ylabel("Number of Experiments")
plt.title("Distribution of Reward Differences")
plt.show()

print("Done!")
