import numpy as np
import gym
from gym.spaces import Box, Discrete, Dict, MultiDiscrete
import matplotlib.pyplot as plt
import copy


class SingleStaticEnv(gym.Env):

    def __init__(self, config):
        super(SingleStaticEnv, self).__init__()

        # Task number information
        if "num_task_max" in config:
            self.num_task_max = config["num_task_max"]  # maximum number of tasks for padding of the observation
        else:
            raise ValueError("num_task_max must be specified in the config")
        self.num_task_min = config["num_task_min"] if "num_task_min" in config else 1  # minimum number of tasks
        self.num_task = None  # current number of tasks

        # Observation space
        embedding_dim = 2  # dimension of the vector where the task information is embedded; here 2 for x and y position
        self.observation_space = Dict({  # task_embeddings: (seq_len, d_k) == (num_task_max, embedding_dim)
                                       "task_embeddings": Box(low=-np.inf, high=np.inf,
                                                              shape=(self.num_task_max, embedding_dim),
                                                              dtype=np.float32
                                                              ),  # Already embedded not (simple) tokens; also padded
                                       # "num_task": Discrete(self.num_task_max),  # (num of current tasks-1)
                                       "pad_tokens": Box(low=0, high=1, shape=(self.num_task_max,),
                                                         dtype=np.int32),  # indicates which tasks are padded
                                       # "pad_tokens": MultiDiscrete([2]*self.num_task_max),  # padded == 1
                                       "completion_tokens": Box(low=0, high=1, shape=(self.num_task_max,),
                                                                dtype=np.int32,  # was supposed to be discrete/int
                                                                ),  # indicates which tasks are available
                                       "prev_action": Box(low=-1, high=self.num_task_max-1, shape=(), dtype=np.int64),
                                       # TODO: MUST:: Can I get prev_action not in an array but in a single value?
                                       #             if you want to amend the form, you may want to make a change in
                                       #             the model accordingly!! (applying squeeze(!!) the val in the model)
                                       #             OR, perhaps, scalar is better as it doesn't require squeeze, which
                                       #             may reduce the computation time in the model
                                       "first_action": Box(low=-1, high=self.num_task_max-1, shape=(1,), dtype=np.int32),
                                       # "time_decision": Discrete(self.num_task_max+1),
                                       # "real_clock": Box(low=0, high=np.inf, shape=(1,)),
                                       })
        self.dummy_observation = self.observation_space.sample()  # for the sake of the shape

        # Action space
        self.action_space = Discrete(self.num_task_max)  # no stop action; from 0 to num_task_max-1

        # Decision counter (indicates conventional time step in the MDP)
        self.decision_count = None
        # # Time step counter (indicates real time in the realworld)
        # self.real_clock = None

        self.prev_action = None
        self.first_action = None

        self.task_embeddings = None  # Relative positions of the tasks from the robot
        self.is_padded = None  # 1 if the task is padded, 0 otherwise
        self.is_completed = None  # 1 if the task is completed, 0 otherwise

        self.robot_position = None
        self.task_positions = None

    def reset(self):
        # Reset the times
        self.decision_count = 0    # Discrete(0, self.num_task_max)
        # self.real_clock = 0.0  # Box(low=0, high=np.inf, shape=(1,))

        # Generate positions of the tasks and the robot
        self.num_task = np.random.randint(low=self.num_task_min, high=self.num_task_max+1)
        self.robot_position = np.random.uniform(low=-0.5, high=0.5, size=(2,))
        self.task_positions = np.random.uniform(low=-0.5, high=0.5, size=(self.num_task, 2))
        # Task embeddings are relative positions of the tasks from the robot
        self.task_embeddings = np.zeros(shape=(self.num_task_max, 2), dtype=np.float32)
        self.task_embeddings[:self.num_task, :] = self.task_positions - self.robot_position

        # Generate pad_tokens and completion_tokens
        # Now the tokens are numpy int32
        self.get_pad_tokens()
        self.is_completed = np.zeros(shape=(self.num_task_max,), dtype=np.int32)  # Nothing is completed at start

        # Actions as observation
        self.prev_action = np.array(-1, dtype=np.int64)
        self.first_action = np.array([-1])

        # Generate observation
        obs = self.get_observation()

        # self.action_space = Discrete(self.num_task)

        return obs

    def get_observation(self):
        obs = self.dummy_observation.copy()  # TODO: Is there a better way? (empty dict?)
        obs["task_embeddings"] = self.task_embeddings
        # obs["num_task"] = self.num_task
        obs["pad_tokens"] = self.is_padded
        obs["completion_tokens"] = self.is_completed
        # obs["time_decision"] = self.decision_count
        obs["prev_action"] = self.prev_action
        obs["first_action"] = self.first_action
        # obs["real_clock"] = self.real_clock
        return obs

    def get_pad_tokens(self):
        self.is_padded = np.zeros(shape=(self.num_task_max,), dtype=np.int32)
        self.is_padded[self.num_task:] = 1

    def step(self, action):
        """
        The robot goes to the task the action indicates. (state update)
        Reward is computed by the distance the robot traveled.
        :param action: Index of the task to be completed
        """
        # Check if the action is valid
        # if action < 0 or action >= self.num_task:
        #     raise ValueError(f"Invalid action: {action} is not in [0, {self.num_task})")
        if isinstance(action, tuple):
            action = action[0]
        if action >= self.num_task:  # TODO: YOU SHOULD REMOVE THIS FOR BEST PERF; JUST FOR DEBUGGING
            print(f"Invalid action: {action} is not in [0, {self.num_task})")
            action = np.random.randint(low=0, high=self.num_task)
            print(f"    Now, randomly selected action: {action}")

        # Update the decision counter
        self.decision_count += 1

        # Compute the travel distance of the robot
        reward = -np.linalg.norm(self.robot_position - self.task_positions[action, :])
        # Normalize the reward by the number of tasks and outputs error if the num_task is 0
        # reward /= 2.0  # tasks were generated in [-1, 1] x [-1, 1]; make it equivalent to the unit square case
        # TODO: Check which reward you want to use.
        #       (1) dist: the (mag of) rewards is larger when num_task is larger
        #       (2) dist / sqrt(num_task): the (mag of) rewards will be uniformly distributed over the num_task
        #       (3) dist / num_task: the (mag of) rewards is larger when num_task is smaller
        reward /= self.num_task  # self.num_task will never be 0; if you amend the code, you may want to assert this
        # reward /= np.sqrt(self.num_task)  # Euclidean shortest path grows approximately proportional to sqrt(num_task)
        # Update the robot position
        self.robot_position = self.task_positions[action, :]
        # Update the task embeddings
        self.task_embeddings[:self.num_task, :] = self.task_positions - self.robot_position
        # Update the completion tokens
        self.is_completed[action] = 1
        # Update the prev_action
        self.prev_action = np.array(action, dtype=np.int64)
        # Update the first_action
        if self.first_action == -1:
            self.first_action = np.array([action])
        # Update the pad tokens
        # self.get_pad_tokens()  # This is NOT update at the moment, as the num of tasks is fixed in this experiment
        # Update num_task
        # your implementation: Not this time! fixed upon reset
        # Update the observation
        obs = self.get_observation()

        # Check if done
        # self.decision_count must be smaller than self.num_task. (self.num_task-1) must be the last step.
        # Please note that self.decision_count starts from 0.
        # TODO: Done=True the environment either if
        #      (1) the robot completed all the tasks, or
        #      (2) decision_count==num_task_max with a little huge penalty
        done = self.decision_count >= self.num_task

        return obs, reward, done, {}

    def render(self, action_probs=None, mode='human'):
        # action_prob: a torch tensor (or a numpy array) of shape (self.num_task_max,); if None, no action_prob is shown
        # But we should only consider the first self.num_task elements of action_prob

        if mode == 'human':
            plt.figure(figsize=(8, 6))

            # Plot the robot position
            plt.scatter(self.robot_position[0], self.robot_position[1], c='r', marker='o', s=200, label='Robot')

            # Plot the tasks
            for i, task_position in enumerate(self.task_positions):
                color = 'g' if self.is_completed[i] == 0 else 'b'  # Uncompleted tasks are green, completed are blue
                plt.scatter(task_position[0], task_position[1], c=color, marker='x', s=100, label=f'Task {i + 1}')

                # If action probabilities are provided, plot them as horizontal bars
                if action_probs is not None:
                    # Normalize the action probability for visualization purposes
                    # normalized_action_prob = action_probs[i].item() / action_probs.max().item()
                    normalized_action_prob = action_probs[i].item() / 5.0
                    plt.barh(task_position[1], normalized_action_prob, left=task_position[0], height=0.01, color='b')

            plt.xlim(-0.6, 0.6)
            plt.ylim(-0.5, 0.5)
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left')  # Place the legend outside of the plot
            # plt.legend()

            # plt.tight_layout()
            # Set equal scaling (i.e., make circles circular) by changing dimensions of the plot box.
            # plt.axis('equal', adjustable='box')
            plt.gca().set_aspect('equal', adjustable='box')
            plt.subplots_adjust(right=0.8)  # adjust the right boundary of the plot to make room for the legend

            plt.show()


class GreedyPolicy:
    def __init__(self, initial_observation=None):
        self.observation = None
        self.num_task = None
        self.action_history = None
        self.reward_history = None
        self.decision_count = None

        if initial_observation is not None:
            self.reset(initial_observation)
        else:
            print("Warning: GreedyPolicy is initialized without initial_observation")
            print("         You must call reset() before using this policy")

    def reset(self, initial_observation):
        self.observation = copy.deepcopy(initial_observation)
        self.num_task = np.sum(self.observation["pad_tokens"] == 0)  # The number of tasks
        self.action_history = np.empty(shape=(self.num_task,), dtype=np.int32)  # Warning: np.empty() returns random arr
        self.reward_history = np.empty(shape=(self.num_task,), dtype=np.float32)
        self.decision_count = 0

    def calculate_policy(self):
        # Get initial information from observation
        task_embeddings = self.observation["task_embeddings"].copy()  # TODO: copy() may not be necessary
        is_completed = self.observation["completion_tokens"].copy()
        is_padded = self.observation["pad_tokens"].copy()

        # Iterate until all tasks are completed
        while not all(is_completed) and not all(is_completed[is_padded == 0]):
            # Calculate distances to tasks, and replace distances to completed or padded tasks with infinity
            distances = np.linalg.norm(task_embeddings, axis=1)
            distances[is_completed == 1] = np.inf
            distances[is_padded == 1] = np.inf

            # Find the closest task
            action = np.argmin(distances)
            reward = -distances[action] / self.num_task  # Normalize the reward with the number of tasks

            # Update the task_embeddings to reflect the robot's new position
            task_embeddings -= task_embeddings[action]

            # Mark this task as completed
            is_completed[action] = 1

            # Store the action and reward
            self.action_history[self.decision_count] = action
            self.reward_history[self.decision_count] = reward
            self.decision_count += 1

    def get_total_travelled_distance(self):
        # The total travelled distance is the sum of the negative rewards
        return np.sum(self.reward_history)

    def get_action_history(self):
        # The action history is the order in which the tasks were completed
        return self.action_history

    def get_reward_history(self):
        # The reward history is the negative distance travelled to each task
        return self.reward_history


