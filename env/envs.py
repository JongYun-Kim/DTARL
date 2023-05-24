import numpy as np
import gym
from gym.spaces import Box, Discrete, Dict
# import matplotlib.pyplot as plt


class SingleStaticEnv(gym.Env):
    def __init__(self, config):
        super(SingleStaticEnv, self).__init__()

        #
        self.num_task_max = config["num_task_max"]  # maximum number of tasks for padding of the observation
        self.num_task = None  # current number of tasks

        # Observation space
        embedding_dim = 2  # dimension of the vector where the task information is embedded; here 2 for x and y position
        self.observation_space = Dict({  # task_embeddings: (seq_len, d_k) == (num_task_max, embedding_dim)
                                       "task_embeddings": Box(low=-np.inf, high=np.inf,
                                                              shape=(self.num_task_max, embedding_dim),
                                                              dtype=np.float32
                                                              ),  # Already embedded not (simple) tokens; also padded
                                       "num_task": Discrete(self.num_task_max),  # num of current tasks -1
                                       "pad_tokens": Box(low=0, high=1, shape=(self.num_task_max,),
                                                         dtype=np.float32),  # indicates which tasks are padded
                                       "completion_tokens": Box(low=0, high=1, shape=(self.num_task_max,),
                                                                dtype=np.float32,  # was supposed to be discrete/int
                                                                ),  # indicates which tasks are available
                                       "time_decision": Discrete(self.num_task_max+1),
                                       # "real_clock": Box(low=0, high=np.inf, shape=(1,)),
                                       })
        self.dummy_observation = self.observation_space.sample()  # for the sake of the shape

        # Action space
        # self.action_space = Dict({"action": Discrete(self.num_task_max)})  # no stop action
        self.action_space = Discrete(self.num_task_max)  # no stop action; from 0 to num_task_max-1

        # Decision counter (indicates conventional time step in the MDP)
        self.decision_count = None
        # # Time step counter (indicates real time in the realworld)
        # self.real_clock = None

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
        self.num_task = np.random.randint(low=1, high=self.num_task_max+1)
        self.robot_position = np.random.uniform(low=-1.0, high=1.0, size=(2,))
        self.task_positions = np.random.uniform(low=-1.0, high=1.0, size=(self.num_task, 2))
        self.task_embeddings = self.task_positions - self.robot_position

        # Generate pad_tokens and completion_tokens
        self.get_pad_tokens()
        self.is_completed = np.zeros(shape=(self.num_task_max,), dtype=np.float32)  # Nothing is completed at start

        # Generate observation
        obs = self.get_observation()

        return obs

    def get_observation(self):
        obs = self.dummy_observation.copy()
        obs["task_embeddings"] = self.task_embeddings
        obs["num_task"] = self.num_task
        obs["pad_tokens"] = self.is_padded
        obs["completion_tokens"] = self.is_completed
        obs["time_decision"] = self.decision_count
        # obs["real_clock"] = self.real_clock
        return obs

    def get_pad_tokens(self):
        self.is_padded = np.zeros(shape=(self.num_task_max,), dtype=np.float32)
        self.is_padded[self.num_task:] = 1.0

    def step(self, action):
        """
        The robot goes to the task the action indicates. (state update)
        Reward is computed by the distance the robot traveled.
        :param action: Index of the task to be completed
        """
        # Update the decision counter
        self.decision_count += 1
        # Compute the travel distance of the robot
        reward = -np.linalg.norm(self.robot_position - self.task_positions[action])

        # Update the robot position
        self.robot_position = self.task_positions[action]
        # Update the task embeddings
        self.task_embeddings = self.task_positions - self.robot_position
        # Update the completion tokens
        self.is_completed[action] = 1.0
        # Update the pad tokens
        # self.get_pad_tokens()  # This is not update for now as the num of tasks is fixed in this experiment
        # Update the observation
        obs = self.get_observation()

        # Check if done
        done = self.decision_count >= self.num_task

        return obs, reward, done, {}

    # def render(self, mode='human'):
    #     if mode == 'human':
    #         plt.figure(figsize=(6, 6))
    #
    #         # Plot the robot position
    #         plt.scatter(self.robot_position[0], self.robot_position[1], c='r', marker='o', s=200, label='Robot')
    #
    #         # Plot the tasks
    #         for i, task_position in enumerate(self.task_positions):
    #             color = 'g' if self.is_completed[i] == 0 else 'b'  # Uncompleted tasks are green, completed are blue
    #             plt.scatter(task_position[0], task_position[1], c=color, marker='x', s=100, label=f'Task {i + 1}')
    #
    #         plt.xlim(-1, 1)
    #         plt.ylim(-1, 1)
    #         plt.grid(True)
    #         plt.legend()
    #         plt.show()
