import numpy as np
import gym
from gym.spaces import Box, Discrete, Dict

# Currently migrating to padded obs env; single robot single rl agent (single robot used in order to check if the dymaics-integration method works)
# Think about the multi-robot cases: (1) keep using single rl agent with joint actions or (2) moving to MARL for better scalability at the cost of optimality
class SingleStaticEnv(gym.Env):
    def __init__(self, config):
        self.num_task_max = config["num_task_max"]
        self.num_task = None  # current number of tasks

        # Observation space
        embdding_dim = 2  # dimension of the vector where the task information is embedded
        self.observation_space = Dict({"num_task": Discrete(self.num_task_max),  # num of current tasks
                                       # task_embeddings: (seq_len, d_k) == (num_task_max, embdding_dim)
                                       "task_embeddings": Box(low=-1, high=1, shape=(self.num_task_max, embdding_dim)),
                                       # TODO: Think about the dim of pad mask and if you generate it beforehand
                                       "pad_mask": Box(low=0, high=1, shape=(self.num_task_max, self.num_task_max)),
                                       })

        # Action space
        self.action_space = Dict({"action": Discrete(self.num_task_max),  # no stop action
                                  "action_mask": Box(low=0,
                                                     high=1,
                                                     shape=(self.num_task_max,),
                                                     dtype=np.float32,  # was supposed to be discrete/int
                                                     ),

                                  })

        self.step_count = None

    def reset(self):
        self.step_count = 0
        return obs

    def step(self,
             action
             ):
        self.step_count += 1

        # Check if done
        done = self.step_count >= self.num_task

        return obs, reward, done, info

