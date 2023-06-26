import numpy as np
import pytest
import gymnasium as gym

import boardgame2
from boardgame2.env import BoardGameEnv


def test_reversi():
    env = gym.make('Reversi-v0')
    assert env.observation_space.shape == (65,)
    assert env.action_space.n == 64
    #assert np.all(env.action_space.high == (7, 7))

    observation, info = env.reset()
    count = 0
    actions = []
    while True:
        #env.render("human")
        # action = 0
        # while not env.is_valid(observation, action):
        #     action = env.action_space.sample()
        action = env.action_space.sample()
        #print("action = ", action)
        obs = np.copy(observation)
        reward = 0
        while reward <= 0:
            observation, reward, termination, truncated, info = env.step(action)
            action = env.action_space.sample()
        #print(observation, reward, termination, info)
        # if np.any(obs != observation[0]):
        #     print(observation[0].reshape((8,8)))

        if termination:
            break
        else:
            actions.append(action)
            # print()
            # env.render()
        count += 1

    if count > 1:
        print("=============")
        for action in actions:
            print(action)
        print("=============")
    env.close()

if __name__ == "__main__":
    for i in range(10000):
        test_reversi()
