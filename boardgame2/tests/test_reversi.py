import numpy as np
import pytest
import gymnasium as gym

import boardgame2


def test_reversi():
    env = gym.make('Reversi-v0')
    assert env.observation_space[0].shape == (8, 8)
    assert env.observation_space[1].shape == ()
    assert env.action_space.shape == (2,)
    assert np.all(env.action_space.high == [7, 7])

    observation, info = env.reset()
    count = 0
    actions = []
    while True:
        #env.render("human")
        action = env.action_space.sample()
        #print("action = ", action)
        observation, reward, termination, truncated, info = env.step(action)
        #print(observation, reward, termination, info)
        if termination:
            break
        else:
            actions.append(action)
            #env.render("human")
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
