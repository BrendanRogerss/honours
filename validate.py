import gym
import numpy as np
from baselines import deepq


def main():
    env = gym.make("SpaceInvadersNoFrameskip-v4")
    act = deepq.load("logs/SpaceInvadersNoFrameskip-v4_100000/model.pkl")

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            action = act(obs[None])[0]
            obs, rew, done, _ = env.step(action)
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()