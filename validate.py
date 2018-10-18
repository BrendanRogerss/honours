import gym
import numpy as np
from baselines import deepq
from baselines.common.atari_wrappers import make_atari
import argparse




def main(name, size):
    model_location = ("logs/"+name+"NoFrameskip-v4_"+str(size)+"/model.pkl")
    env_name = name+"NoFrameskip-v4"
    #env = gym.make(name+"NoFrameskip-v4")

    env = make_atari(env_name)
    env = deepq.wrap_atari_dqn(env)
    act = deepq.load(model_location)
    episodeRewards = []

    for i in range(100):
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            # env.render()
            action = act(obs[None])[0]
            obs, rew, done, _ = env.step(action)

            episode_rew += rew
        print(episode_rew)
        episodeRewards.append(episode_rew)

    output = name+","+str(size)
    for r in episodeRewards:
        output+=","+str(r)
    output+="\n"
    with open("validationStats.csv", 'a') as myfile:
        myfile.write(output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str, default="")
    parser.add_argument('buffer', type=int, default=0)
    args = parser.parse_args()



    main(args.env, args.buffer)