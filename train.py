from baselines import deepq
from baselines.common import set_global_seeds
from baselines import bench
import argparse
from baselines import logger
from baselines.common.atari_wrappers import make_atari
import os

def main(envName='BreakoutNoFrameskip-v4', bufferSize=10000, timesteps=3e6):
    # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    # parser.add_argument('--buffer', type=int, default=10000)
    # parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    # parser.add_argument('--prioritized', type=int, default=1)
    # parser.add_argument('--prioritized-replay-alpha', type=float, default=0.6)
    # parser.add_argument('--dueling', type=int, default=1)
    # parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    # parser.add_argument('--checkpoint-freq', type=int, default=10000)
    # parser.add_argument('--checkpoint-path', type=str, default=os.getcwd()+"/logs")
    # args = parser.parse_args()
    # logger.configure(dir=args.checkpoint_path)
    logger.configure(dir=os.getcwd()+"/logs/"+str(envName)+"_"+str(bufferSize))
    set_global_seeds(0)
    env = make_atari(envName)
    env = bench.Monitor(env, logger.get_dir())
    env = deepq.wrap_atari_dqn(env)
    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[512],
        dueling=bool(1),
    )

    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-4,
        max_timesteps=int(timesteps),
        buffer_size=bufferSize,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=bool(1),
        prioritized_replay_alpha=0.6,
        checkpoint_freq=10000,
    )
    act.save(os.getcwd()+"/logs/"+str(envName)+"_"+str(bufferSize)+"/model.pkl")
    env.close()


if __name__ == '__main__':
    main()