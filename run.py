import train
import time
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=int, default=0)
    parser.add_argument('--buffer', type=int, default=0)
    args = parser.parse_args()

    buffers = [1000,10000,100000]
    games = ['VideoPinballNoFrameskip-v4', 'GopherNoFrameskip-v4', 'RoadRunnerNoFrameskip-v4', 'DemonAttackNoFrameskip-v4', 'AssaultNoFrameskip-v4', 'KangarooNoFrameskip-v4','BeamRiderNoFrameskip-v4', 'KungFuMasterNoFrameskip-v4','QbertNoFrameskip-v4']
    start_time = time.time()
    g = games[args.env]
    b = buffers[args.buffer]
    train.main(g,b)
    with open('times.txt', 'a') as myfile:
        line = g+"_"+str(b)+":"+str(time.time() - start_time)+"\n"
        myfile.write(line)