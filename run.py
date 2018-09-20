import train
import time


if __name__ == '__main__':
    start_time = time.time()
    train.main('AtlantisNoFrameskip-v4',100000)
    print("--- %s seconds ---" % (time.time() - start_time))
