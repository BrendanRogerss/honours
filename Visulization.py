import matplotlib.pyplot as plt


def baseline_log(filename):
    x = []
    for line in open(filename):
        x.append(line.split(',')[0])

    x.pop(0)
    x.pop(0)

    y = [float(i) for i in x]
    return y


def baselineGraph():
    y = baseline_log("logs/AtlantisNoFrameskip-v4_1000/monitor.csv")
    y_avg = []
    print(y)
    for i in range(0, len(y), 10):
        y_avg.append(sum(y[i:(i + 10)]) / 10)
    x = [i for i in range(0, len(y_avg))]

    plt.plot(x, y_avg)
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    plt.title('Breakout, 264 units')
    plt.show()


baselineGraph()
