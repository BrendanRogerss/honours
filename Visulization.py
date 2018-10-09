import matplotlib.pyplot as plt
import matplotlib as mpl


def baseline_log(filename):
    x = []
    for line in open(filename):
        x.append(line.split(',')[0])

    x.pop(0)
    x.pop(0)

    y = [float(i) for i in x]
    return y

def baselineplot(file, avg):
    y = baseline_log(file)
    y_avg = []
    # print(y)
    for i in range(0, len(y), avg):
        sumY = sum(y[i:(i + avg)])
        if i + avg > len(y):
            y_avg.append(sumY / (len(y) - i))
        else:
            y_avg.append(sumY / avg)
    # print(y_avg)
    x = [j for j in range(0, len(y_avg))]

    return x, y_avg

def baselineGraph(name, size, avg):
    mpl.style.use("seaborn")



    fileName = "logs/" + name + "NoFrameskip-v4_" + str(1000) + "/monitor.csv"
    x, y_avg = baselineplot(fileName, avg)
    plt.plot(x, y_avg, label="1000")

    fileName = "logs/" + name + "NoFrameskip-v4_" + str(10000) + "/monitor.csv"
    x, y_avg = baselineplot(fileName, avg)
    plt.plot(x, y_avg, label="10000")

    fileName = "logs/" + name + "NoFrameskip-v4_" + str(100000) + "/monitor.csv"
    x, y_avg = baselineplot(fileName, avg)
    plt.plot(x, y_avg, label="100000")

    plt.legend()

    plt.ylabel('Reward')
    plt.xlabel('Episodes * '+str(avg))
    plt.title(name)
    #plt.show()
    plt.savefig("graphs/"+name+".png")
    plt.close()

sizes = [1000, 10000, 100000]
#for size in sizes:
names = ["Assault", "Atlantis", "BeamRider","CrazyClimber", "DemonAttack", "Gopher", "Kangaroo", "KungFuMaster", "Pong", "RoadRunner", "SpaceInvaders", "StarGunner", "VideoPinball"]
for name in names:
    baselineGraph(name, None, 100)
