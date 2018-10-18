import csv


def scan(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)

    return your_list


def metric(L):
    L = list(map(float, L))
    L = list(map(int, L))

    low = min(L)
    avg = sum(L)/len(L)
    high = max(L)
    return [int(low), int(avg), int(high)]

if __name__ == "__main__":

    raw = scan("/home/brendan/PycharmProjects/honours/validationStats.csv")
    metrics = [metric(i[2:]) for i in raw]
    results = []
    for i in range(len(raw)):
        result = raw[i][:2]+metrics[i]
        #print(result)
        results.append(result)
    strResults = [list(map(str, i)) for i in results]

    for i in strResults:
        print(', '.join(i))