from CaseMap_CUDA import CaseMap
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import time


def draw_pic(_data: np.array, cost: np.array, N: int):
    plt.plot(np.linspace(0, N, N), _data[1:, 0], c='red')
    plt.xlabel("time")
    plt.ylabel("people counting")
    plt.title("Normal healthy people")
    plt.savefig("Normal healthy people.png",dpi=800)
    plt.show()
    plt.plot(np.linspace(0, N, N), _data[1:, 1], c='blue')
    plt.xlabel("time")
    plt.ylabel("people counting")
    plt.title("Undetected infected people")
    plt.savefig("Undetected infected people.png", dpi=800)
    plt.show()
    plt.plot(np.linspace(0, N, N), _data[1:, 2], c='yellow')
    plt.xlabel("time")
    plt.ylabel("people counting")
    plt.title("Diagnosed")
    plt.savefig("Diagnosed.png", dpi=800)
    plt.show()
    plt.plot(np.linspace(0, N, N), _data[1:, 3], c='green')
    plt.xlabel("time")
    plt.ylabel("people counting")
    plt.title("People with immunity")
    plt.savefig("People with immunity.png", dpi=800)

    plt.show()
    plt.plot(np.linspace(0, N, N), _data[1:, 4], c='gray')
    plt.xlabel("time")
    plt.ylabel("people counting")
    plt.title("Death")
    plt.savefig("Death.png",dpi=800)
    plt.show()
    # money
    plt.plot(np.linspace(0, N, N), cost[1:, 0], c='gray')
    plt.xlabel("time")
    plt.ylabel("RMB")
    plt.title("Cost_Total")
    plt.show()


if __name__ == "__main__":
    origin = np.array([12424, 10112, 8128, 6548, 5244, 4172, 3333, 2676, 2152,
                       1738, 1387, 1131, 935, 781, 632, 546, 459, 402,
                       344, 300, 259, 222, 190, 176, 172, 158, 157,
                       144, 135, 114, 103, 94, 83, 73, 68, 62,
                       56, 57, 45, 43, 44, 41, 43, 44, 41,
                       44, 45, 40, 33, 30, 29, 26, 25, 24,
                       19, 14, 11, 8, 8, 7, 6, 6, 4,
                       3, 1, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0], dtype=np.int_)
    np.random.seed(1)
    a = np.random.random_sample([1024, 1024])
    time_start = time.time()
    p = [0.001, 0.00, 0.001]  # 初始时,1,2,3状态的比例
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i, j] < p[0]:
                a[i, j] = 1
            elif a[i, j] < p[0] + p[1]:
                a[i, j] = 2
            elif a[i, j] < p[0] + p[1] + p[2]:
                a[i, j] = 3
    a = a.astype(np.int_)
    time_end = time.time()
    print(time_end - time_start)
    print(a)
    rule_array = np.array([
        [0, 0.008, 0, 0.002, 0],
        [0, 0.695, 0.07, 0, 0.005],
        [0, 0, 0.79, 0.1, 0.005],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]
    ], dtype=np.float_)
    cost_per_step = np.array([63595 / 365,
                              456,
                              (19.1 / 100 * 15 + (1 - 19.1 / 100) * 2.15),
                              163595 / 365 * 0.02,
                              79],
                             dtype=np.float_)

    cellmachine = CaseMap(a, rule_array, cost_per_step)
    data = np.zeros([1, 5], dtype=np.int_)
    cost = np.zeros([1, 1], dtype=np.float_)
    for i in range(120):
        data = np.row_stack((data, cellmachine.step_main()))
        # cellmachine.step_main()
        cost = np.row_stack((cost, cellmachine.cost_total))

    cellmachine.show()
    print(data)
    print(cellmachine.statics)
    cnt = data[1:, 1] + data[1:, 2]
    draw_pic(data, cost, 120)
    plt.xlabel('time')
    plt.ylabel('epoches')
    plt.title('Comparision')
    sns.kdeplot(np.array((40, 42, 43, 54, 58, 58, 59, 59, 67)), shade= True)
    plt.savefig("img/Comparision.png",dpi=800)


    plt.show()
