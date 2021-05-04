from CaseMap_CUDA import CaseMap
import numpy as np
import threading

data100 = []


def ONE_SEED(_seed_num: int):
    global data100
    np.random.seed(_seed_num)
    a = np.random.random_sample([1024, 1024])

    p = [0.001, 0.011, 0.001]
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i, j] < p[0]:
                a[i, j] = 1
            elif a[i, j] < p[0] + p[1]:
                a[i, j] = 2
            elif a[i, j] < p[0] + p[1] + p[2]:
                a[i, j] = 3
    a = a.astype(np.int_)
    rule_array = np.array([
        [0, 0.008, 0, 0.005, 0],
        [0, 0.695, 0.3, 0, 0.005],
        [0, 0, 0.79, 0.1, 0.005],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]
    ], dtype=np.float_)
    cellmachine = CaseMap(a, rule_array)
    data = np.zeros([1, 5], dtype=np.int_)
    print("running seed:{}".format(_seed_num))
    for i in range(120):
        cellmachine.step_main()
    temp = np.column_stack((np.array([[_seed_num]]), cellmachine.statics))
    data100.append(temp.tolist()[0])


def main():
    th = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    for x in range(16):
        r = threading.Thread(target=ONE_SEED, args=[x])
        th[x % 16].append(r)

    for i in range(100//16+1):
        for j in range(16):
            try:
                th[j][i].start()
            except:
                pass
        for j in range(16):
            try:
                th[j][i].join()
            except:
                pass


if __name__ == "__main__":
    # WALK_result = np.zeros((1, 5), dtype=np.int_)
    # for i in range(100):
    #     temp = ONE_SEED(i)
    #     WALK_result = np.row_stack((WALK_result, temp))
    #     print(WALK_result)
    main()
    data100.sort(key=lambda x: x[0])
    print(data100)
