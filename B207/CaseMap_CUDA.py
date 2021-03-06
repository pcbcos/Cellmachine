from numba import cuda
import numba as nb
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import time
import math
import os


@cuda.jit
def step_in_gpu(data_in_gpu, statics_in_gpu, rand_arr_in_gpu, rand_arr2_in_gpu, rules_in_gpu,
                illness_state_in_gpu, next_data_in_gpu,timer_in_gpu,cost_per_step):
    m = cuda.threadIdx.x + 2
    n = cuda.blockIdx.x + 2
    count = 0
    if data_in_gpu[m, n] == 0:
        cuda.atomic.add(statics_in_gpu, (0, 0), 1)
        r = 1 - (1 - rules_in_gpu[0, 1]) ** (illness_state_in_gpu[m, n])
        if rand_arr_in_gpu[m, n] <= r:
            next_data_in_gpu[m, n] = 1
            cuda.atomic.add(timer_in_gpu, (0,4), 1)

        else:
            next_data_in_gpu[m, n] = 0
            if rand_arr2_in_gpu[m, n] <= rules_in_gpu[0, 3]:
                next_data_in_gpu[m, n] = 3
                cuda.atomic.add(timer_in_gpu, (0,1), 1)
                

    if data_in_gpu[m, n] == 1:
        cuda.atomic.add(statics_in_gpu, (0, 1), 1)
        r = rand_arr_in_gpu[m, n]
        if r <= rules_in_gpu[1, 1]:
            next_data_in_gpu[m, n] = 1
            cuda.atomic.add(timer_in_gpu, (0,4), 1)

        elif r <= rules_in_gpu[1, 1] + rules_in_gpu[1, 2]:
            next_data_in_gpu[m, n] = 2
        else:
            next_data_in_gpu[m, n] = 4
            cuda.atomic.add(timer_in_gpu, (0,3), 1)

    if data_in_gpu[m, n] == 2:
        cuda.atomic.add(statics_in_gpu, (0, 2), 1)
        r = rand_arr_in_gpu[m, n]
        if r <= rules_in_gpu[2, 2]:
            next_data_in_gpu[m, n] = 2
        elif r <= rules_in_gpu[2, 2] + rules_in_gpu[2, 3]:
            next_data_in_gpu[m, n] = 3
            cuda.atomic.add(timer_in_gpu, (0,1), 1)
        else:
            next_data_in_gpu[m, n] = 4
            cuda.atomic.add(timer_in_gpu, (0,3), 1)

    if data_in_gpu[m, n] == 3:
        cuda.atomic.add(statics_in_gpu, (0, 3), 1)
        next_data_in_gpu[m, n] = 3
        cuda.atomic.add(timer_in_gpu, (0,1), 1)

    if data_in_gpu[m, n] == 4:
        cuda.atomic.add(statics_in_gpu, (0, 4), 1)
        next_data_in_gpu[m, n] = 4
        cuda.atomic.add(timer_in_gpu, (0,3), 1)


@cuda.jit
def search_in_gpu(search_result_in_gpu, data_in_gpu):
    m = cuda.blockIdx.x
    n = cuda.threadIdx.x
    for i in range(m - 2, m + 3):
        for j in range(n - 2, n + 3):
            if i != m and j != n and data_in_gpu[i, j] == 1:
                search_result_in_gpu[m, n] += 1


class CaseMap:
    def __init__(self, origin: np.array, rule_array: np.array,cost_per_step):
        np.random.seed(2020)
        self.rules = rule_array.copy()
        self.data = origin
        self.shape = [r + 4 for r in list(np.shape(self.data))]
        random_side_row = np.zeros((2, self.shape[1] - 4), dtype=int)
        random_side_column = np.zeros((self.shape[0], 2), dtype=int)
        self.data = np.row_stack((random_side_row, self.data))
        self.data = np.row_stack((self.data, random_side_row))
        self.data = np.column_stack((self.data, random_side_column))
        self.data = np.column_stack((random_side_column, self.data))  # ?????????array??????????????????+2?????????+2
        self.Time = 0
        self.statics = np.zeros([1, 5], dtype=np.int64)
        self.im0 = 0
        self.im1=0

        self.flag0 = False  # ??????0:?????????????????????????????????p01?????? (????????????)
        self.flag1 = False  # ??????1:?????????????????????????????????,p12?????????p23??????,p03?????? (???????????????) (?????????)
        self.flag2 = False  # ??????3:?????????????????????????????????????????????????????????
        self.timer=np.array([[0, 0, 0, 0, 0]],dtype=np.int64) # timer for vaccine, word, test, death, isolotion is with per word.once there is one, plus 1
        self.cost_per_step=cost_per_step
        self.cost_total=0

    def event_eff(self):
        if (not self.flag0) and self.statics[0, 2] > 700:
            self.rules[0, 1] *= 0.5
            self.flag0 = True
            print("??????????????????")

        if (self.flag0) and self.statics[0, 2] < 700:
            self.rules[0, 1] *= 2
            self.flag0 = False
            print("??????????????????")

        if (not self.flag1) and self.statics[0, 2] > 1500:
            self.rules[1, 2] *= 0.5
            self.rules[2, 3] *= 0.5
            self.rules[0, 3] *= 0.5
            self.rules[2, 4] *= 1.2
            self.flag1 = True
            print("???????????????")

        if (self.flag1) and self.statics[0, 2] < 1500:
            self.rules[1, 2] *= 2
            self.rules[2, 3] *= 2
            self.rules[0, 3] *= 2
            self.rules[2, 4] /=1.2
            self.flag1 = False
            print("??????????????????")


        if (not self.flag2):
            self.flag2 = True

        if (self.flag2) and self.im1-self.im0<5 and self.statics[0, 4]< 15000:
            print("??????????????????")
            print(self.Time)
            self.flag2 = False

    def search_main(self):  # ?????????????????????5*5??????????????????
        self.data_in_gpu = cuda.to_device(self.data)
        self.search_result = np.zeros_like(self.data, dtype=np.int32)
        self.search_result_in_gpu = cuda.to_device(self.search_result)

        cuda.synchronize()
        search_in_gpu[1024, 1024](self.search_result_in_gpu, self.data_in_gpu)
        cuda.synchronize()

        self.search_result = self.search_result_in_gpu.copy_to_host()
        return self.search_result

    def step_main(self):
        
        self.next_data = np.zeros_like(self.data, dtype=np.int_)
        self.randomarray = np.random.random_sample(np.shape(self.data))
        self.randomarray2 = np.random.random_sample(np.shape(self.data))
        self.timer_in_gpu=cuda.to_device(self.timer)
        self.cost_per_step_in_gpu=cuda.to_device(self.cost_per_step)
        self.statics = np.zeros([1, 5], dtype=np.int64)
        self.rules_in_gpu = cuda.to_device(self.rules)
        self.statics_in_gpu = cuda.to_device(self.statics)
        self.data_in_gpu = cuda.to_device(self.data)
        self.next_data_in_gpu = cuda.to_device(self.next_data)
        self.rand_arr_in_gpu = cuda.to_device(self.randomarray)
        self.rand_arr2_in_gpu = cuda.to_device(self.randomarray2)
        self.illness_state_in_gpu = cuda.to_device(self.search_main())

        threads_per_block = 1024
        blocks_per_grid = np.shape(self.data)[1] - 4
        cuda.synchronize()

        step_in_gpu[1024, 1024](self.data_in_gpu,
                                self.statics_in_gpu,
                                self.rand_arr_in_gpu,
                                self.rand_arr2_in_gpu,
                                self.rules_in_gpu,
                                self.illness_state_in_gpu,
                                self.next_data_in_gpu,
                                self.timer_in_gpu,
                                self.cost_per_step_in_gpu)

        cuda.synchronize()

        self.next_data = self.next_data_in_gpu.copy_to_host()
        self.data = self.next_data
        self.statics = self.statics_in_gpu.copy_to_host()
        self.timer=self.timer_in_gpu.copy_to_host()
        self.cost_per_step=self.cost_per_step_in_gpu.copy_to_host()
        self.timer[0, 0] = self.statics[0, 1] + self.statics[0, 2]
        self.timer[0, 2] = self.statics[0, 1] + self.statics[0, 2]
        self.im1=self.statics[0,1]+self.statics[0,2]
        self.cost_total=np.dot(self.timer,self.cost_per_step)
        self.rules[0, 1] *= (1+self.statics[0,1]/60000+self.statics[0,2]/60000)  # ???????????????????????????????????????          1.034
        self.rules[0, 3] *= 0.98  # ????????????                             1
        self.Time += 1
        self.event_eff()
        self.im1=self.im0
        return self.statics


    def show(self):
        try:
            os.mkdir("img/heatmap")
        except:
            pass
        fig_name = 'heatmap/heatmap-{}.png'.format(self.Time)
        fig_path = "img" + '/' + fig_name
        fig = sns.heatmap(self.data, annot=False, cmap="Blues")
        heatmap = fig.get_figure()
        heatmap.savefig(fig_path, dpi=1600)
        plt.close(heatmap)
