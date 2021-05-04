import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import time


class CaseMap:
    def __init__(self, origin: np.array, rule_array: np.array):
        self.rules = rule_array
        self.data = origin
        self.shape = [r + 4 for r in list(np.shape(self.data))]
        random_side_row = np.zeros((2, self.shape[1] - 4), dtype=int)
        random_side_column = np.zeros((self.shape[0], 2), dtype=int)
        self.data = np.row_stack((random_side_row, self.data))
        self.data = np.row_stack((self.data, random_side_row))
        self.data = np.column_stack((self.data, random_side_column))
        self.data = np.column_stack((random_side_column, self.data))  # 以上为array拓宽，即行数+2，列数+2
        self.Time = 0
        self.statics = [0, 0, 0, 0, 0]

    def __search__(self, m, n):  # 统计每一单元格5*5范围内的情况
        temp_result = 0  # 未感染  携带者  确诊者（已隔离） 已注射疫苗者(感染康复者)
        temp_data = self.data[m - 2:m + 3, n - 2:n + 3]
        temp_data[2, 2] = 6
        temp_result = np.sum(temp_data == 1)
        return temp_result

    def step_main(self):
        time_start = time.time()
        self.next_data = np.zeros_like(self.data)
        self.randomarray = np.random.random_sample(np.shape(self.data))
        self.randomarray2 = np.random.random_sample(np.shape(self.data))
        self.statics = [0, 0, 0, 0, 0]

        for m in range(2, self.shape[0] - 2):
            for n in range(2, self.shape[1] - 2):
                if self.data[m, n] == 0:
                    self.statics[0] += 1
                    neigbor_state = self.__search__(m, n)
                    # p = neigbor_state * self.rules[0, 1]
                    p = 1 - (1 - self.rules[0, 1]) ** neigbor_state  # state 0->1
                    r = self.randomarray[m, n]
                    if r <= p:
                        self.next_data[m, n] = 1
                    else:
                        self.next_data[m, n] = 0
                        if self.randomarray2[m, n] <= self.rules[0, 3]:
                            self.next_data[m, n] = 3

                if self.data[m, n] == 1:
                    self.statics[1] += 1
                    r = self.randomarray[m, n]
                    if r <= self.rules[1, 1]:
                        self.next_data[m, n] = 1
                    elif r <= self.rules[1, 1] + self.rules[1, 2]:
                        self.next_data[m, n] = 2
                    else:
                        self.next_data[m, n] = 4

                if self.data[m, n] == 2:
                    self.statics[2] += 1
                    r = self.randomarray[m, n]
                    if r <= self.rules[2, 2]:
                        self.next_data[m, n] = 2
                    elif r <= self.rules[2, 2] + self.rules[2, 3]:
                        self.next_data[m, n] = 3
                    else:
                        self.next_data[m, n] = 4
                if self.data[m, n] == 3:
                    self.statics[3] += 1
                    self.next_data[m, n] = 3

                if self.data[m, n] == 4:
                    self.statics[4] += 1
                    self.next_data[m, n] = 4

        self.data = self.next_data
        self.rules[0, 1] *= 1.02  # 因病毒变异，每轮传染率扩大
        self.rules[0, 3] *= 0.95

        time_end = time.time()
        print((time_end - time_start))

        print("第{}步计算完成，正在出图...".format(self.Time))
        self.Time+=1
        return self.statics

    def show(self):
        fig_name = 'heatmap-{}.png'.format(self.Time)
        fig_path = "img" + '/' + fig_name
        fig = sns.heatmap(self.data, annot=False, cmap="Blues")
        heatmap = fig.get_figure()
        heatmap.savefig(fig_path, dpi=1000)
        plt.close(heatmap)
