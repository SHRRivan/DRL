import random
import pandas as pd
import numpy as np


class Kmeans():
    def __init__(self, k):
        self.k = k
        self.prioritized_times = 1

    def create_data(self):
        return [[1, 1], [1, 2], [2, 1], [6, 4], [6, 3], [5, 4], [2, 2], [4, 6]]
        # return [[1, 1, 1], [1, 2, 1], [2, 1, 1], [6, 4, 2], [6, 3, 3], [5, 4, 2]]  # 我的想法是构建[s,a,r]三元组或[s,a,r,s']

    def cal_dis(self, datas, centroids):
        result_list = []
        for data in datas:
            diff = np.tile(data, (self.k, 1)) - centroids  # 相减  (np.tile(a,(2,1))就是把a先沿x轴复制1倍（没复制)仍然是[0,1,2]
            squared_diff = diff ** 2                  # 再把结果沿y方向复制2倍得到array([[0,1,2],[0,1,2]]));运算后一维变二维数据
            squared_dis = np.sum(squared_diff, axis=1) # 和  (axis=1表示按行求和),
            distance = squared_dis ** 0.5             # 开根号;distance=[dis1,dis2，disk],分别表示距离k个质心的欧拉距离
            result_list.append(distance)
        result_list = np.array(result_list)           # 返回一个每个点到各质心的距离len(dateSet)*k的数组
        return result_list                            # 转np.array是因为后面要用np的函数，更方便

    def classify(self, data, centroids):
        cal_list = self.cal_dis(data, centroids)  # 计算样本到质心的距离
        # 分组并计算新的质心
        min_dist_indices = np.argmin(cal_list, axis=1)  # 按行求出各点距离更近的那个质心的标号，为0~k-1
        temp_centroids = pd.DataFrame(data).groupby(min_dist_indices).mean()
        # DataFramte(dataSet)对DataSet分组，groupby(min)按照min进行统计分类，mean()对分类结果求均值--为表格形式(字典)
        new_centroids = temp_centroids.values  # 返回新质心的坐标：min中相同元素求和再平均

        # 计算变化量
        # if len(new_centroids) == (self.k - 1):
        #     add_dimension = [new_centroids[0][0] / 2, new_centroids[0][1] / 2]
        #     new_centroids = np.append(new_centroids, [add_dimension], axis=0)
        changed = new_centroids - centroids

        return changed, new_centroids

    def get_result(self, data):
        centroids = random.sample(data, self.k)     # 在dataset中随机选取k个质心，形如[[1,1],[6,4]]
        changed, new_centroids = self.classify(data, centroids)  # 更新质心 直到变化量全为0
        while np.any(changed != 0):
            changed, new_centroids = self.classify(data, new_centroids)

        centroids = sorted(new_centroids.tolist())  # tolist()将矩阵转换成列表 sorted()排序

        # 根据质心计算每个集群
        cluster = []
        cal_list = self.cal_dis(data, centroids)    # 调用欧式距离
        min_dist_indices = np.argmin(cal_list, axis=1)
        for i in range(self.k):
            cluster.append([])
        for i, j in enumerate(min_dist_indices):    # enumerate()可同时遍历索引和遍历元素
            cluster[j].append(data[i])

        class_id = self.plot_2d(data=data, centroids=centroids, cluster=cluster)
        return centroids, cluster, class_id

        # return centroids, cluster

    def plot_3d(self, data, centroids, cluster):
        # print('质心为：%s' % centroids)
        # print('集群为：%s' % cluster)
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.set_xlabel('State')
        # ax.set_ylabel('Action')
        # ax.set_zlabel('Reward')
        # for i in range(len(data)):
        #     ax.scatter(data[i][0], data[i][1], data[i][2], marker='o', color='green', label='原始点')
        #     for j in range(len(centroids)):
        #         ax.scatter(centroids[j][0], centroids[j][1], centroids[j][2], marker='x', color='red', label='质心')
        # plt.show()

        class_1 = np.array(cluster[0]).sum(axis=0)
        class_2 = np.array(cluster[1]).sum(axis=0)
        if class_1.any() == 0.0:
            class_index = 1
        elif class_2.any() == 0.0:
            class_index = 0
        else:
            class_index = 0 if class_1[1] > class_2[1] else 1
        # print('Reward, Length in class_1 are:{}, {}'.format(class_1[2],len(cluster[0])))
        # print('Reward, Length in class_2 are:{}, {}'.format(class_2[2],len(cluster[1])))
        return class_index

    def plot_2d(self, data, centroids, cluster):
        # print('质心为： %s' % centroids)
        # print('集群为： %s' % cluster)
        #
        # for i in range(len(data)):
        #     plt.scatter(data[i][0], data[i][1], marker='o', color='green', s=40, label='原始点')
        #     #  记号形状       颜色      点的大小      设置标签
        #     for j in range(len(centroids)):
        #         plt.scatter(centroids[j][0], centroids[j][1], marker='x', color='red', s=50, label='质心')
        # plt.show()

        average_reward_0 = np.array(cluster[0]).mean(axis=0)[1]
        average_reward_1 = np.array(cluster[1]).mean(axis=0)[1]

        if average_reward_0 == 0.0:
            class_index = 1
        elif average_reward_1 == 0.0:
            class_index = 0
        else:
            self.prioritized_times += 1
            class_index = 0 if average_reward_0 > average_reward_1 else 1
        # print('Reward, Length in class_1 are:{}, {}'.format(class_1[2],len(cluster[0])))
        # print('Reward, Length in class_2 are:{}, {}'.format(class_2[2],len(cluster[1])))
        return class_index


if __name__ == '__main__':
    KM = Kmeans(k=2)
    data = KM.create_data()
    centroids, cluster, class_id = KM.get_result(data)

