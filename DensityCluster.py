# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 08:47:35 2019

@author: ziyi
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import time
import csv

class DensityCluster:
    def __init__(self, n_clusters=7, dc_times=1):
        self.n_clusters = n_clusters
        self.dc_times = dc_times
    
    def fit_predict(self, data):
        self.xy = data
        
        self.dij = [[self.dist(i, j) for i in data] for j in data]
        
        self.pointscount = len(self.dij)

        self.dmax = max(self.dij)
# =============================================================================
#         生成截断距离，从距离数组中选取截断距离，选取数组1%-2%位置的数据为截断距离或者选择距离平均值为截断距离
# =============================================================================
        self.dc = self.dc_times * np.average(self.dij) / 15
# =============================================================================
#         生成密度列表并进行排序，保留对应的点下标
# =============================================================================
        self.gen_rou()
        self.dense_sort = sorted(self.dense, key=(lambda x:x[0]), reverse=True)
        
# =============================================================================
#         按密度降序排好序的点下标，
# =============================================================================
        self.dense_sort_index = [i[1] for i in self.dense_sort]
        
# =============================================================================
#         密度列表ρ，按编号
# =============================================================================
        self.dense_dense = [i[0] for i in self.dense]
        
# =============================================================================
#         生成最小距离列表б，按编号
# =============================================================================
        self.gen_segema()
        
# =============================================================================
#         生成密度与最小距离的综合衡量指标γ
# =============================================================================
        self.gama = np.multiply(np.array(self.dense_dense), np.array(self.segema)).tolist()
        #print('self.gama:', self.gama)
        
# =============================================================================
#         生成按降序排序的γ列表
# =============================================================================
        self.gama_sort = sorted(self.gama, reverse=True)
        
# =============================================================================
#         生成聚类中心
# =============================================================================
        self.gen_center()
        
# =============================================================================
#         进行分类
# =============================================================================
        self.gen_cluster()

# =============================================================================
#         画图
# =============================================================================
        self.draw()
        
# =============================================================================
#         生成csv结果文件
# =============================================================================
        self.gen_csv()
        
# =============================================================================
#         gen_cluster(self)
#         对所有点按聚类中心进行分类
#         遍历按降序排序的密度列表，归类到距离最近的点所属于的集合
# =============================================================================
    def gen_cluster(self):
        self.typeno = [-1 if i not in self.centerindex else i for i in range(0, self.pointscount)]
        for i in range(0, self.pointscount):
            if self.typeno[self.dense_sort_index[i]] == -1:
                self.typeno[self.dense_sort_index[i]] = self.typeno[self.minindex[self.dense_sort_index[i]]]       
        
# =============================================================================
#     gen_center(self)
#     我们从数据点图中看出有7个聚类中心，因此从排好序的γ列表中选取前7个为聚类中心，并生成聚类中心列表，存储编号
#     #或者从egema列表中选取
#     生成聚类中心
# =============================================================================
    def gen_center(self):

        self.centerindex = [0] * self.n_clusters
        for i in range(0, self.n_clusters):
            self.centerindex[i] = self.gama.index(self.gama_sort[i])

        print('centerindex:', self.centerindex)        
        
# =============================================================================
#     def draw(self):
#     画图
# =============================================================================
    def draw(self):
        xx = self.xy[:,0]
        yy = self.xy[:,1]
        plt.title('x-y points graph')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.plot(xx, yy, 'o')
        plt.show()
        
        typeno = [-1 if i not in self.centerindex else i for i in range(0, self.pointscount)]
        centerindex = self.centerindex[:]
        centerindex.append(-1)
        colors = ['#FFB6C1', '#800080', '#0000FF', '#DBDB70', '#8FBC8F', '#FF7F00', '#2F4F4F', '#CCCCCC']
        plt.title('x-y center points graph')
        plt.xlabel('x')
        plt.ylabel('y')
        for i in range(0, self.xy.shape[0]):
            plt.plot(self.xy[i][0], self.xy[i][1], 'o', color=colors[centerindex.index(typeno[i])])
        plt.show()
               
        colors = ['#FFB6C1', '#800080', '#0000FF', '#DBDB70', '#8FBC8F', '#FF7F00', '#2F4F4F']
        plt.title('x-y points cluster graph; dc=' + str(self.dc))
        plt.xlabel('x')
        plt.ylabel('y')
        for i in range(0, self.xy.shape[0]):
            plt.plot(self.xy[i][0], self.xy[i][1], 'o', color=colors[self.centerindex.index(self.typeno[i])])
        plt.savefig("clusterGraph" + str(time.time()) + '.png')
        plt.show()
        
        plt.title('ρ-б graph')
        plt.xlabel('ρ')
        plt.ylabel('б')
        plt.plot(self.dense_dense,self.segema,'o')
        plt.show()
        
# =============================================================================
#         plt.title('γ graph')
#         plt.xlabel('No.')
#         plt.ylabel('γ')
#         plt.plot(self.gama,'o')
#         plt.show()   
# =============================================================================

        plt.title('γ-sort graph')
        plt.xlabel('No.')
        plt.ylabel('γ')
        plt.plot(self.gama_sort,'o')
        plt.show()   
    
# =============================================================================
#     def gen_csv(self):
#     生成csv结果文件
# =============================================================================
    def gen_csv(self):
        with open('result.csv', 'w', newline='') as csvout:
            writer = csv.writer(csvout)
            for i in range(0, self.pointscount):
                writer.writerow([self.xy[i][0], self.xy[i][1], self.centerindex.index(self.typeno[i])])
        
        
# =============================================================================
#     dist(self, x, y)
#     x: [i, j]
#     y: [i, j]
#     计算两个点的最短距离
# =============================================================================
        
    def dist(self, x, y):
        return np.sqrt(np.power(x[0]-y[0],2)+np.power(x[1]-y[1],2))
    
    
# =============================================================================
#     gen_dist(self, xy)
#     计算所有点之间的距离矩阵并保存为文本
#     xy: [[i, j],..] 点集合
#     对距离矩阵进行排序，并转换列表，删除重复元素，保存成文本
# =============================================================================
        
    def gen_dist(self, xy):
        dij = [[self.dist(i, j) for i in xy] for j in xy]
        np.savetxt('dist.txt',dij)

        dij_sort = [self.dist(i, j) for i in xy for j in xy]
        dij_sort.sort()
        np.savetxt('dist_sort.txt', dij_sort[::2])
    
# =============================================================================
#     gen_segema(self)
#     计算每个点距离 密度比自身大的那些点之间距离的最小距离
#     遍历降序排列的密度下标数组，从之前的点的距离中选出最小的
# =============================================================================
    def gen_segema(self):
        self.segema = []
        dijmin = [0 for i in range(0, len(self.dense_sort_index))]
        
        self.minindex = [0 for i in range(0, len(self.dense_sort_index))]

        for i in range(1, len(self.dense_sort_index)):
                                    
                dij = [self.dij[self.dense_sort_index[i]][self.dense_sort_index[j]] for j in range(0, i)]
                mindist = min(dij)
                mindistindex = self.dense_sort_index[i]
                dijmin[mindistindex] = mindist
                
                self.minindex[mindistindex] = self.dense_sort_index[dij.index(mindist)]
                
        #print(dijmin)
        dijmin[self.dense_sort_index[0]] = max(dijmin)
        self.segema[:]=dijmin[:]
        
        self.minindex[self.dense_sort_index[0]] = 0

                        
    def gaussian_kernel(self, distance, dc):
        return np.power(math.e, -np.power(distance/dc,2))
        
    def cut_off_kernel(self, distance, dc):
        return 1 if (distance - dc) < 0 else 0
        
# =============================================================================
#     gen_rou(self)
#     生成密度列表
#     计算每个点截断距离内的密度值
#     有两种方式 cut-off与gaussian
#     cut-off是截断距离内点的个数 离散函数
#     gaussian是截断距离内密度的估量 连续函数
# =============================================================================
    def gen_rou(self):
        self.dense = []

        for i in range(0, self.pointscount):
            kernel = [self.cut_off_kernel(x, self.dc) for x in self.dij[i]]
            #kernel = [self.gaussian_kernel(x, self.dc) for x in self.dij[i]]
            self.dense.append([sum(kernel),i])
            #print(kernel)
            
            
if __name__ == '__main__':
    data = np.loadtxt('Aggregation.txt', delimiter=',')
    mc = DensityCluster(n_clusters=7)   
    mc.fit_predict(data)
    

