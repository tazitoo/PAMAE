from sklearn.datasets import make_blobs
from matplotlib import pyplot
import numpy as np
import random
import pickle
import copy


def euclid_distance(x, y):
    # 欧式距离
    # Euclidean distance
    return np.sqrt(sum(np.square(x - y)))


def assign_points(data, centroids):
    """
    将所有数据点划分到距离最近的中心
    Divide all data points to the nearest center
    :param data:数据集
    :param data:data set
    :param centroids:中心点集合
    :param centroids:Central point collection
    :return:
    """
    cluster_points = [[centroid] for centroid in centroids]
    labels = []
    for point in data:
        # 计算节点point到每个中心的距离，并将其划分到最近的中心点
        # Calculate the distance from the node point to each center and divide it to the nearest center point
        distances = [euclid_distance(point, centroid) for centroid in centroids]
        label = np.argmin(distances)  # 选择距离最近的簇中心 Select the nearest cluster center
        labels.append(label)
        cluster_points[label].append(point)  # 将point加入距离最近的簇中 Add point to the nearest cluster
    return labels, cluster_points


def pam(data, centroid_num):
    """
    kmedoids算法
    :param data: 待聚类的数据集
    :param centroid_num: 簇中心个数
    :return:
    """
    # 随机初始簇中心
    # Random initial cluster center
    indexs = list(range(len(data)))
    random.shuffle(indexs)
    init_centroids_index = indexs[:centroid_num]
    centroids = data[init_centroids_index, :]  # 中心点的数组 Array of center points
    labels = []  # 每个数据的类别标签  Category label for each data
    stop_flag = False  # 算法停止迭代的标志 A sign that the algorithm stops iterating
    while not stop_flag:
        stop_flag = True
        cluster_points = [[centroid] for centroid in centroids]  # 第i个元素，为第i类数据点的集合 The i-th element is a collection of data points of the i-th type
        labels = []  # 每个数据的类别标签 Category label for each data
        # 遍历数据 Iterate over the data
        for point in data:
            # 计算节点point到每个中心的距离，并将其划分到最近的中心点
            # Calculate the distance from the node point to each center and divide it to the nearest center point
            distances = [euclid_distance(point, centroid) for centroid in centroids]
            label = np.argmin(distances)  # 选择距离最近的簇中心 Select the nearest cluster center
            labels.append(label)
            cluster_points[label].append(point)  # 将point加入距离最近的簇中 Add point to the nearest cluster

        # 计算出当前中心点和其他所有点的距离总和
        # Calculate the total distance between the current center point and all other points
        distances = []
        for i in range(centroid_num):
            distances.extend([euclid_distance(point_1, centroids[i]) for point_1 in cluster_points[i]])
        old_distances_sum = sum(distances)

        # 尝试让整个数据集的每个非中心点替换中心点，若聚类误差降低，则改变中心点
        # Try to replace the center point with every non-central point in the entire data set. If the clustering error is reduced, change the center point
        for i in range(centroid_num):
            # # 计算第i个簇中，每个节点到原始簇中心的距离
            #Calculate the distance from each node to the center of the original cluster in the i-th cluster
            for point in data:
                new_centroids = copy.deepcopy(centroids)  # 假设的中心集合 Hypothetical center set
                new_centroids[i] = point
                labels, cluster_points = assign_points(data, new_centroids)
                # 计算新的聚类误差
                # Calculate new clustering error
                distances = []
                for j in range(centroid_num):
                    distances.extend([euclid_distance(point_1, new_centroids[j]) for point_1 in cluster_points[j]])
                new_distances_sum = sum(distances)

                # 判断聚类误差是否降低
                # Determine whether the clustering error is reduced
                if new_distances_sum < old_distances_sum:
                    old_distances_sum = new_distances_sum
                    centroids[i] = point  # 修改第i个簇的中心 Modify the center of the i-th cluster
                    stop_flag = False
    return centroids, labels, old_distances_sum
