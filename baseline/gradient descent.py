import pandas as pd
import numpy as np


class BaselineCFBySGD(object):
    def __init__(self, epochs, alpha, reg, dataset, columns=None):
        """
        构造函数
        :param epochs: 迭代次数
        :param alpha: 学习率
        :param reg: 正则化系数
        :param dataset: 数据集，"uid-iid-rating"
        :param columns: 数据集中的三个类型
        """
        self.dataset = dataset
        if columns is None:
            self.columns = ["userId", "itemId", "rating"]
        else:
            self.columns = columns
        self.epochs = epochs
        self.alpha = alpha
        self.reg = reg
        # 将数据集用uid分组，每一列为一个列表，共两列，存储iid和rating
        self.users_ratings = dataset.groupby(self.columns[0]).agg([list])
        self.items_ratings = dataset.groupby(self.columns[1]).agg([list])
        # 计算全局平均分
        self.global_mean = self.dataset[self.columns[2]].mean()

    def fit(self):
        """
        训练
        :return: None
        """
        self.bu, self.bi = self.sgd()

    def sgd(self):
        """
        随机梯度下降求解优化bu和bi
        :return: bu，bi
        """
        # 初始化bu，bi,每个用户和物品偏置值均初始化为0
        bu = dict(zip(self.users_ratings.index, np.random.rand(len(self.users_ratings))))
        bi = dict(zip(self.items_ratings.index, np.random.rand(len(self.items_ratings))))

        for i in range(self.epochs):
            print(f"iter{i + 1}")
            for uid, iid, rating in self.dataset.itertuples(index=False):
                error = rating - (self.global_mean + bu[uid] + bi[iid])
                # +是因为梯度取了相反数
                bu[uid] += self.alpha * (error - self.reg * bu[uid])
                bi[iid] += self.alpha * (error - self.reg * bi[iid])

        return bu, bi

    def predict(self, uid, iid):
        predict_rating = self.global_mean + self.bu[uid] + self.bi[iid]
        return predict_rating


if __name__ == "__main__":
    dtype = [("userId", np.int32), ("movieId", np.int32), ("rating", np.float32)]
    datasets = pd.read_csv("../datasets/ml-latest-small/ratings.csv", usecols=range(3),
                           dtype=dict(dtype))
    bcf = BaselineCFBySGD(30, 0.05, 0.3, datasets, ["userId", "movieId", "rating"])
    bcf.fit()

    uid = 1
    iid = 10
    print(bcf.predict(uid, iid))
