import numpy as np
import pandas as pd


def data_split(data_path, training_rate=0.8, random=False):
    """
    分割训练集和测试集
    :param data_path: 数据路径
    :param training_rate: 训练集的比例
    :param random: 是否随机打乱数据
    :return:
    """
    print("开始切分数据集：")
    dtype = {"userId": np.int32, "movieId": np.int32, "rating": np.float32}
    ratings = pd.read_csv(data_path, usecols=range(3), dtype=dtype)
    testset_index = []

    # 需要保证每个用户在训练集和测试集中都存在，所以按userId进行聚合
    for uid in ratings.groupby("userId").any().index:
        # 得到用户所有已经评分的数据（dropna的作用）,是一个三列的dataframe，第一列为userId，第二列为movieId，第三列为rating
        user_rating_data = ratings.where(ratings["userId"] == uid).dropna()
        if random:
            # dataframe不能打散，所以转成list
            index = list(user_rating_data.index)
            np.random.shuffle(index)
            # 测试集开始的位置
            start = round(len(user_rating_data) * training_rate)
            testset_index += list(index[start:])
        else:
            start = round(len(user_rating_data) * training_rate)
            testset_index += list(user_rating_data.index[start:])

    test_set = ratings.loc[testset_index]
    train_set = ratings.drop(testset_index)
    print("完成数据集切分")
    return train_set, test_set


def evaluate(predict_results, method="all"):
    """
    评估
    :param predict_results: 预测结果，含有uid，iid，real_rating,predict_rating
    :param method: 指标方法，all表示全部计算
    :return:
    """
    def rmse(predict_results):
        """
        rmse评估指标
        :param predict_results:
        :return:
        """
        length = 0
        loss_sum = 0
        for uid, iid, real_rating, predict_rating in predict_results:
            length += 1
            loss_sum += (predict_rating - real_rating) ** 2
        return round(np.sqrt(loss_sum / length), 4)

    def mae(predict_results):
        """
        mae评估指标
        :param predict_results:
        :return:
        """
        length = 0
        loss_sum = 0
        for uid, iid, real_rating, predict_rating in predict_results:
            length += 1
            loss_sum += np.abs(predict_rating - real_rating)
        return round(loss_sum / length, 4)

    def rmse_and_mae(predict_results):
        length = 0
        mae_loss_sum = 0
        rmse_loss_sum = 0
        for uid, iid, real_rating, predict_rating in predict_results:
            length += 1
            mae_loss_sum += np.abs(predict_rating - real_rating)
            rmse_loss_sum += (predict_rating - real_rating) ** 2
        return round(np.sqrt(rmse_loss_sum / length), 4), round(mae_loss_sum / length, 4)

    if method == "all":
        return rmse_and_mae(predict_results)
    elif method == "rmse":
        return rmse(predict_results)
    elif method == "mae":
        return mae(predict_results)
    else:
        raise Exception("不存在的评估方式")


class BaselineCFByALS(object):
    def __init__(self, epochs, reg1, reg2, dataset, columns=None):
        """
        构造函数
        :param epochs: 迭代次数
        :param reg1: 用户偏置正则化系数
        :param reg2: 物品偏置正则化系数
        :param dataset: 数据集，"uid-iid-rating"
        :param columns: 数据集中的三个类型
        """
        self.dataset = dataset
        if columns is None:
            self.columns = ["userId", "itemId", "rating"]
        else:
            self.columns = columns
        self.epochs = epochs
        self.reg1 = reg1
        self.reg2 = reg2
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
        self.bu, self.bi = self.als()

    def als(self):
        """
        随机梯度下降求解优化bu和bi
        :return: bu，bi
        """
        # 初始化bu，bi,每个用户和物品偏置值均初始化为0
        bu = dict(zip(self.users_ratings.index, np.random.rand(len(self.users_ratings))))
        bi = dict(zip(self.items_ratings.index, np.random.rand(len(self.items_ratings))))

        for i in range(self.epochs):
            print(f"iter{i + 1}")
            # iids和ratings都是一个列表
            for uid, iids, ratings in self.users_ratings.itertuples(index=True):
                _sum = 0
                for iid, rating in zip(iids, ratings):
                    _sum += rating - self.global_mean - bi[iid]
                # 简化的方式，假设每个偏置都相等
                bu[uid] = _sum / (self.reg1 + len(iids))

            for iid, uids, ratings in self.items_ratings.itertuples(index=True):
                _sum = 0
                for uid, rating in zip(uids, ratings):
                    _sum += rating - self.global_mean - bu[uid]
                bi[iid] = _sum / (self.reg2 + len(uids))

        return bu, bi

    def predict(self, uid, iid):
        if iid not in self.items_ratings.index:
            raise Exception(f"训练集中不存在{iid}项目")
        predict_rating = self.global_mean + self.bu[uid] + self.bi[iid]
        print(f"用户{uid}对项目{iid}的评分为{predict_rating}")
        return predict_rating

    def test(self, test_set):
        """
        预测测试集中的得分
        :param test_set:
        :return:
        """
        for uid, iid, real_rating in test_set.itertuples(index=False):
            # try尝试执行，有错误跳转到except
            try:
                pred = self.predict(uid, iid)
            except Exception as e:
                print(e)
            # 没有错误时会执行
            else:
                yield uid, iid, real_rating, pred


if __name__ == "__main__":
    training_set, testing_set = data_split("../datasets/ml-latest-small/ratings.csv",
                                           random=True)
    bcf = BaselineCFByALS(30, 0.1, 0.1, training_set, ["userId", "movieId", "rating"])
    bcf.fit()
    pred_result = bcf.test(testing_set)

    RMSE, MAE = evaluate(pred_result)
    print(f"RMSE:{RMSE}, MAE:{MAE}")


