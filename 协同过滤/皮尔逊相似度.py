import copy

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


def rating_by_pearson(similarities, dataframe):
    # 计算新的评分表格
    new_dataframe = copy.deepcopy(dataframe)
    for i in similarities.index:
        # 取出每一列数据,去掉自身
        similarity = similarities.loc[i].drop([i])
        similarity_sorted = similarity.sort_values(ascending=False)
        # 得到相似度最大的两个
        top2 = similarity_sorted[:2]

        # 得到该用户对商品的评分
        rate = new_dataframe.loc[i]
        # 获取未评分的数据的列索引
        idx = rate.index[np.where(np.isnan(rate))]
        # 计算加权评分
        for j in idx:
            new_dataframe.loc[i, j] = (new_dataframe.loc[top2.index[0], j] * top2[0] +
                                       new_dataframe.loc[top2.index[1], j] * top2[1]) / \
                                      (top2[0] + top2[1])
            print(f"物品{j}的评分为：{new_dataframe.loc[i, j].round(4)}")


if __name__ == "__main__":
    users = ["User1", "User2", "User3", "User4", "User5"]
    items = ["ItemA", "ItemB", "ItemC", "ItemD", "ItemE"]
    # 用户评分记录数据集,行为用户，列为评分
    datasets = [[5, 3, 4, 4, np.nan],
                [3, 1, 2, 3, 3],
                [4, 3, 4, 3, 5],
                [3, 3, 1, 5, 4],
                [1, 5, 5, 2, 1]]
    df = pd.DataFrame(datasets, columns=items, index=users)
    # 直接计算皮尔逊相似度，但是由于默认是列计算，若要计算用户间的相似度，需要转置
    user_similarity = df.T.corr()
    print(user_similarity)

    item_similarity = df.corr()
    print(item_similarity)

    # 根据用户相似度评分
    rating_by_pearson(user_similarity, df)
    # 根据物品相似度评分，所以需要转置
    rating_by_pearson(item_similarity, df.T)
