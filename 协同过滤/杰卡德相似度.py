import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


if __name__ == "__main__":
    users = ["User1", "User2", "User3", "User4", "User5"]
    items = ["ItemA", "ItemB", "ItemC", "ItemD", "ItemE"]
    # 用户购买记录数据集
    datasets = [[1, 0, 1, 1, 0],
                [1, 0, 0, 1, 1],
                [1, 0, 1, 0, 0],
                [0, 1, 0, 1, 1],
                [1, 1, 1, 0, 1]]
    df = pd.DataFrame(datasets, columns=items, index=users)

    # 1-两两之间的距离得到杰卡德相似度
    user_similarity = 1 - pairwise_distances(df.values, metric="jaccard")
    user_similarity = pd.DataFrame(user_similarity, columns=users, index=users)

    # 物品之间的杰卡德相似度
    item_similarity = 1 - pairwise_distances(df.T.values, metric="jaccard")
    item_similarity = pd.DataFrame(item_similarity, columns=items, index=items)

    # 根据用户之间的相似度推荐物品
    top_n_users = dict()
    # 找到与自身最相似的n个
    for i in user_similarity.index:
        # 取出每一列数据，并去掉自己
        similarity = user_similarity.loc[i].drop([i])
        # 排序，False表示降序
        similarity_sorted = similarity.sort_values(ascending=False)
        # 得到相似度最大的两个
        top2 = list(similarity_sorted.index[:2])
        top_n_users[i] = top2

    rs_results = dict()
    for user, sim_users in top_n_users.items():
        rs_result = set()
        for sim_user in sim_users:
            sim_user_row = df.loc[sim_user]
            sim_user_row[sim_user_row == 0] = np.nan
            rs_result = rs_result.union(set(sim_user_row.dropna().index))

        user_row = df.loc[user]
        user_row[user_row == 0] = np.nan
        rs_result -= set(user_row.dropna().index)
        rs_results[user] = rs_result

    print(f"基于用户相似度的推荐结果：{rs_results}")

    top_n_items = dict()
    # 找到与自身最相似的n个
    for i in item_similarity.index:
        # 取出每一列数据，并去掉自己
        similarity = item_similarity.loc[i].drop([i])
        # 排序，False表示降序
        similarity_sorted = similarity.sort_values(ascending=False)
        # 得到相似度最大的两个
        top2 = list(similarity_sorted.index[:2])
        top_n_items[i] = top2

    rs_results = dict()
    for user in df.index:
        rs_result = set()
        items = df.loc[user]
        items[items == 0] = np.nan
        items = items.dropna().index
        for item in items:
            rs_result = rs_result.union(top_n_items[item])

        # 过滤掉重复的
        rs_result -= set(items)
        rs_results[user] = rs_result

    print(f"基于物品相似度的推荐结果：{rs_results}")






