import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix

from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender
from implicit.nearest_neighbours import bm25_weight


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS

    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    def __init__(self, data, weighted=True):

        self.FILTER_ID = 999999

        self.top_purchases = data.groupby(['item_id'])['quantity'].count().reset_index()
        self.top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.top_purchases = self.top_purchases.loc[self.top_purchases['item_id'] != self.FILTER_ID]

        self.user_item_matrix = self.prepare_matrix(data)
        (self.id_to_item_id, self.id_to_user_id,
         self.item_id_to_id, self.user_id_to_id) = self.prepare_dicts(self.user_item_matrix)

        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)

        if weighted:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

        self.model = self.fit(self.user_item_matrix)

    @staticmethod
    def prepare_matrix(data):
        user_item_matrix = pd.pivot_table(data,
                                          index='user_id', columns='item_id',
                                          values='quantity',
                                          aggfunc='count',
                                          fill_value=0
                                          )
        user_item_matrix = user_item_matrix.astype(float)
        return user_item_matrix

    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""

        user_ids = user_item_matrix.index.values
        item_ids = user_item_matrix.columns.values

        matrix_user_ids = np.arange(len(user_ids))
        matrix_item_ids = np.arange(len(item_ids))

        id_to_item_id = dict(zip(matrix_item_ids, item_ids))
        id_to_user_id = dict(zip(matrix_user_ids, user_ids))

        item_id_to_id = dict(zip(item_ids, matrix_item_ids))
        user_id_to_id = dict(zip(user_ids, matrix_user_ids))

        return id_to_item_id, id_to_user_id, item_id_to_id, user_id_to_id

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())

        return own_recommender

    def get_own_recommendations(self, user_id, rec_number=5):
        recommendations = self.own_recommender.recommend(userid=self.user_id_to_id[user_id],
                                                         user_items=csr_matrix(self.user_item_matrix).tocsr(),
                                                         N=rec_number,
                                                         filter_already_liked_items=False,
                                                         filter_items=None,
                                                         recalculate_user=False)
        recommendations = [self.id_to_item_id[rec[0]] for rec in recommendations]
        return recommendations

    def update_dict(self, user_id):
        if user_id not in self.user_id_to_id.values():
            max_id = max(list(self.user_id_to_id.values()))
            max_id += 1

            self.user_id_to_id.update({user_id: max_id})
            self.id_to_user_id.update({max_id: user_id})

    def get_similar_item(self, item_id):
        rec = self.model.similar_items(self.item_id_to_id[item_id], N=2)
        return self.id_to_item_id[rec[1][0]]

    def extend_rec_with_popular(self, recommendations, rec_number=5):
        if len(recommendations) < rec_number:
            recommendations.extend(self.top_purchases.item_id.tolist())
            recommendations = recommendations[:rec_number]
        return recommendations

    def get_recommendations(self, user_id, model, rec_number):
        self.update_dict(user_id)
        recommendations = model.recommend(userid=self.user_id_to_id[user_id],
                                          user_items=csr_matrix(self.user_item_matrix).tocsr(),
                                          N=rec_number,
                                          filter_already_liked_items=False,
                                          filter_items=[self.item_id_to_id[self.FILTER_ID]],
                                          recalculate_user=True)
        recommendations = [self.id_to_item_id[rec[0]] for rec in recommendations]
        recommendations = self.extend_rec_with_popular(recommendations, rec_number=rec_number)
        assert len(recommendations) == rec_number, f'Количество рекомендаций не равно {rec_number}'
        return recommendations

    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())

        return model

    def get_similar_items_recommendation(self, user_id, rec_number=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""
        own_items = self.get_own_recommendations(user_id, rec_number)

        recommendations = []

        for item_id in own_items:
            rec = self.model.similar_items(self.item_id_to_id[item_id], N=2)
            recommendations.append(self.id_to_item_id[rec[1][0]])

        assert len(recommendations) == rec_number, f'Количество рекомендаций != {rec_number}'
        return recommendations

    def get_similar_users_recommendation(self, user_id, n_users=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        users = self.model.similar_users(self.user_id_to_id[user_id], N=n_users + 1)
        users = [self.id_to_user_id[user[0]] for user in users][1:]

        recommendations = []

        for user in users:
            rec = self.get_own_recommendations(user, 1)[0]
            recommendations.append(rec)

        assert len(recommendations) == n_users, f'Количество рекомендаций != {n_users}'
        return recommendations

    # assert len(res) == N, f'Количество рекомендаций != {N}'
    # return res
