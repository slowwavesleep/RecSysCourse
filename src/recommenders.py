import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix

from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender
from implicit.nearest_neighbours import bm25_weight

from lightgbm import LGBMClassifier

from src.utils import unique_list

from src.constants import dummy_id_filter

FILTER_ID = dummy_id_filter


class MainRecommender:
    """First level of the recommendation system. Can be used for initial filtering (selecting candidates)
     or as an independent model.

    Input
    -----
    data: pd.DataFrame
       Dataframe containing transactions (i.e. purchases) as rows.
       Must contain the following columns: `user_id`, `item_id`, `quantity`.
    weighted: Boolean
       Indicates whether resulting user-item matrix should be weighted.
    """

    def __init__(self, data, weighted=True):

        self.FILTER_ID = FILTER_ID

        self.top_purchases = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.top_purchases = self.top_purchases[self.top_purchases['item_id'] != self.FILTER_ID]

        self.overall_top_purchases = data.groupby('item_id')['quantity'].count().reset_index()
        self.overall_top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.overall_top_purchases = self.overall_top_purchases[self.overall_top_purchases['item_id'] != self.FILTER_ID]
        self.overall_top_purchases = self.overall_top_purchases.item_id.tolist()

        self.user_item_matrix = self.prepare_matrix(data)

        (self.id_to_item_id, self.id_to_user_id,
         self.item_id_to_id, self.user_id_to_id) = self.prepare_dicts(self.user_item_matrix)

        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)

        if weighted:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

        self._model = None

        # self.model = self.fit(self.user_item_matrix)
        # self.item_factors = self.model.item_factors
        # self.user_factors = self.model.user_factors

    # TODO implement rank_items() for ALS

    @property
    def model(self):
        if self._model is None:
            self.fit()
            return self._model
        else:
            return self._model

    @staticmethod
    def prepare_matrix(data):
        """Transform the data into an user-item matrix."""
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
        """Prepares various id to id dictionaries."""

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
        """Fits a model that recommends items previously purchased by the user."""

        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr(), show_progress=False)

        return own_recommender

    def get_own_recommendations(self, user_id, rec_number=5):
        self.update_dict(user_id=user_id)
        return self.get_recommendations(user_id, model=self.own_recommender, rec_number=rec_number)

    def update_dict(self, user_id):
        if user_id not in self.user_id_to_id.keys():
            max_id = max(list(self.user_id_to_id.values())) + 1
            self.user_id_to_id.update({user_id: max_id})
            self.id_to_user_id.update({max_id: user_id})

    def get_similar_item(self, item_id):
        rec = self.model.similar_items(self.item_id_to_id[item_id], N=2)
        return self.id_to_item_id[rec[1][0]]

    def extend_rec_with_popular(self, recommendations, rec_number=5):
        if len(recommendations) < rec_number:
            recommendations.extend(self.overall_top_purchases[:rec_number])
            recommendations = unique_list(recommendations)
        return recommendations[:rec_number]

    def get_recommendations(self, user_id, model, rec_number):
        self.update_dict(user_id)
        recommendations = model.recommend(userid=self.user_id_to_id[user_id],
                                          user_items=csr_matrix(self.user_item_matrix).tocsr(),
                                          N=rec_number,
                                          filter_already_liked_items=False,
                                          filter_items=None,
                                          recalculate_user=True)
        recommendations = [self.id_to_item_id[rec[0]] for rec in recommendations]
        recommendations = self.extend_rec_with_popular(recommendations, rec_number=rec_number)
        assert len(recommendations) == rec_number, f'Количество рекомендаций не равно {rec_number}'
        return recommendations

    def get_als_recommendations(self, user_id, rec_number=5):
        """Get recommendations predicted by ALS model."""
        return self.get_recommendations(user_id, model=self.model, rec_number=rec_number)

    def fit(self, n_factors=30, regularization=0.001, iterations=15, num_threads=8):
        """Обучает ALS"""

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)
        model.fit(csr_matrix(self.user_item_matrix).T.tocsr(), show_progress=False)

        self._model = model

    def get_similar_items_recommendation(self, user_id, rec_number=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        top_users_purchases = self.top_purchases[self.top_purchases['user_id'] == user_id].head(rec_number)

        recommendations = top_users_purchases['item_id'].apply(lambda x: self.get_similar_item(x)).tolist()
        recommendations = self.extend_rec_with_popular(recommendations, rec_number=rec_number)

        assert len(recommendations) == rec_number, f'Количество рекомендаций != {rec_number}'
        return recommendations

    def get_similar_users_recommendation(self, user_id, n_users=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        users = self.model.similar_users(self.user_id_to_id[user_id], N=n_users + 1)
        users = [user[0] for user in users][1:]

        recommendations = []

        for user in users:
            recommendations.extend(self.get_own_recommendations(user, min(n_users * 3, 9)))

        recommendations = unique_list(recommendations)

        recommendations = self.extend_rec_with_popular(recommendations, n_users)

        assert len(recommendations) == n_users, f'Количество рекомендаций != {n_users}'
        return recommendations


class SecondLevelRecommender:
    """Second level of the recommendation system. It is expected that the input will contain data only for
    a limited amount of candidate items selected by first level model.
    X: np.Array or pd.DataFrame
          An array or a dataframe. Each row should contain user_id, item_id and various features to train
           the model on.
    y: np.Array
       An array containing boolean values indicating whether an item was actually purchased by the user.
    """

    def __init__(self, X, y, categorical_features):
        self.X = X
        self.y = y
        self.categorical_features = categorical_features
        self.model = LGBMClassifier(objective='binary', max_depth=7, categorical_column=self.categorical_features)

    def fit(self):
        self.model.fit(self.X, self.y)

    def predict(self, X_train):
        return self.model.predict_proba(X_train)[:, 1]

# TODO Add feature extractor
