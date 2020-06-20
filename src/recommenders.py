import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix

from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender
from implicit.nearest_neighbours import bm25_weight

from lightgbm import LGBMClassifier

from src.utils import unique_list
from src.metrics import recall_at_k, precision_at_k
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

        self._user_item_matrix = self.prepare_matrix(data)

        (self.id_to_item_id, self.id_to_user_id,
         self.item_id_to_id, self.user_id_to_id) = self.prepare_dicts(self.user_item_matrix)

        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)

        if weighted:
            self._user_item_matrix = bm25_weight(self.user_item_matrix.T).T

        self._model = None
        self._user_factors = None
        self._item_factors = None

    # TODO implement rank_items() for ALS

    @property
    def user_item_matrix(self):
        return self._user_item_matrix

    @property
    def model(self):
        if self._model is None:
            self.fit()
            return self._model
        else:
            return self._model

    @property
    def item_factors(self):
        if self._model is None:
            self.fit()
        if self._item_factors is None:
            ids = pd.DataFrame.from_dict(self.id_to_item_id, orient='index', columns=['item_id'])
            self._item_factors = pd.DataFrame(self.model.item_factors)
            self._item_factors = ids.merge(self._item_factors, left_index=True, right_index=True)
            self._item_factors.columns = ['item_id'] + [f'item_factor_{i}' for i in range(1, self.model.factors+1)]
        return self._item_factors

    @property
    def user_factors(self):
        if self._model is None:
            self.fit()
        if self._user_factors is None:
            ids = pd.DataFrame.from_dict(self.id_to_user_id, orient='index', columns=['user_id'])
            self._user_factors = pd.DataFrame(self.model.user_factors)
            self._user_factors = ids.merge(self.user_factors, left_index=True, right_index=True)
            self._user_factors.columns = ['user_id'] + [f'user_factor_{i}' for i in range(1, self.model.factors + 1)]

        return self._user_factors

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
        assert len(recommendations) == rec_number, f'Number of recommendations is not equal to {rec_number}'
        return recommendations

    def get_als_recommendations(self, user_id, rec_number=5):
        """Get recommendations predicted by ALS model."""
        return self.get_recommendations(user_id, model=self.model, rec_number=rec_number)

    def fit(self, n_factors=30, regularization=0.001, iterations=15, num_threads=8):
        """Fit ALS model."""

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)
        model.fit(csr_matrix(self.user_item_matrix).T.tocsr(), show_progress=False)

        self._model = model

    def get_similar_items_recommendation(self, user_id, rec_number=5):
        """Recommend a similar item for each of user's top N most bought items."""

        top_users_purchases = self.top_purchases[self.top_purchases['user_id'] == user_id].head(rec_number)

        recommendations = top_users_purchases['item_id'].apply(lambda x: self.get_similar_item(x)).tolist()
        recommendations = self.extend_rec_with_popular(recommendations, rec_number=rec_number)

        assert len(recommendations) == rec_number, f'Number of recommendations is not equal to {rec_number}'
        return recommendations

    def get_similar_users_recommendation(self, user_id, n_users=5):
        """Recommend an item for top N of a user's most similar users."""

        users = self.model.similar_users(self.user_id_to_id[user_id], N=n_users + 1)
        users = [user[0] for user in users][1:]

        recommendations = []

        for user in users:
            recommendations.extend(self.get_own_recommendations(user, min(n_users * 3, 9)))

        recommendations = unique_list(recommendations)

        recommendations = self.extend_rec_with_popular(recommendations, n_users)

        assert len(recommendations) == n_users, f'Number of recommendations is not equal to {n_users}'
        return recommendations

    def df_als_predictions(self, data, rec_number=200):
        data['als_candidates'] = data['user_id'].apply(lambda x: self.get_als_recommendations(x, rec_number))
        return data


class SecondLevelRecommender:
    """Second level of the recommendation system. It is expected that the input will contain data only for
    a limited amount of candidate items selected by first level model.
    X: np.Array or pd.DataFrame
          An array or a dataframe. Each row should contain user_id, item_id and various features to train
           the model on.
    y: np.Array
       An array containing boolean values indicating whether an item was actually purchased by the user.
    """

    def __init__(self, data, categories):
        self.categories = categories
        self.data = data

        self.x = self.data.drop('target', axis=1)
        self.x = self.x[self.categories].astype('category')
        self.y = self.data['target']

        self.model = LGBMClassifier(objective='binary',
                                    max_depth=10,
                                    learning_rate=0.001,
                                    categorical_column=self.categories,
                                    num_iterations=100,
                                    num_leaves=100,
                                    dart=True,
                                    scale_pos_weight=0.03
                                    )

    def fit(self):
        self.model.fit(self.x, self.y)

    def predict(self):
        """Predict probability of purchase for each item."""
        return self.model.predict_proba(self.x)[:, 1]

    def df_predict(self):
        self.data['preds'] = self.predict()
        self.data.sort_values(['user_id', 'preds'], ascending=[True, False], inplace=True)
        lgb_candidates = self.data.groupby('user_id').head(5).groupby('user_id')['item_id'].unique().reset_index()
        lgb_candidates.columns = ['user_id', 'candidates']
        return lgb_candidates

    @staticmethod
    def eval_prediction(valid_data, candidates):
        valid_data = valid_data.groupby('user_id')['item_id'].unique().reset_index()\
            .rename(columns={'item_id': 'actual'})
        valid_data = valid_data.merge(candidates, on='user_id', how='inner')
        precision = valid_data[valid_data.candidates.notna()]. \
            apply(lambda row: precision_at_k(row['candidates'], row['actual'], k=5), axis=1).mean()
        return precision


class DataTransformer:
    """This class is used bring train and valid datasets to a unified format."""
    def __init__(self, data, user_features, item_features):

        self.data = data

        self.user_features = user_features
        self.user_features.columns = [col.lower() for col in self.user_features.columns]
        self.user_features.rename(columns={'household_key': 'user_id'}, inplace=True)

        self.item_features = item_features
        self.item_features.columns = [col.lower() for col in self.item_features.columns]
        self.item_features.rename(columns={'product_id': 'item_id'}, inplace=True)

        self.data = self.data.merge(self.item_features, on='item_id', how='left')

        self.data['month'] = self.data['week_no'].apply(lambda x: np.ceil(x / 30).astype('int32'))

        self.data['weekend'] = self.data.day.apply(lambda x: True if (x == 6) or (x == 7) else False)

        # save categorical column names for ease of access
        self.categorical = ['manufacturer', 'department', 'brand', 'commodity_desc', 'sub_commodity_desc',
                            'curr_size_of_product', 'age_desc', 'marital_status_code', 'income_desc', 'homeowner_desc',
                            'hh_comp_desc', 'household_size_desc', 'kid_category_desc']

    def transform(self):
        """Adds all the additional columns to the dataframe and modifies it as to make it
        usable for second level recommender. Returns transformed data."""

        # TODO merge with features instead

        # merge with custom features
        self.user_features = self.user_features.merge(self.weekend_purchases_ratio, on='user_id', how='left')
        self.user_features.weekend_purchases_ratio = self.user_features.weekend_purchases_ratio.fillna(0)

        self.user_features = self.user_features.merge(self.user_avg_basket_price, on='user_id', how='left')
        self.user_features.user_avg_basket_price = self.user_features.user_avg_basket_price.fillna(
            self.user_features.user_avg_basket_price.mean()
        )

        # self.user_features = self.user_features.merge(self.purchases_in_category,
        #                                               on=['user_id', 'commodity_desc'], how='left')

        self.user_features = self.user_features.merge(self.purchases_per_month, on='user_id', how='left')
        self.user_features.purchases_per_month = self.user_features.purchases_per_month.fillna(
            self.user_features.purchases_per_month.mean()
        )

        # TODO fill after merging

        # fill missing values
        self.user_features.income_desc = self.user_features.income_desc.fillna('Unknown')
        self.user_features.homeowner_desc = self.user_features.homeowner_desc.fillna('Unknown')
        self.user_features.hh_comp_desc = self.user_features.hh_comp_desc.fillna('Unknown')
        self.user_features.household_size_desc = self.user_features.household_size_desc.fillna('Unknown')
        self.user_features.kid_category_desc = self.user_features.kid_category_desc.replace('None/Unknown', 'Unknown').\
            fillna('Unknown')
        self.user_features.age_desc = self.user_features.age_desc.fillna('Unknown')
        self.user_features.marital_status_code = self.user_features.marital_status_code.fillna('U')

    def prepare_train_df(self, data_train_1, data_train_2, recommender):

        users_lvl_2 = pd.DataFrame(data_train_2['user_id'].unique())
        users_lvl_2.columns = ['user_id']

        train_users = data_train_1['user_id'].unique()
        users_lvl_2 = users_lvl_2[users_lvl_2['user_id'].isin(train_users)]

        users_lvl_2['candidates'] = users_lvl_2['user_id'].apply(lambda x: recommender.get_own_recommendations(x, 200))

        s = users_lvl_2.apply(lambda x: pd.Series(x['candidates']), axis=1).stack().reset_index(level=1, drop=True)
        s.name = 'item_id'

        users_lvl_2 = users_lvl_2.drop('candidates', axis=1).join(s)
        users_lvl_2['flag'] = 1

        targets_lvl_2 = data_train_2[['user_id', 'item_id']].copy()
        targets_lvl_2['target'] = 1

        targets_lvl_2 = users_lvl_2.merge(targets_lvl_2, on=['user_id', 'item_id'], how='left')

        targets_lvl_2['target'].fillna(0, inplace=True)
        targets_lvl_2.drop('flag', axis=1, inplace=True)

        targets_lvl_2 = targets_lvl_2.merge(self.item_features, on='item_id', how='left')
        targets_lvl_2 = targets_lvl_2.merge(self.user_features, on='user_id', how='left')
        targets_lvl_2 = targets_lvl_2.merge(self.purchases_in_category, on=['user_id', 'commodity_desc'],
                                            how='left')
        targets_lvl_2.purchases_in_category = targets_lvl_2.purchases_in_category.fillna(0)

        targets_lvl_2.weekend_purchases_ratio = targets_lvl_2.weekend_purchases_ratio.fillna(0)

        targets_lvl_2.user_avg_basket_price = targets_lvl_2.user_avg_basket_price.fillna(
            self.user_features.user_avg_basket_price.mean()
        )

        targets_lvl_2.purchases_per_month = targets_lvl_2.purchases_per_month.fillna(
            self.user_features.purchases_per_month.mean()
        )

        targets_lvl_2.income_desc = targets_lvl_2.income_desc.fillna('Unknown')
        targets_lvl_2.homeowner_desc = targets_lvl_2.homeowner_desc.fillna('Unknown')
        targets_lvl_2.hh_comp_desc = targets_lvl_2.hh_comp_desc.fillna('Unknown')
        targets_lvl_2.household_size_desc = targets_lvl_2.household_size_desc.fillna('Unknown')
        targets_lvl_2.kid_category_desc = targets_lvl_2.kid_category_desc.fillna('Unknown')
        targets_lvl_2.age_desc = targets_lvl_2.age_desc.fillna('Unknown')
        targets_lvl_2.marital_status_code = targets_lvl_2.marital_status_code.fillna('U')

        targets_lvl_2 = targets_lvl_2.merge(recommender.item_factors, on='item_id', how='left')
        targets_lvl_2 = targets_lvl_2.merge(recommender.user_factors, on='user_id', how='left')

        return targets_lvl_2

    def time_format(self):

        def fix_time(time):
            if len(time) == 1:
                return '00:0' + time
            elif len(time) == 2:
                return '00:' + time
            elif len(time) == 3:
                time = time[:1] + ':' + time[1:]
                return '0' + time
            else:
                time = time[:2] + ':' + time[2:]
                return time

        time = self.data.trans_time.astype(str)
        time = time.apply(fix_time)
        time = pd.to_datetime(time, format='%H:%M').dt.time
        return time

    # time consuming method
    def add_purchases_hours_cat(self):

        self.data['trans_time_type'] = self.trans_time_type
        self.categorical.append('trans_time_type')

    @property
    def trans_time_type(self):

        time = self.time_format()

        def work_hours(time):
            four_am = pd.to_datetime('04:00', format='%H:%M').time()
            nine_am = pd.to_datetime('09:00', format='%H:%M').time()
            six_pm = pd.to_datetime('18:00', format='%H:%M').time()

            if four_am <= time < nine_am:
                return 'before_wh'
            elif nine_am <= time < six_pm:
                return 'during_wh'
            else:
                return 'after_wh'

        return time.apply(work_hours)

    @property
    def weekend_purchases_ratio(self):
        df = pd.merge(self.total_purchases, self.weekend_purchases, on='user_id')
        weekend_purchases_ratio = df.apply(lambda row: row['weekend_purchases'] / row['total_purchases'], axis=1)
        weekend_purchases_ratio.name = 'weekend_purchases_ratio'
        return weekend_purchases_ratio

    @property
    def user_avg_basket_price(self):
        user_avg_basket_price = self.data.groupby(['user_id', 'basket_id'])['sales_value'].\
            sum().groupby('user_id').mean()
        user_avg_basket_price.name = 'user_avg_basket_price'
        return user_avg_basket_price

    @property
    def weekend_purchases(self):
        weekend_purchases = self.data.groupby('user_id')['weekend'].sum()
        weekend_purchases.name = 'weekend_purchases'
        return weekend_purchases

    @property
    def total_purchases(self):
        total_purchases = self.data.groupby('user_id')['basket_id'].nunique()
        total_purchases.name = 'total_purchases'
        return total_purchases

    @property
    def purchases_in_category(self):
        purchases_in_category = self.data.groupby(['user_id', 'commodity_desc'])['basket_id'].count()
        purchases_in_category.name = 'purchases_in_category'
        return purchases_in_category

    @property
    def purchases_per_month(self):
        purchases_per_month = self.data.groupby(['user_id', 'month'])['basket_id'].nunique()
        purchases_per_month.name = 'purchases_per_month'
        return purchases_per_month.groupby('user_id').mean()

    @staticmethod
    def valid_items(data_valid, data_train=None, warm_start=True):

        result = data_valid.groupby('user_id')['item_id'].unique().reset_index()

        if warm_start and (data_train is not None):
            train_users = data_train['user_id'].unique()
            result = result.loc[result.user_id.isin(train_users)]

        result.columns = ['user_id', 'actual']

        return result

    @staticmethod
    def eval_recall_at_k(data, eval_column, k=200):
        recall = data[data[eval_column].notna()]. \
            apply(lambda row: recall_at_k(row[eval_column], row['actual'], k), axis=1).mean()
        return recall

    @staticmethod
    def eval_precision_at_k(data, eval_column, k=5):
        precision = data[data[eval_column].notna()]. \
            apply(lambda row: precision_at_k(row[eval_column], row['actual'], k), axis=1).mean()
        return precision


