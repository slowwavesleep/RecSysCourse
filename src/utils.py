import pandas as pd
import numpy as np
from src.constants import post_filter_column, dummy_id_filter, min_sales

CATEGORY_NAME = post_filter_column
DUMMY_ID = dummy_id_filter
MIN_SALES = min_sales
LAST_YEAR_WEEKS = 53


def train_test_split(data, valid_1_weeks=6, valid_2_weeks=3):
    """Splits data into four parts: level 1 train, level 1 valid, level 2 train, level 2 valid."""
    data_train_1 = data[data['week_no'] < data['week_no'].max() - (valid_1_weeks + valid_2_weeks)]

    data_valid_1 = data[
        (data['week_no'] >= data['week_no'].max() - (valid_1_weeks + valid_2_weeks)) &
        (data['week_no'] < data['week_no'].max() - valid_2_weeks)]

    data_train_2 = data_valid_1.copy()
    data_valid_2 = data[data['week_no'] >= data['week_no'].max() - valid_2_weeks]

    return data_train_1, data_valid_1, data_train_2, data_valid_2


def unique_list(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def pre_filter_items(data,  take_n_popular=5000, item_features=None):

    # filter out rare departments
    if item_features is not None:
        department_size = pd.DataFrame(item_features.
                                       groupby('DEPARTMENT')['PRODUCT_ID'].nunique().
                                       sort_values(ascending=False)).reset_index()

        department_size.columns = ['DEPARTMENT', 'n_items']
        rare_departments = department_size[department_size['n_items'] < 150].DEPARTMENT.tolist()
        items_in_rare_departments = item_features[
            item_features['DEPARTMENT'].isin(rare_departments)].PRODUCT_ID.unique().tolist()

        data = data[~data['item_id'].isin(items_in_rare_departments)]

    # TODO consider taking discount into account
    # filter out items that are too cheap
    data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
    data = data.loc[data['price'] > 2]

    # filter out items that are too expensive
    data = data[data['price'] < 50]

    # filter out most popular items
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index() / data['user_id'].nunique()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)

    top_popular = popularity[popularity['share_unique_users'] > 0.2].item_id.tolist()
    data = data[~data['item_id'].isin(top_popular)]

    # filter out least popular items
    top_not_popular = popularity[popularity['share_unique_users'] < 0.02].item_id.tolist()
    data = data[~data['item_id'].isin(top_not_popular)]

    # filter out unpopular items by absolute value
    total_sales = data.groupby('item_id')['item_id'].count()
    data = data.loc[~data.item_id.isin(total_sales.loc[total_sales < MIN_SALES].index)]

    # filter out items that had no sales over the past year
    min_week = data.week_no.max() - LAST_YEAR_WEEKS
    last_sale = data.groupby('item_id')['week_no'].max()
    data = data.loc[~data.item_id.isin(last_sale.loc[last_sale < min_week].index)]

    # top n items
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

    top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()

    data.loc[~data['item_id'].isin(top), 'item_id'] = DUMMY_ID
    data = data.loc[data.item_id != DUMMY_ID]

    return data


def post_filter_items(recommendations, item_features, rec_number):
    """Post filter recommended items.
    Input
    -----
    recommendations: list
        Ranked (in descending order) list of recommended items (as `item_id`).
    item_features: pd.DataFrame
        Dataframe containing item features.
    """

    unique_recommendations = unique_list(recommendations)

    # different categories
    categories_used = []
    final_recommendations = []

    for item in unique_recommendations:
        category = item_features.loc[item_features['item_id'] == item, CATEGORY_NAME].values[0]

        if category not in categories_used:
            final_recommendations.append(item)

        unique_recommendations.remove(item)
        categories_used.append(category)

    n_rec = len(final_recommendations)
    if n_rec < rec_number:
        final_recommendations.extend(unique_recommendations[:rec_number - n_rec])
    else:
        final_recommendations = final_recommendations[:rec_number]

    assert len(final_recommendations) == rec_number, f'Количество рекомендаций != {rec_number}'

    return final_recommendations
