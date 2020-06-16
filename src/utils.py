import pandas as pd
import numpy as np
from src.constants import post_filter_column

CATEGORY_NAME = post_filter_column


def unique_list(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def pre_filter_items(data,  take_n_popular=5000, item_features=None):
    # Уберем самые популярные товары (их и так купят)
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index() / data['user_id'].nunique()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)

    top_popular = popularity[popularity['share_unique_users'] > 0.2].item_id.tolist()
    data = data[~data['item_id'].isin(top_popular)]

    # Уберем самые НЕ популярные товары (их и так НЕ купят)
    top_not_popular = popularity[popularity['share_unique_users'] < 0.02].item_id.tolist()
    data = data[~data['item_id'].isin(top_not_popular)]

    # Уберем товары, которые не продавались за последние 12 месяцев

    # Уберем не интересные для рекоммендаций категории (department)
    if item_features is not None:
        department_size = pd.DataFrame(item_features.
                                       groupby('DEPARTMENT')['PRODUCT_ID'].nunique().
                                       sort_values(ascending=False)).reset_index()

        department_size.columns = ['DEPARTMENT', 'n_items']
        rare_departments = department_size[department_size['n_items'] < 150].DEPARTMENT.tolist()
        items_in_rare_departments = item_features[
            item_features['DEPARTMENT'].isin(rare_departments)].PRODUCT_ID.unique().tolist()

        data = data[~data['item_id'].isin(items_in_rare_departments)]

    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
    data = data[data['price'] > 2]

    # Уберем слишком дорогие товарыs
    data = data[data['price'] < 50]

    # Возбмем топ по популярности
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

    top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()

    # Заведем фиктивный item_id (если юзер покупал товары из топ-5000, то он "купил" такой товар)
    data.loc[~data['item_id'].isin(top), 'item_id'] = 999999

    # ...

    return data


def post_filter_items(recommendations, item_features, rec_number):
    """Пост-фильтрация товаров

    Input
    -----
    recommendations: list
        Ранжированный список item_id для рекомендаций
    item_info: pd.DataFrame
        Датафрейм с информацией о товарах
    """

    unique_recommendations = unique_list(recommendations)

    # Разные категории
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
