from src.recommenders import MainRecommender
from src.utils import pre_filter_items
import pandas as pd

data = pd.read_csv('../data/retail_train.csv')

data = pre_filter_items(data)
rec = MainRecommender(data.head(100000), weighted=True)

x = rec.get_similar_items_recommendation(2375, 2)
print(x)

y = rec.get_similar_users_recommendation(2375, 2)
print(y)
