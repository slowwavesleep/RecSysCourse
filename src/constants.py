import yaml

with open("../src/config.yml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

post_filter_column = config['post_filter_column']
dummy_id_filter = config['dummy_id_filter']
min_sales = config['min_sales']
