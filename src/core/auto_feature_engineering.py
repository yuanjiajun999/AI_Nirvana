# core/auto_feature_engineering.py

import pandas as pd
from featuretools import EntitySet, dfs


class AutoFeatureEngineer:
    def __init__(self, data):
        self.data = data
        self.entity_set = None
        self.feature_matrix = None

    def create_entity_set(self):
        es = EntitySet("dataset")
        es = es.entity_from_dataframe(entity_id="data", dataframe=self.data, 
                                      index="id", time_index="timestamp")
        self.entity_set = es

    def generate_features(self, target_entity="data", max_depth=2):
        feature_matrix, feature_defs = dfs(entityset=self.entity_set, 
                                           target_entity=target_entity,
                                           max_depth=max_depth)
        self.feature_matrix = feature_matrix
        return feature_matrix, feature_defs

    def get_important_features(self, n=10):
        return self.feature_matrix.columns[:n].tolist()