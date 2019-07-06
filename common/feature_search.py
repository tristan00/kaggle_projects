import random
import pandas as pd
from sklearn.model_selection import cross_val_score



class FeatureFinder:
    def __init__(self, df, target_column_name):
        self.df = df
        self.target_column_name = target_column_name

    def find_row_features(self, max_iterations = 1000, max_features = 100, sample_size_per_step = 10):
        x = [i for i in self.df.columns if i != self.target_column_name]

        for iteration in range(max_iterations):
            pass

    def remove_worst_row_feature(self, current_features, max_features_to_consider):
        pass



    def add_best_found_row_feature(self, current_features, max_features_to_consider):
        pass


