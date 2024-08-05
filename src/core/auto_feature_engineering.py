import featuretools as ft
import pandas as pd
import numpy as np
import warnings
from typing import List, Dict, Callable, Union, Optional
from featuretools import EntitySet, dfs
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif


class AutoFeatureEngineer:
    """
    A class for automated feature engineering using Featuretools.

    This class provides methods for creating entity sets, generating features,
    and performing various feature selection and preprocessing tasks.

    Attributes:
        data (pd.DataFrame): The input data.
        target_column (str): The name of the target column.
        feature_matrix (Optional[pd.DataFrame]): The generated feature matrix.
        feature_defs (Optional[List]): The feature definitions.
        custom_features (Dict[str, Callable]): Custom feature functions.
        entityset (Optional[EntitySet]): The Featuretools EntitySet.
        index_column (Optional[str]): The name of the index column.
    """

    def __init__(self, data, target_column):
        """
        Initialize the AutoFeatureEngineer.

        Args:
            data (pd.DataFrame): The input data.
            target_column (str): The name of the target column.
        """
        self.data = data
        self.target_column = target_column
        self.feature_matrix = None
        self.feature_defs = None
        self.custom_features = {}
        self.entityset = None

    def set_entityset(self, es: ft.EntitySet):
        self.entityset = es

    def create_entity_set(self, index_column, time_index=None):
        """
        Create an EntitySet from the input data.

        Args:
            index_column (str): The name of the column to use as the index.
            time_index (Optional[str]): The name of the column to use as the time index.

        Returns:
            EntitySet: The created EntitySet.

        Raises:
            ValueError: If the index column is not found in the data.
        """
         # 确保 DataFrame 索引设置正确
        if self.data.index.name != index_column:
            self.data.set_index(index_column, inplace=True)
        
        self.entityset = ft.EntitySet(id="data")
        self.entityset = self.entityset.add_dataframe(
            dataframe_name="data",
            dataframe=self.data,
            index=index_column,
            time_index=time_index
        )
        return self.entityset
    
    def generate_features(self, max_depth: int = 2, primitives: Optional[List[str]] = None, show_warnings: bool = False) -> tuple[pd.DataFrame, List]:
        """
        Generate features using Featuretools' deep feature synthesis.

        Args:
            max_depth (int): The maximum depth of feature generations.
            primitives (Optional[List[str]]): A list of primitive functions to use.
            show_warnings (bool): Whether to show warnings during feature generation.

        Returns:
            tuple[pd.DataFrame, List]: The feature matrix and feature definitions.

        Raises:
            ValueError: If the entity set has not been created.
        """
        if self.entityset is None:
            raise ValueError("Entity set not created. Call create_entity_set() first.")

        if primitives is None:
            primitives = ["count", "sum", "mean", "max", "min", "std"]

        with warnings.catch_warnings():
            if not show_warnings:
                warnings.simplefilter("ignore", category=UserWarning)
            feature_matrix, feature_defs = ft.dfs(
                entityset=self.entityset,
                target_dataframe_name="data",
                agg_primitives=primitives,
                trans_primitives=[],
                max_depth=max_depth,
                features_only=False,
                verbose=True
            )

        self.feature_matrix = feature_matrix
        self.feature_defs = feature_defs

        return self.feature_matrix, self.feature_defs
    
    def create_custom_feature(self, feature_name: str, function: Callable):
        """
        Create a custom feature.

        Args:
            feature_name (str): The name of the custom feature.
            function (Callable): The function to generate the feature.
        """
        self.custom_features[feature_name] = function
        if self.feature_matrix is not None:
            self.feature_matrix[feature_name] = self.feature_matrix.apply(function, axis=1)

    def get_important_features(self, n: int = 10, method: str = 'correlation') -> List[str]:
        """
        Get the most important features based on the specified method.

        Args:
            n (int): The number of important features to return.
            method (str): The method to use for feature importance ('correlation', 'mutual_info', 'mutual_info_regression', or 'mutual_info_classif').

        Returns:
            List[str]: A list of the most important feature names.

        Raises:
            ValueError: If the feature matrix has not been generated or if an invalid method is specified.
        """
        if self.feature_matrix is None:
            raise ValueError("Feature matrix not generated. Call generate_features() first.")

        numeric_features = self.feature_matrix.select_dtypes(include=[np.number])
        target = self.feature_matrix[self.target_column]

        if method == 'correlation':
            corr = numeric_features.corrwith(target).abs().sort_values(ascending=False)
            return corr.index[:n].tolist()
        elif method in ['mutual_info', 'mutual_info_regression', 'mutual_info_classif']:
            X = numeric_features
            y = target
            if y.dtype == 'object' or isinstance(y.dtype, pd.CategoricalDtype):
                mi = mutual_info_classif(X, y)
            else:
                mi = mutual_info_regression(X, y)
            mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
            return mi_series.index[:n].tolist()
        else:
            raise ValueError(f"Invalid method: {method}. Choose 'correlation', 'mutual_info', 'mutual_info_regression', or 'mutual_info_classif'.")

    def get_feature_types(self) -> Dict[str, str]:
        """
        Get the types of all features in the feature matrix.

        Returns:
            Dict[str, str]: A dictionary mapping feature names to their types.

        Raises:
            ValueError: If the feature matrix has not been generated.
        """
        if self.feature_matrix is None:
            raise ValueError("Feature matrix not generated. Call generate_features() first.")

        feature_types = {}
        for column in self.feature_matrix.columns:
            if pd.api.types.is_numeric_dtype(self.feature_matrix[column]):
                feature_types[column] = 'numeric'
            elif isinstance(self.feature_matrix[column].dtype, pd.CategoricalDtype):
                feature_types[column] = 'categorical'
            elif pd.api.types.is_datetime64_any_dtype(self.feature_matrix[column]):
                feature_types[column] = 'datetime'
            else:
                feature_types[column] = 'object'
        return feature_types

    def get_feature_descriptions(self) -> List[str]:
        """
        Get descriptions of all features in the feature matrix.

        Returns:
            List[str]: A list of feature descriptions.

        Raises:
            ValueError: If feature definitions have not been generated.
        """
        if self.feature_defs is None:
            raise ValueError("Feature definitions not generated. Call generate_features() first.")

        return [f.generate_name() for f in self.feature_defs]

    def get_feature_matrix(self) -> pd.DataFrame:
        """
        Get the feature matrix.

        Returns:
            pd.DataFrame: The feature matrix.

        Raises:
            ValueError: If the feature matrix has not been generated.
        """
        if self.feature_matrix is None:
            raise ValueError("Feature matrix not generated. Call generate_features() first.")

        return self.feature_matrix

    def remove_low_information_features(self, threshold: float = 0.95) -> List[str]:
        """
        Remove features with low information content.

        Args:
            threshold (float): The threshold for determining low information content.

        Returns:
            List[str]: A list of removed feature names.

        Raises:
            ValueError: If the feature matrix has not been generated.
        """
        if self.feature_matrix is None:
            raise ValueError("Feature matrix not generated. Call generate_features() first.")

        nunique = self.feature_matrix.nunique() / len(self.feature_matrix)
        columns_to_drop = nunique[nunique <= threshold].index  # 修改这里，使用 <= 而不是 >
        self.feature_matrix = self.feature_matrix.drop(columns=columns_to_drop)
        return list(columns_to_drop)

    def remove_highly_correlated_features(self, threshold: float = 0.9) -> List[str]:
        """
        Remove highly correlated features.

        Args:
            threshold (float): The correlation threshold for removing features.

        Returns:
            List[str]: A list of removed feature names.

        Raises:
            ValueError: If the feature matrix has not been generated.
        """
        if self.feature_matrix is None:
            raise ValueError("Feature matrix not generated. Call generate_features() first.")

        numeric_features = self.feature_matrix.select_dtypes(include=[np.number])

        corr_matrix = numeric_features.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        self.feature_matrix = self.feature_matrix.drop(columns=to_drop)
        return to_drop

    def normalize_features(self, method: str = 'standard'):
        """
        Normalize numeric features in the feature matrix.

        Args:
            method (str): The normalization method to use ('standard' or 'minmax').

        Raises:
            ValueError: If the feature matrix has not been generated or if an invalid method is specified.
        """
        if self.feature_matrix is None:
            raise ValueError("Feature matrix not generated. Call generate_features() first.")

        numeric_features = self.feature_matrix.select_dtypes(include=[np.number]).columns
        if method == 'standard':
            self.feature_matrix[numeric_features] = (self.feature_matrix[numeric_features] - self.feature_matrix[numeric_features].mean()) / self.feature_matrix[numeric_features].std()
        elif method == 'minmax':
            self.feature_matrix[numeric_features] = (self.feature_matrix[numeric_features] - self.feature_matrix[numeric_features].min()) / (self.feature_matrix[numeric_features].max() - self.feature_matrix[numeric_features].min())
        else:
            raise ValueError("Invalid normalization method. Choose 'standard' or 'minmax'.")

    def encode_categorical_features(self, method: str = 'onehot'):
        """
        Encode categorical features in the feature matrix.

        Args:
            method (str): The encoding method to use ('onehot' or 'label').

        Raises:
            ValueError: If the feature matrix has not been generated or if an invalid method is specified.
        """
        if self.feature_matrix is None:
            raise ValueError("Feature matrix not generated. Call generate_features() first.")

        categorical_features = self.feature_matrix.select_dtypes(include=['object']).columns
        if method == 'onehot':
            self.feature_matrix = pd.get_dummies(self.feature_matrix, columns=categorical_features)
        elif method == 'label':
            for feature in categorical_features:
                self.feature_matrix[feature] = self.feature_matrix[feature].astype('category').cat.codes
        else:
            raise ValueError("Invalid encoding method. Choose 'onehot' or 'label'.")