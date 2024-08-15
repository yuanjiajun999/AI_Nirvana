# core/active_learning.py  

import numpy as np  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score  
from sklearn.base import BaseEstimator, clone  
from sklearn.cluster import KMeans  
from typing import Tuple, List, Optional, Callable  
import logging  
from joblib import Parallel, delayed  
import matplotlib.pyplot as plt  

logger = logging.getLogger(__name__)  

class ActiveLearner:  
    def __init__(self, X_pool, y_pool, X_test, y_test, random_state=None):  
        self.X_pool = X_pool  
        self.y_pool = y_pool  
        self.X_test = X_test  
        self.y_test = y_test  
        self.labeled_indices = np.array([], dtype=int)  
        self.model = RandomForestClassifier(random_state=random_state)  
        self.is_fitted = False  
        self.random_state = random_state  
        np.random.seed(random_state)  

    def uncertainty_sampling(self, n_samples):  
        if not self.is_fitted:  
            return np.random.choice(len(self.X_pool), n_samples, replace=False)  
        probas = self.model.predict_proba(self.X_pool)  
        uncertainties = 1 - np.max(probas, axis=1)  
        return np.argsort(uncertainties)[-n_samples:]  

    def diversity_sampling(self, n_samples):  
        kmeans = KMeans(n_clusters=n_samples, random_state=self.random_state)  
        kmeans.fit(self.X_pool)  
        distances = kmeans.transform(self.X_pool)  
        selected_indices = [np.argmin(distances[:, i]) for i in range(n_samples)]  
        return np.array(selected_indices)  

    def expected_model_change_sampling(self, n_samples):  
        if not self.is_fitted:  
            logger.warning("Model not fitted. Performing random sampling instead of expected model change sampling.")  
            return np.random.choice(len(self.X_pool), n_samples, replace=False)  
        gradients = self._compute_gradients()  
        selected_indices = np.argsort(np.linalg.norm(gradients, axis=1))[-n_samples:]  
        return selected_indices  

    def _compute_gradients(self):  
        # This is a placeholder. The actual implementation would depend on the model type.  
        return np.random.rand(len(self.X_pool), len(np.unique(self.y_pool)))  

    def density_weighted_sampling(self, n_samples):  
        if not self.is_fitted:  
            logger.warning("Model not fitted. Performing random sampling instead of density-weighted uncertainty sampling.")  
            return np.random.choice(len(self.X_pool), n_samples, replace=False)  
        probas = self.model.predict_proba(self.X_pool)  
        uncertainties = 1 - np.max(probas, axis=1)  
        densities = self._compute_densities()  
        scores = uncertainties * densities  
        selected_indices = np.argsort(scores)[-n_samples:]  
        return selected_indices  

    def _compute_densities(self):  
        kmeans = KMeans(n_clusters=min(100, len(self.X_pool)), random_state=self.random_state)  
        kmeans.fit(self.X_pool)  
        distances = kmeans.transform(self.X_pool)  
        return 1 / (1 + np.min(distances, axis=1))  

    def _sample(self, n_samples, strategy='uncertainty'):  
        if strategy == 'uncertainty':  
            return self.uncertainty_sampling(n_samples)  
        elif strategy == 'diversity':  
            return self.diversity_sampling(n_samples)  
        elif strategy == 'expected_model_change':  
            return self.expected_model_change_sampling(n_samples)  
        elif strategy == 'density_weighted':  
            return self.density_weighted_sampling(n_samples)  
        else:  
            raise ValueError(f"Unknown sampling strategy: {strategy}")  

    def train(self, X, y):  
        self.model.fit(X, y)  
        self.is_fitted = True  

    def evaluate(self, metric='accuracy'):  
        if not self.is_fitted:  
            raise ValueError("Model is not fitted yet. Call train() first.")  
        y_pred = self.model.predict(self.X_test)  
        if metric == 'accuracy':  
            return accuracy_score(self.y_test, y_pred)  
        elif metric == 'f1':  
            return f1_score(self.y_test, y_pred, average='weighted')  
        elif metric == 'precision':  
            return precision_score(self.y_test, y_pred, average='weighted')  
        elif metric == 'recall':  
            return recall_score(self.y_test, y_pred, average='weighted')  
        else:  
            raise ValueError(f"Unsupported metric: {metric}")  

    def active_learning_loop(self, initial_samples: int, n_iterations: int,  
                             samples_per_iteration: int, strategy: str = 'uncertainty',  
                             batch_mode: bool = False) -> Tuple[float, List[float]]:  
        # 初始化  
        initial_indices = np.random.choice(len(self.X_pool), initial_samples, replace=False)  
        X_train = self.X_pool[initial_indices]  
        y_train = self.y_pool[initial_indices]  
        self.labeled_indices = initial_indices  
        mask = np.ones(len(self.X_pool), dtype=bool)  
        mask[initial_indices] = False  
        self.X_pool = self.X_pool[mask]  
        self.y_pool = self.y_pool[mask]  

        accuracy_history = []  

        for iteration in range(n_iterations):  
            self.train(X_train, y_train)  
            current_accuracy = self.evaluate()  
            accuracy_history.append(current_accuracy)  
        
            logger.info(f"Iteration {iteration + 1}/{n_iterations}, Accuracy: {current_accuracy:.4f}")  

            selected_indices = self._sample(samples_per_iteration, strategy)  

            X_new = self.X_pool[selected_indices]  
            y_new = self.y_pool[selected_indices]  
            
            if X_new.ndim == 1:  
                X_new = X_new.reshape(1, -1)  
            
            X_train = np.vstack((X_train, X_new))  
            y_train = np.concatenate((y_train, y_new))  
            self.labeled_indices = np.concatenate((self.labeled_indices, selected_indices))  
            self.X_pool = np.delete(self.X_pool, selected_indices, axis=0)  
            self.y_pool = np.delete(self.y_pool, selected_indices)  

        final_accuracy = self.evaluate()  
        return final_accuracy, accuracy_history  

    def get_model(self) -> BaseEstimator:  
        return self.model  

    def set_model(self, new_model: BaseEstimator) -> None:  
        self.model = new_model  
        self.is_fitted = False  

    def get_pool_size(self) -> int:  
        return len(self.X_pool)  

    def get_labeled_size(self) -> int:  
        return len(self.labeled_indices)  

    def create_committee(self, n_models=3):
        if self.get_labeled_size() == 0:
            raise ValueError("No labeled data available to train the committee. Please label some data first.")
        self.committee = [clone(self.model) for _ in range(n_models)]
        X_train, y_train = self.get_labeled_data()
        for model in self.committee:
            model.fit(X_train, y_train) 

    def committee_uncertainty_sampling(self, n_samples):  
        if not hasattr(self, 'committee'):  
            raise ValueError("Committee not created. Call create_committee() first.")  
        # Implement committee-based uncertainty sampling here  
        return np.random.choice(len(self.X_pool), n_samples, replace=False)  
    
    def plot_learning_curve(self, accuracy_history: List[float]) -> None:  
        plt.figure(figsize=(10, 6))  
        plt.plot(range(1, len(accuracy_history) + 1), accuracy_history)  
        plt.title('Active Learning Curve')  
        plt.xlabel('Iteration')  
        plt.ylabel('Accuracy')  
        plt.show()  

    def handle_multilabel(self, y: np.ndarray) -> np.ndarray:  
        # This is a placeholder for multilabel handling  
        return y  
    
    def predict(self, X):  
        if not self.is_fitted:  
            raise ValueError("Model is not fitted yet. Call train() first.")  
        return self.model.predict(X)  
    
    def update_pool(self, indices_to_remove):  
        self.X_pool = np.delete(self.X_pool, indices_to_remove, axis=0)  
        self.y_pool = np.delete(self.y_pool, indices_to_remove)  
        self.labeled_indices = np.setdiff1d(self.labeled_indices, indices_to_remove)  

    def get_labeled_data(self):  
        return self.X_pool[self.labeled_indices], self.y_pool[self.labeled_indices]  

    def label_samples(self, indices):  
        self.labeled_indices = np.union1d(self.labeled_indices, indices).astype(int)

    def active_learning_step(self, n_samples, strategy='uncertainty'):
        selected_indices = self._sample(n_samples, strategy)
        X_new = self.X_pool[selected_indices]
        y_new = self.y_pool[selected_indices]
    
        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)
    
        X_train, y_train = self.get_labeled_data()
        X_train = np.vstack((X_train, X_new))
        y_train = np.concatenate((y_train, y_new))
    
        self.train(X_train, y_train)
        self.update_pool(selected_indices)
    
        return self.evaluate()  # 确保返回准确率  