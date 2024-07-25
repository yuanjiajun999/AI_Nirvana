# core/active_learning.py

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class ActiveLearner:
    def __init__(self, X_pool, y_pool, X_test, y_test):
        self.X_pool = X_pool
        self.y_pool = y_pool
        self.X_test = X_test
        self.y_test = y_test
        self.model = RandomForestClassifier(n_estimators=100)

    def uncertainty_sampling(self, n_samples):
        probas = self.model.predict_proba(self.X_pool)
        uncertainties = 1 - np.max(probas, axis=1)
        selected_indices = np.argsort(uncertainties)[-n_samples:]
        return selected_indices

    def train(self, X, y):
        self.model.fit(X, y)

    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred)


def active_learning_loop(self, initial_samples, n_iterations, samples_per_iteration):
    X_train = self.X_pool[:initial_samples]
    y_train = self.y_pool[:initial_samples]
    self.X_pool = self.X_pool[initial_samples:]
    self.y_pool = self.y_pool[initial_samples:]

    for _ in range(n_iterations):
        self.train(X_train, y_train)
        selected_indices = self.uncertainty_sampling(samples_per_iteration)
        X_new = self.X_pool[selected_indices]
        y_new = self.y_pool[selected_indices]
        X_train = np.vstack((X_train, X_new))
        y_train = np.concatenate((y_train, y_new))
        self.X_pool = np.delete(self.X_pool, selected_indices, axis=0)
        self.y_pool = np.delete(self.y_pool, selected_indices)

    final_accuracy = self.evaluate()
    return final_accuracy
