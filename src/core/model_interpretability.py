import matplotlib.pyplot as plt  
import numpy as np  
import pandas as pd  
import shap  
from lime import lime_tabular  
import dtreeviz  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.inspection import PartialDependenceDisplay  
from sklearn.inspection import permutation_importance
from sklearn.inspection import partial_dependence
from sklearn.inspection import plot_partial_dependence
from pdpbox import pdp, info_plots  
from sklearn.model_selection import cross_val_score, learning_curve  
from sklearn.metrics import confusion_matrix, roc_curve, auc  
import seaborn as sns  
from interpret import show  
from interpret.blackbox import LimeTabular  
from interpret.perf import ROC  
from sklearn.preprocessing import label_binarize  
from sklearn.utils import resample  
from sklearn.ensemble import RandomForestClassifier  
from eli5 import explain_weights  
from eli5.sklearn import PermutationImportance  
import plotly.graph_objs as go  
from plotly.subplots import make_subplots  
import joblib  
import os  
from sklearn.inspection import permutation_importance as sk_permutation_importance
import lime  
import lime.lime_tabular
from sklearn.base import is_classifier  
from sklearn.utils.multiclass import type_of_target

class ModelInterpreter:  
    """  
    A class for interpreting machine learning models.  

    This class provides various methods for visualizing and understanding  
    the behavior of machine learning models, including feature importance,  
    partial dependence plots, and more.  

    Attributes:  
        model: The trained machine learning model.  
        X: The feature dataset used for training.  
        y: The target variable.  
        model_type: Type of the model ('classifier' or 'regressor').  
    """  

    def __init__(self, model, X, y, feature_names=None, class_names=None, model_type='tree', test_size=0.2, random_state=42):
        self.model = model
    
        # 处理 X  
        if isinstance(X, pd.DataFrame):  
            self.X = X  
        elif isinstance(X, np.ndarray):  
            self.X = pd.DataFrame(X, columns=feature_names or [f'Feature {i}' for i in range(X.shape[1])])  
        else:  
            raise TypeError("X should be a pandas DataFrame or numpy array")  
    
        # 处理 y  
        if isinstance(y, pd.Series):  
            self.y = y  
        elif isinstance(y, np.ndarray):  
            self.y = pd.Series(y)  
        else:  
            raise TypeError("y should be a pandas Series or numpy array")  
    
        # 如果没有提供 feature_names，使用 DataFrame 的列名  
        self.feature_names = self.X.columns.tolist()  
        self.class_names = class_names  
        self.model_type = model_type  

        # Split the data into train and test sets  
        from sklearn.model_selection import train_test_split  
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)  

        self.explainer = None
        self.lime_explainer = None  

        # 设置中文字体，但要注意可能的兼容性问题  
        try:  
            plt.rcParams['font.sans-serif'] = ['SimHei']  
            plt.rcParams['axes.unicode_minus'] = False  
        except Exception as e:  
            print(f"Warning: Could not set Chinese font: {e}")  

        shap.initjs()

        # 如果模型不为 None，创建 SHAP 解释器
        if self.model is not None:
            self.create_explainer()

    def create_explainer(self, method='shap'):  
        """  
        Create an explainer object for the specified method.  

        Args:  
            method (str): The explanation method ('shap' or 'lime').  

        Raises:  
            ValueError: If an unsupported explanation method is specified.  
        """  
        if method == 'shap':  
            if self.model_type == 'tree':  
                self.explainer = shap.TreeExplainer(self.model)  
            elif self.model_type == 'linear':  
                self.explainer = shap.LinearExplainer(self.model, self.X_train)  
        elif method == 'lime':  
            self.lime_explainer = lime_tabular.LimeTabularExplainer(  
                self.X.values,  
                feature_names=self.X.columns,  
                class_names=['Class ' + str(i) for i in range(len(np.unique(self.y)))],  
                mode='classification'  
            )  
        else:  
            raise ValueError("Unsupported explanation method")  
        
    def get_shap_values(self):  
        """  
        Calculate SHAP values for the dataset.  

        Returns:  
            np.array: SHAP values for each instance and feature.  
        """  
        return self.explainer.shap_values(self.X)  

    def plot_summary(self, plot_type="bar", class_index=None, save_path=None):
        shap_values = self.get_shap_values()
        if isinstance(shap_values, list):
            if class_index is not None:
                shap_values = shap_values[class_index]
            else:
                shap_values = np.abs(np.array(shap_values)).mean(0)
        feature_names = np.array(self.feature_names)
        shap.summary_plot(shap_values, self.X, plot_type=plot_type, 
                          feature_names=feature_names)
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_force(self, instance_index, save_path=None):  
        shap_values = self.explainer(self.X_test[instance_index:instance_index+1])  
        plt.figure()  
        shap.plots.waterfall(shap_values[0, :, 0])  # 使用第一个类别的SHAP值  
        if save_path:  
            plt.savefig(save_path)  
            plt.close()

    def explain_instance_lime(self, instance_index, save_path=None):  
        if self.lime_explainer is None:  
            self.create_explainer('lime')  
    
        try:  
            instance = self.X.iloc[instance_index].values  
            exp = self.lime_explainer.explain_instance(  
                instance,   
                self.model.predict_proba,   
                num_features=10  
            )  
            if save_path:  
                # 尝试直接保存为 HTML  
                exp.save_to_file(save_path)  
            
                # 如果上面的方法失败，尝试手动创建 HTML  
                if not os.path.exists(save_path):  
                    html = exp.as_html()  
                    with open(save_path, 'w', encoding='utf-8') as f:  
                        f.write(html)  
        except Exception as e:  
            print(f"Error in LIME explanation: {str(e)}")  
            # 如果 LIME 解释失败，创建一个空的 HTML 文件  
            if save_path:  
                with open(save_path, 'w') as f:  
                    f.write("<html><body><p>LIME explanation failed</p></body></html>")
    
    def feature_importance(self, save_path=None):  
        """  
        Plot feature importances based on the model's feature_importances_ attribute.  

        Args:  
            save_path (str): Path to save the plot image.  
        """  
        if not hasattr(self.model, 'feature_importances_'):  
            raise AttributeError("Model does not have feature_importances_ attribute")  
        
        importances = self.model.feature_importances_  
        feature_importance = pd.DataFrame({'feature': self.X.columns, 'importance': importances})  
        feature_importance = feature_importance.sort_values('importance', ascending=False)  
        plt.figure(figsize=(10, 6))  
        sns.barplot(x='importance', y='feature', data=feature_importance)  
        plt.title('Feature Importance')  
        plt.tight_layout()  
        if save_path:  
            plt.savefig(save_path)  
            plt.close()  
        else:  
            plt.show()  

    def decision_path_visualization(self, instance_index, save_path=None):  
        """  
        Visualize the decision path for a single instance in a decision tree.  

        Args:  
            instance_index (int): Index of the instance to visualize.  
            save_path (str): Path to save the plot image.  
        """  
        if not isinstance(self.model, DecisionTreeClassifier):  
            single_tree = DecisionTreeClassifier()  
            single_tree.fit(self.X, self.y)  
        else:  
            single_tree = self.model  
        feature_names = self.X.columns.tolist()  
        tree_viz = dtreeviz.model(single_tree, self.X, self.y, feature_names=feature_names, class_names=single_tree.classes_)  
        viz = tree_viz.view(x=self.X.iloc[instance_index])  
        if save_path:
            save_path = save_path.replace('.png', '.svg')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            try:
                viz.save(save_path)
                print(f"Decision path saved to {save_path}")
            except Exception as e:
                print(f"Error saving decision path: {e}")
        else:
            viz.view()
   
    def plot_decision_boundary(self, feature1, feature2, resolution=100, save_path=None):  
        """  
        Plot the decision boundary for two selected features.  

        Args:  
            feature1 (str): Name of the first feature.  
            feature2 (str): Name of the second feature.  
            resolution (int): Resolution of the plot.  
            save_path (str): Path to save the plot image.  
        """  
        f1, f2 = self.X[feature1], self.X[feature2]  
        x1 = np.linspace(f1.min(), f1.max(), resolution)  
        x2 = np.linspace(f2.min(), f2.max(), resolution)  
        X1, X2 = np.meshgrid(x1, x2)  
        X_plot = np.c_[X1.ravel(), X2.ravel()]
        X_plot_df = pd.DataFrame(X_plot, columns=[feature1, feature2])
        other_features = [f for f in self.X.columns if f not in [feature1, feature2]]
        for f in other_features:
            X_plot_df[f] = self.X[f].mean()
        y_plot = self.model.predict(X_plot_df).reshape(X1.shape) 
        plt.figure(figsize=(10, 8))  
        plt.contourf(X1, X2, y_plot, alpha=0.8, cmap=plt.cm.RdYlBu)  
        scatter = plt.scatter(f1, f2, c=self.y, cmap=plt.cm.RdYlBu, edgecolor='black')  
        plt.xlabel(feature1)  
        plt.ylabel(feature2)  
        plt.title(f'Decision Boundary - {feature1} vs {feature2}')  
        plt.colorbar(scatter)  
        if save_path:  
            plt.savefig(save_path)  
            plt.close()  
        else:  
            plt.show()  

    def plot_feature_importance(self, save_path=None):  
        if hasattr(self.model, 'feature_importances_'):  
            importances = self.model.feature_importances_  
        elif hasattr(self.model, 'coef_'):  
            importances = np.abs(self.model.coef_).flatten()  
        else:  
            raise ValueError("Model doesn't have feature_importances_ or coef_ attribute")  
    
        feature_names = self.X.columns if hasattr(self.X, 'columns') else [f'Feature {i}' for i in range(len(importances))]  
    
        indices = np.argsort(importances)[::-1]  
    
        plt.figure(figsize=(10, 6))  
        plt.title("Feature Importances")  
        plt.bar(range(len(importances)), importances[indices])  
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)  
        plt.tight_layout()  
    
        if save_path:  
            plt.savefig(save_path)  
            plt.close()  
        else:  
            plt.show()  
    
    def plot_partial_dependence(self, features, save_path=None, target_class=None):  
        if is_classifier(self.model):  
            target_type = type_of_target(self.y)  
            if target_type == "multiclass":  
                if target_class is None:  
                    target_class = 0  # Default to first class if not specified  
                display = PartialDependenceDisplay.from_estimator(  
                    self.model, self.X, features, target=target_class, kind="average"  
                )  
            else:  
                display = PartialDependenceDisplay.from_estimator(  
                    self.model, self.X, features, kind="average"  
                )  
        else:  
            display = PartialDependenceDisplay.from_estimator(  
                self.model, self.X, features, kind="average"  
            )  
    
        if save_path:  
            plt.savefig(save_path)  
            plt.close()  
        return display  

    def plot_pdp_interact(self, features, save_path=None, target_class=None):  
        if is_classifier(self.model):  
            target_type = type_of_target(self.y)  
            if target_type == "multiclass":  
                if target_class is None:  
                    target_class = 0  # Default to first class if not specified  
                display = PartialDependenceDisplay.from_estimator(  
                    self.model, self.X, features, target=target_class, kind="average"  
                )  
            else:  
                display = PartialDependenceDisplay.from_estimator(  
                    self.model, self.X, features, kind="average"  
                )  
        else:  
            display = PartialDependenceDisplay.from_estimator(  
                self.model, self.X, features, kind="average"  
            )  
        
        if save_path:  
            plt.savefig(save_path)  
            plt.close()  
        return display  
    
    def cross_validation(self, cv=5):  
        """  
        Perform cross-validation and print the results.  

        Args:  
            cv (int): Number of folds for cross-validation.  
        """  
        scores = cross_val_score(self.model, self.X, self.y, cv=cv)  
        print(f"Cross-validation scores: {scores}")  
        print(f"Mean score: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")  

    def plot_confusion_matrix(self, save_path=None):  
        """  
        Plot the confusion matrix.  

        Args:  
            save_path (str): Path to save the plot image.  
        """  
        y_pred = self.model.predict(self.X)  
        cm = confusion_matrix(self.y, y_pred)  
        plt.figure(figsize=(10, 8))  
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  
        plt.title('Confusion Matrix')  
        plt.xlabel('Predicted Label')  
        plt.ylabel('True Label')  
        if save_path:  
            plt.savefig(save_path)  
            plt.close()  
        else:  
            plt.show()  

    def plot_roc_curve(self, save_path=None):  
        """  
        Plot the ROC curve.  

        Args:  
            save_path (str): Path to save the plot image.  
        """  
        y_pred_proba = self.model.predict_proba(self.X)  
        n_classes = y_pred_proba.shape[1]  
        y_test_bin = label_binarize(self.y, classes=range(n_classes))  
        
        fpr = dict()  
        tpr = dict()  
        roc_auc = dict()  
        for i in range(n_classes):  
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])  
            roc_auc[i] = auc(fpr[i], tpr[i])  
        
        plt.figure(figsize=(10, 8))  
        for i in range(n_classes):  
            plt.plot(fpr[i], tpr[i], label=f'ROC curve (AUC = {roc_auc[i]:.2f}) for class {i}')  
        
        plt.plot([0, 1], [0, 1], 'k--')  
        plt.xlim([0.0, 1.0])  
        plt.ylim([0.0, 1.05])  
        plt.xlabel('False Positive Rate')  
        plt.ylabel('True Positive Rate')  
        plt.title('Receiver Operating Characteristic (ROC) Curve')  
        plt.legend(loc="lower right")  
        if save_path:  
            plt.savefig(save_path)  
            plt.close()  
        else:  
            plt.show()  

    def plot_learning_curve(self, save_path=None):  
        """  
        Plot the learning curve.  

        Args:  
            save_path (str): Path to save the plot image.  
        """  
        train_sizes, train_scores, test_scores = learning_curve(  
            self.model, self.X, self.y, cv=5, n_jobs=-1,   
            train_sizes=np.linspace(0.1, 1.0, 10))  
        
        train_mean = np.mean(train_scores, axis=1)  
        train_std = np.std(train_scores, axis=1)  
        test_mean = np.mean(test_scores, axis=1)  
        test_std = np.std(test_scores, axis=1)  
        
        plt.figure(figsize=(10, 8))  
        plt.plot(train_sizes, train_mean, label='Training score')  
        plt.plot(train_sizes, test_mean, label='Cross-validation score')  
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)  
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)  
        plt.xlabel('Training examples')  
        plt.ylabel('Score')  
        plt.title('Learning Curve')  
        plt.legend(loc='best')  
        if save_path:  
            plt.savefig(save_path)  
            plt.close()  
        else:  
            plt.show()  

    def permutation_importance(self, n_repeats=10, save_path=None):
        perm_importance = permutation_importance(self.model, self.X, self.y, n_repeats=n_repeats, random_state=42)
    
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': perm_importance.importances_mean,
            'std': perm_importance.importances_std
        }).sort_values('importance', ascending=False).head(20)
    
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.errorbar(x=feature_importance['importance'], y=range(len(feature_importance)), 
                     xerr=feature_importance['std'], fmt='none', c='black', capsize=3)
        plt.title('Permutation Importance (top 20 features)')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def eli5_weights(self):  
        """  
        Display feature weights using ELI5.  
        """  
        return explain_weights(self.model, feature_names=self.feature_names)  

    def plot_interactive_force(self, index):  
        """  
        Create an interactive force plot using Plotly.  

        Args:  
            index (int): Index of the instance to explain.  
        """  
        shap_values = self.explainer.shap_values(self.X_test)  
        if isinstance(shap_values, list):  
            shap_values = shap_values[0]  
        expected_value = self.explainer.expected_value[0] if isinstance(self.explainer.expected_value, list) else self.explainer.expected_value  

        instance = self.X_test.iloc[index]  
        feature_names = self.feature_names  
        shap_values_instance = shap_values[index]  

        cumulative_values = np.cumsum(shap_values_instance)  
        base_value = expected_value  

        fig = make_subplots(rows=1, cols=1)  

        # Add base value line  
        fig.add_shape(type="line",  
                      x0=-0.5, y0=base_value, x1=len(feature_names)-0.5, y1=base_value,  
                      line=dict(color="red", width=2, dash="dash"))  

        # Add feature contributions  
        for i, (feature, value) in enumerate(zip(feature_names, shap_values_instance)):  
            if i == 0:  
                y_start = base_value  
            else:  
                y_start = cumulative_values[i-1] + base_value  
            y_end = cumulative_values[i] + base_value  
            
            color = "blue" if np.any(value >= 0) else "red"
            
            fig.add_trace(go.Scatter(x=[i, i], y=[y_start, y_end], mode='lines',  
                                     line=dict(color=color, width=20), name=feature))  

        # Customize layout  
        fig.update_layout(title="SHAP Force Plot",  
                          xaxis_title="Features",  
                          yaxis_title="Feature Impact",  
                          showlegend=False)  
        fig.update_xaxes(ticktext=feature_names, tickvals=list(range(len(feature_names))))  

        fig.show()  

    def save_model(self, filepath):  
        """  
        Save the model to a file.  

        Args:  
            filepath (str): Path to save the model file.  
        """  
        joblib.dump(self.model, filepath)  
        print(f"Model saved to {filepath}")  

    def load_model(self, filepath):  
        """  
        Load a model from a file.  

        Args:  
            filepath (str): Path to the saved model file.  
        """  
        self.model = joblib.load(filepath)  
        print(f"Model loaded from {filepath}")  

    def run_all_analyses(self, output_dir):  
        """  
        Run all analyses and save results to the specified output directory.  

        Args:  
            output_dir (str): Directory to save all output files.  
        """  
        import os  

        os.makedirs(output_dir, exist_ok=True)  

        analyses = [  
            ("shap_summary.png", self.plot_summary),  
            ("shap_force_plot.png", lambda p: self.plot_force(0, p)),  
            ("lime_explanation.html", lambda p: self.explain_instance_lime(0, save_path=p)),  
            ("feature_importance.png", self.feature_importance),  
            ("confusion_matrix.png", self.plot_confusion_matrix),  
            ("roc_curve.png", self.plot_roc_curve),  
            ("learning_curve.png", self.plot_learning_curve),  
            ("permutation_importance.png", self.permutation_importance)  
        ]  

        for filename, analysis_func in analyses:  
            try:  
                full_path = os.path.join(output_dir, filename)  
                if analysis_func.__name__ == '<lambda>':  
                    analysis_func(full_path)  
                else:  
                    analysis_func(save_path=full_path)  
                print(f"Saved {filename}")  
            except Exception as e:  
                print(f"Error in {analysis_func.__name__ if hasattr(analysis_func, '__name__') else 'analysis'} for {filename}: {str(e)}")  
                # 如果是 LIME 解释失败，创建一个空的 HTML 文件  
                if filename == "lime_explanation.html":  
                    with open(full_path, 'w') as f:  
                        f.write("<html><body><p>LIME explanation failed</p></body></html>")  
                    print(f"Created empty {filename} due to error")  

        try:  
            with open(os.path.join(output_dir, "eli5_weights.txt"), "w") as f:  
                f.write(str(self.eli5_weights()))  
        except Exception as e:  
            print(f"Error saving ELI5 weights: {str(e)}")  

        print(f"All analyses completed and saved to {output_dir}")   
        
if __name__ == "__main__":  
    # Example usage  
    from sklearn.datasets import load_iris  
    from sklearn.ensemble import RandomForestClassifier  

    # Load Iris dataset  
    iris = load_iris()  
    X = pd.DataFrame(iris.data, columns=iris.feature_names)  
    y = pd.Series(iris.target)  

    # Create and train a model  
    model = RandomForestClassifier(n_estimators=100, random_state=42)  
    model.fit(X, y)  

    # Create ModelInterpreter instance  
    interpreter = ModelInterpreter(model, X, y, iris.feature_names)  

    # Run all analyses  
    interpreter.run_all_analyses("iris_analysis_results")