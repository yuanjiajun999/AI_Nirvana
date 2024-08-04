import unittest
import os
import joblib
import sys
import numpy as np
import shutil
import lime  
import lime.lime_tabular
import shap
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from src.core.model_interpretability import ModelInterpreter
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  
import matplotlib  
matplotlib.use('Agg')  
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestModelInterpreter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Setup the class with required initial data and models"""  
        # Load Iris dataset  
        cls.iris = load_iris()  
        cls.X = pd.DataFrame(cls.iris.data, columns=cls.iris.feature_names)  
        cls.y = cls.iris.target  
        cls.feature_names = cls.iris.feature_names 
        # Create and train a model  
        cls.model = RandomForestClassifier(n_estimators=100, random_state=42)  
        cls.model.fit(cls.X, cls.y)  

        # Create ModelInterpreter instance  
        cls.interpreter = ModelInterpreter(  
            cls.model,  
            cls.X,  
            cls.y,  
            feature_names=cls.iris.feature_names,  
            class_names=['setosa', 'versicolor', 'virginica'],  
            model_type='tree'  
        )  
        
        # Create output directory for tests  
        cls.output_dir = "test_output"  
        os.makedirs(cls.output_dir, exist_ok=True) 

    @classmethod
    def tearDownClass(cls):
        """Clean up the test output directory"""
        if os.path.exists(cls.output_dir):
            shutil.rmtree(cls.output_dir)

    def test_create_explainer_shap(self):
        self.interpreter.create_explainer('shap')
        self.assertIsNotNone(self.interpreter.explainer)

    def test_create_explainer_lime(self):
        self.interpreter.create_explainer('lime')
        self.assertIsNotNone(self.interpreter.lime_explainer)

    def test_get_shap_values(self):
        shap_values = self.interpreter.get_shap_values()
        self.assertTrue(isinstance(shap_values, (list, np.ndarray)))

    def test_plot_summary(self):
        self.interpreter.plot_summary(plot_type="bar", save_path=os.path.join(self.output_dir, "shap_summary_bar.png"))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "shap_summary_bar.png")))

    def test_plot_force(self):
        self.interpreter.plot_force(0, save_path=os.path.join(self.output_dir, "shap_force_plot.png"))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "shap_force_plot.png")))

    def test_explain_instance_lime(self):
        self.interpreter.explain_instance_lime(0, save_path=os.path.join(self.output_dir, "lime_explanation.png"))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "lime_explanation.png")))

    def test_feature_importance(self):
        self.interpreter.feature_importance(save_path=os.path.join(self.output_dir, "feature_importance.png"))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "feature_importance.png")))

    def test_decision_path_visualization(self):  
        save_path_png = os.path.join(self.output_dir, "decision_path.png")  
        save_path_svg = os.path.join(self.output_dir, "decision_path.svg")  
        self.interpreter.decision_path_visualization(0, save_path=save_path_png)  
    
        # 检查 PNG 或 SVG 文件是否存在  
        file_exists = os.path.exists(save_path_png) or os.path.exists(save_path_svg)  
        self.assertTrue(file_exists, f"File not found: neither {save_path_png} nor {save_path_svg}")  
    
        # 如果文件未找到，打印输出目录的内容  
        if not file_exists:  
            print(f"Contents of {self.output_dir}:")  
            for file in os.listdir(self.output_dir):  
                print(file)

    def test_plot_decision_boundary(self):
        self.interpreter.plot_decision_boundary(feature1=self.iris.feature_names[0], feature2=self.iris.feature_names[1], save_path=os.path.join(self.output_dir, "decision_boundary.png"))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "decision_boundary.png")))

    def test_plot_partial_dependence(self):
        self.interpreter.plot_partial_dependence([0, 1], save_path=os.path.join(self.output_dir, "partial_dependence.png"))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "partial_dependence.png")))

    def test_plot_pdp_interact(self):
        self.interpreter.plot_pdp_interact([0, 1], save_path=os.path.join(self.output_dir, "pdp_interact.png"))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "pdp_interact.png")))

    def test_cross_validation(self):
        self.interpreter.cross_validation(cv=5)
        # No file to check, just ensure no exception is raised

    def test_plot_confusion_matrix(self):
        self.interpreter.plot_confusion_matrix(save_path=os.path.join(self.output_dir, "confusion_matrix.png"))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "confusion_matrix.png")))

    def test_plot_roc_curve(self):
        self.interpreter.plot_roc_curve(save_path=os.path.join(self.output_dir, "roc_curve.png"))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "roc_curve.png")))

    def test_plot_learning_curve(self):
        self.interpreter.plot_learning_curve(save_path=os.path.join(self.output_dir, "learning_curve.png"))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "learning_curve.png")))

    def test_permutation_importance(self):
        self.interpreter.permutation_importance(save_path=os.path.join(self.output_dir, "permutation_importance.png"))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "permutation_importance.png")))

    def test_eli5_weights(self):
        weights = self.interpreter.eli5_weights()
        self.assertIsNotNone(weights)

    def test_plot_interactive_force(self):
        # This method shows a plot and does not save to file, so we test by calling it
        self.interpreter.plot_interactive_force(0)
        # No file to check, just ensure no exception is raised

    def test_plot_feature_importance(self):  
        self.interpreter.plot_feature_importance(save_path=os.path.join(self.output_dir, "feature_importance.png"))  
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "feature_importance.png")))  

    def test_multiclass_partial_dependence(self):  
        # 创建一个多类分类问题的测试数据  
        X_multi = pd.DataFrame(np.random.rand(100, 4), columns=['A', 'B', 'C', 'D'])  
        y_multi = np.random.choice([0, 1, 2], size=100)  
        model_multi = RandomForestClassifier().fit(X_multi, y_multi)  
    
        interpreter_multi = ModelInterpreter(model_multi, X_multi, y_multi)  
        interpreter_multi.plot_partial_dependence(['A', 'B'], save_path=os.path.join(self.output_dir, "multi_pdp.png"), target_class=1)  
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "multi_pdp.png")))
    
    def test_save_and_load_model(self):
        # 保存原始模型
        model_path = os.path.join(self.output_dir, "test_model.joblib")
        self.interpreter.save_model(model_path)

        # 创建新的 ModelInterpreter 实例，暂时不传入模型
        new_interpreter = ModelInterpreter(
            None,
            self.X, 
            self.y, 
            feature_names=self.feature_names,
            class_names=self.interpreter.class_names,
            model_type=self.interpreter.model_type
        )

        # 加载保存的模型
        new_interpreter.load_model(model_path)

        # 验证加载的模型
        self.assertIsNotNone(new_interpreter.model)

        # 比较预测结果
        original_predictions = self.interpreter.model.predict(self.X)
        new_predictions = new_interpreter.model.predict(self.X)
        np.testing.assert_array_equal(original_predictions, new_predictions)

    def test_run_all_analyses(self):
        run_all_output_dir = os.path.join(self.output_dir, "run_all")
        self.interpreter.run_all_analyses(run_all_output_dir)
        self.assertTrue(os.path.exists(os.path.join(run_all_output_dir, "shap_summary.png")))
        self.assertTrue(os.path.exists(os.path.join(run_all_output_dir, "shap_force_plot.png")))
        self.assertTrue(os.path.exists(os.path.join(run_all_output_dir, "lime_explanation.html")))
        self.assertTrue(os.path.exists(os.path.join(run_all_output_dir, "feature_importance.png")))
        self.assertTrue(os.path.exists(os.path.join(run_all_output_dir, "confusion_matrix.png")))
        self.assertTrue(os.path.exists(os.path.join(run_all_output_dir, "roc_curve.png")))
        self.assertTrue(os.path.exists(os.path.join(run_all_output_dir, "learning_curve.png")))
        self.assertTrue(os.path.exists(os.path.join(run_all_output_dir, "permutation_importance.png")))
        self.assertTrue(os.path.exists(os.path.join(run_all_output_dir, "eli5_weights.txt")))

    def test_create_explainer(self):
        self.interpreter.create_explainer('shap')
        self.assertIsInstance(self.interpreter.explainer, shap.TreeExplainer)
        self.interpreter.create_explainer('lime')
        self.assertIsInstance(self.interpreter.lime_explainer, lime.lime_tabular.LimeTabularExplainer)
        with self.assertRaises(ValueError):
            self.interpreter.create_explainer('invalid_method')

    def test_plot_summary(self):
        self.interpreter.plot_summary(plot_type="bar", class_index=0, save_path="test_summary.png")
        self.assertTrue(os.path.exists("test_summary.png"))
        os.remove("test_summary.png")    

    def test_plot_force_and_lime(self):
        self.interpreter.plot_force(0, save_path="test_force.png")
        self.assertTrue(os.path.exists("test_force.png"))
        os.remove("test_force.png")

        self.interpreter.explain_instance_lime(0, save_path="test_lime.html")
        self.assertTrue(os.path.exists("test_lime.html"))
        os.remove("test_lime.html")   

    def test_feature_importance(self):
        self.interpreter.feature_importance(save_path="test_importance.png")
        self.assertTrue(os.path.exists("test_importance.png"))
        os.remove("test_importance.png") 

    def test_decision_path_and_boundary(self):  
        save_path = os.path.join(self.output_dir, "test_decision_path.svg")  
        self.interpreter.decision_path_visualization(0, save_path=save_path)  
        self.assertTrue(os.path.exists(save_path))  

        boundary_path = os.path.join(self.output_dir, "test_boundary.png")  
        self.interpreter.plot_decision_boundary(self.feature_names[0], self.feature_names[1], save_path=boundary_path)  
        self.assertTrue(os.path.exists(boundary_path))  

    def test_partial_dependence_and_interact(self):  
        # 获取数据集的前两个特征名  
        feature_names = self.X.columns.tolist()[:2]  
    
        # 测试 plot_partial_dependence  
        pdp_save_path = os.path.join(self.output_dir, "test_pdp.png")  
        self.interpreter.plot_partial_dependence(feature_names, save_path=pdp_save_path, target_class=0)  
        self.assertTrue(os.path.exists(pdp_save_path))  
        os.remove(pdp_save_path)  

        # 测试 plot_pdp_interact  
        pdp_interact_save_path = os.path.join(self.output_dir, "test_pdp_interact.png")  
        self.interpreter.plot_pdp_interact(feature_names, save_path=pdp_interact_save_path, target_class=0)  
        self.assertTrue(os.path.exists(pdp_interact_save_path))  
        os.remove(pdp_interact_save_path) 

    def test_confusion_matrix_and_roc(self):
        self.interpreter.plot_confusion_matrix(save_path="test_confusion.png")
        self.assertTrue(os.path.exists("test_confusion.png"))
        os.remove("test_confusion.png")

        self.interpreter.plot_roc_curve(save_path="test_roc.png")
        self.assertTrue(os.path.exists("test_roc.png"))
        os.remove("test_roc.png") 

    def test_learning_curve_and_permutation(self):
        self.interpreter.plot_learning_curve(save_path="test_learning.png")
        self.assertTrue(os.path.exists("test_learning.png"))
        os.remove("test_learning.png")

        self.interpreter.permutation_importance(save_path="test_permutation.png")
        self.assertTrue(os.path.exists("test_permutation.png"))
        os.remove("test_permutation.png")

    def test_eli5_and_interactive_force(self):
        weights = self.interpreter.eli5_weights()
        self.assertIsNotNone(weights)

        # Note: This test might need to be run in a notebook environment
        # self.interpreter.plot_interactive_force(0)   

    def test_save_and_load_model(self):
        self.interpreter.save_model("test_model.joblib")
        self.assertTrue(os.path.exists("test_model.joblib"))

        new_interpreter = ModelInterpreter(None, self.X, self.y, self.feature_names)
        new_interpreter.load_model("test_model.joblib")
        self.assertIsNotNone(new_interpreter.model)

        os.remove("test_model.joblib")          

if __name__ == "__main__":
    unittest.main()
