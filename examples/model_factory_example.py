# model_factory_example.py  

from model_factory import ModelFactory  
from model_interface import ModelInterface  

# 定义一个自定义模型  
class CustomModel(ModelInterface):  
    def __init__(self, param1, param2):  
        self.param1 = param1  
        self.param2 = param2  

    def train(self, training_data, training_labels):  
        print(f"Training CustomModel with param1={self.param1}, param2={self.param2}")  

    def predict(self, input_data):  
        return [self.param1] * len(input_data)  

    def evaluate(self, test_data, test_labels):  
        return {"accuracy": self.param2}  

# 注册自定义模型  
ModelFactory.register_model("custom", CustomModel)  

# 创建自定义模型实例  
model = ModelFactory.create_model("custom", param1=5, param2=0.9)  

# 使用模型  
model.train([1, 2, 3], [0, 1, 1])  
predictions = model.predict([4, 5, 6])  
evaluation = model.evaluate([7, 8, 9], [1, 0, 1])  

print(f"Predictions: {predictions}")  
print(f"Evaluation: {evaluation}")  

# 获取可用模型列表  
available_models = ModelFactory.get_available_models()  
print(f"Available models: {available_models}")