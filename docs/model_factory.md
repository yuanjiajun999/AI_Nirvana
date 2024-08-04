# ModelFactory  

`ModelFactory` 是一个用于管理和创建实现了 `ModelInterface` 的机器学习模型的工厂类。  

## 概述  

这个类提供了一种集中式的方法来注册、创建和管理不同类型的机器学习模型。所有被管理的模型都必须实现 `ModelInterface` 接口。  

## 属性  

- `_models (Dict[str, Type[ModelInterface]])`: 存储已注册模型类的字典。  

## 方法  

### register_model  

```python  
@classmethod  
def register_model(cls, name: str, model_class: Type[ModelInterface]):  
注册新的模型类。

参数
name (str): 用于注册模型的名称。
model_class (Type[ModelInterface]): 要注册的模型类。
示例
ModelFactory.register_model("custom", CustomModel)  
create_model
@classmethod  
def create_model(cls, name: str, **kwargs) -> ModelInterface:  
创建已注册模型的实例。

参数
name (str): 要创建的模型名称。
**kwargs: 传递给模型构造函数的额外参数。
返回
ModelInterface: 请求的模型实例。
异常
ValueError: 如果请求的模型未注册。
示例
model = ModelFactory.create_model("custom", param1=5, param2=0.9)  
get_available_models
@classmethod  
def get_available_models(cls) -> List[str]:  
获取所有已注册模型的名称列表。

返回
List[str]: 已注册模型名称的列表。
示例
available_models = ModelFactory.get_available_models()  
使用示例
请参考 model_factory_example.py 文件以获取完整的使用示例。