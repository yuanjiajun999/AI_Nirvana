import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import odeint
from typing import Callable, List, Tuple, Any
from src.utils.error_handler import error_handler, logger
from scipy.optimize import minimize, differential_evolution

class DigitalTwin:
    def __init__(self, physical_system_model: Callable):
        """
        初始化数字孪生对象。

        Args:
            physical_system_model (Callable): 物理系统模型函数。
        """
        self.physical_system_model = physical_system_model
        logger.info("DigitalTwin initialized with physical system model.")

    @error_handler
    def simulate(self, initial_conditions: List[float], time_steps: List[float]) -> np.ndarray:
        """
        模拟物理系统的状态变化。

        Args:
            initial_conditions (List[float]): 初始条件。
            time_steps (List[float]): 时间步长。

        Returns:
            np.ndarray: 系统状态随时间的变化。
        """
        states = odeint(self.physical_system_model, initial_conditions, time_steps)
        logger.info(f"Simulation completed for {len(time_steps)} time steps.")
        return states

    @error_handler
    def monitor(self, sensor_data: np.ndarray) -> List[Any]:
        """
        监测系统状态，检测异常。

        Args:
            sensor_data (np.ndarray): 传感器数据。

        Returns:
            List[Any]: 检测到的异常列表。
        """
        if not hasattr(self.physical_system_model, 'detect_anomalies'):
            raise AttributeError("Physical system model does not have a 'detect_anomalies' method.")
        
        anomalies = self.physical_system_model.detect_anomalies(sensor_data)
        logger.info(f"Monitoring completed. Detected {len(anomalies)} anomalies.")
        return anomalies

    @error_handler
    def optimize(self, objective_function: Callable, constraints: List[Tuple], method='COBYLA') -> Any:
        """
        优化系统参数。

        Args:
            objective_function (Callable): 优化目标函数。
            constraints (List[Tuple]): 优化约束条件。
            method (str): 优化方法，可选 'COBYLA' 或 'differential_evolution'。

        Returns:
            Any: 优化后的参数。
        """
        if method == 'COBYLA':
            constraint_dicts = [{'type': c_type, 'fun': c_func} for c_func, c_type in constraints]
            result = minimize(objective_function, x0=[1.0, 1.0], method='COBYLA', constraints=constraint_dicts)
        elif method == 'differential_evolution':
            bounds = [(-10, 10), (-10, 10)]  # 假设有两个参数，每个参数的范围是 -10 到 10
            result = differential_evolution(objective_function, bounds, constraints=constraints)
        else:
            raise ValueError("Unsupported optimization method")
    
        logger.info(f"Optimization completed using {method} method.")
        return result.x

    @error_handler
    def update_model(self, new_model: Callable) -> None:
        """
        更新物理系统模型。

        Args:
            new_model (Callable): 新的物理系统模型函数。
        """
        self.physical_system_model = new_model
        logger.info("Physical system model updated.")
    
    @error_handler
    def validate_model(self, validation_data: np.ndarray) -> float:
        """
        验证模型的准确性。

        Args:
            validation_data (np.ndarray): 用于验证的数据。

        Returns:
            float: 模型的准确性评分。
        """
        # 这里应该实现具体的验证逻辑
        # 作为示例，我们可以使用一个简单的方法来计算准确性
        predicted_values = self.simulate(validation_data[0], validation_data[1:])
        actual_values = np.array(validation_data[1:])
        mse = np.mean((predicted_values - actual_values) ** 2)
        accuracy = 1 / (1 + mse)  # 将均方误差转换为0-1范围的准确度
        logger.info(f"Model validation completed. Accuracy: {accuracy}")
        return accuracy
    
    def preprocess_data(self, data):
        # 将输入数据转换为 numpy 数组
        data_array = np.array(data)
    
        # 如果数据量太少，无法进行有意义的预处理，则直接返回原始数据
        if len(data_array) < 4:
           return data_array

        # 使用 pandas Series 来进行分位数裁剪
        data_series = pd.Series(data_array)
        data_clipped = data_series.clip(lower=data_series.quantile(0.01), upper=data_series.quantile(0.99))

        # 平滑处理
        # 注意：window_length 必须是奇数，且小于数据长度
        window_length = min(5, len(data_array) - 1 if len(data_array) % 2 == 0 else len(data_array))
        window_length = max(3, window_length)  # 确保 window_length 至少为 3
        data_smoothed = signal.savgol_filter(data_clipped, window_length, polyorder=2)

        return data_smoothed

    def postprocess_results(self, results):
        # 在这里可以添加结果的后处理逻辑，例如格式化、单位转换等
        return results

    @error_handler
    def simulate(self, initial_conditions: List[float], time_steps: List[float]) -> np.ndarray:
        """
        模拟物理系统的状态变化。

        Args:
            initial_conditions (List[float]): 初始条件。
            time_steps (List[float]): 时间步长。

        Returns:
            np.ndarray: 系统状态随时间的变化。
        """
        initial_conditions = self.preprocess_data(initial_conditions)
        if len(initial_conditions) == 1:
            initial_conditions = [initial_conditions[0], initial_conditions[0]]  # 假设第二个状态变量的初始值与第一个相同
        states = odeint(self.physical_system_model, initial_conditions, time_steps)
        logger.info(f"Simulation completed for {len(time_steps)} time steps.")
        return self.postprocess_results(states)
    
    def visualize_simulation(self, states, time_steps):
        plt.figure(figsize=(10, 6))
        for i in range(states.shape[1]):
            plt.plot(time_steps, states[:, i], label=f'State {i+1}')
        plt.xlabel('Time')
        plt.ylabel('State Value')
        plt.title('Digital Twin Simulation Results')
        plt.legend()
        plt.grid(True)
        plt.show()