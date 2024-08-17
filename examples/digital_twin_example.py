from src.core.digital_twin import DigitalTwin
import numpy as np

def simple_pendulum_model(state, t, L=1.0, g=9.81):
    theta, omega = state
    dtheta_dt = omega
    domega_dt = -g/L * np.sin(theta)
    return [dtheta_dt, domega_dt]

# 不再需要单独的 PendulumSystem 类
def detect_anomalies(sensor_data):
    return [i for i, value in enumerate(sensor_data) if abs(value) > np.pi/2]

simple_pendulum_model.detect_anomalies = detect_anomalies

def main():
    # 初始化数字孪生
    pendulum_twin = DigitalTwin(simple_pendulum_model)

    # 模拟
    initial_conditions = [np.pi/4, 0]  # 初始角度和角速度
    time_steps = np.linspace(0, 10, 1000)
    states = pendulum_twin.simulate(initial_conditions, time_steps)
    print("Simulation completed. Final state:", states[-1])

    # 监测
    sensor_data = states[:, 0]  # 使用角度数据作为传感器数据
    anomalies = pendulum_twin.monitor(sensor_data)
    print("Detected anomalies at time steps:", anomalies)

    # 优化
    def objective_function(params):
        L, g = params
        return abs(2 * np.pi * np.sqrt(L/g) - 1)  # 优化周期为1秒的摆长

    constraints = [(lambda x: 0.1 - x[0], 'ineq'), (lambda x: x[0] - 2.0, 'ineq'),
                   (lambda x: 9.0 - x[1], 'ineq'), (lambda x: x[1] - 10.0, 'ineq')]
    optimal_params = pendulum_twin.optimize(objective_function, constraints)
    print("Optimized parameters (L, g):", optimal_params)

    # 验证
    validation_data = np.random.rand(100) * np.pi - np.pi/2
    accuracy = pendulum_twin.validate_model(validation_data)
    print("Model validation accuracy:", accuracy)

    # 可视化
    pendulum_twin.visualize_simulation(states, time_steps)

if __name__ == "__main__":
    main()