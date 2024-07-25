from src.core.digital_twin import DigitalTwin
import numpy as np


def simple_physical_model(state, t):
    # 简单的物理模型：弹簧 - 质量系统
    k = 1.0  # 弹簧常数
    m = 1.0  # 质量
    return np.array([state[1], -k / m * state[0]])


def main():
    # 初始化数字孪生
    digital_twin = DigitalTwin(simple_physical_model)

    # 模拟物理系统
    initial_conditions = [1.0, 0.0]  # 初始位置和速度
    time_steps = np.linspace(0, 10, 100)
    states = digital_twin.simulate(initial_conditions, time_steps)

    print("Digital Twin Simulation Results:")
    print("Time | Position | Velocity")
    for t, state in zip(time_steps[::10], states[::10]):
        print(f"{t:.2f} | {state[0]:.2f} | {state[1]:.2f}")

    # 监控异常
    sensor_data = states + np.random.normal(0, 0.1, states.shape)
    anomalies = digital_twin.monitor(sensor_data)
    print("\nAnomalies detected:", anomalies)

    # 优化参数
    def objective_function(params):
        k, m = params
        model = lambda state, t: np.array([state[1], -k / m * state[0]])
        simulated_states = digital_twin.simulate(initial_conditions, time_steps, model)
        return np.sum((simulated_states - sensor_data) ** 2)

    optimal_params = digital_twin.optimize(objective_function, [(0.5, 1.5), (0.5, 1.5)])
    print("\nOptimized parameters:", optimal_params)


if __name__ == "__main__":
    main()
