import numpy as np
from scipy.integrate import odeint


class DigitalTwin:
    def __init__(self, physical_system_model):
        self.physical_system_model = physical_system_model

    def simulate(self, initial_conditions, time_steps):
        states = odeint(self.physical_system_model, initial_conditions, time_steps)
        return states

    def monitor(self, sensor_data):
        anomalies = self.physical_system_model.detect_anomalies(sensor_data)
        return anomalies

    def optimize(self, objective_function, constraints):
        optimal_parameters = self.physical_system_model.optimize(objective_function, constraints)
        return optimal_parameters