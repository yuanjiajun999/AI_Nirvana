# examples/intelligent_agent_example.py

from src.core.intelligent_agent import IntelligentAgent

class SimpleEnvironment:
    def __init__(self):
        self.state = 0

    def get_observation(self):
        return f"Current state: {self.state}"

    def take_action(self, action):
        if action == "increment":
            self.state += 1
        elif action == "decrement":
            self.state -= 1
        return f"Action taken: {action}. New state: {self.state}"

def main():
    env = SimpleEnvironment()
    agent = IntelligentAgent()

    print("Starting Intelligent Agent simulation:")
    for i in range(5):
        observation = env.get_observation()
        print(f"\nStep {i+1}")
        print("Observation:", observation)
        
        action = agent.decide(observation)
        print("Chosen action:", action)
        
        result = env.take_action(action)
        print("Result:", result)
        
        agent.learn(observation, action, result)

    print("\nSimulation completed.")

if __name__ == "__main__":
    main()