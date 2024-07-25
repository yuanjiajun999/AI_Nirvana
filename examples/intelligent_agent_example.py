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


class SimpleAgent(IntelligentAgent):
    def perceive(self, observation):
        return int(observation.split(":")[1])

    def decide(self, state):
        return "increment" if state < 5 else "decrement"

    def act(self, action):
        return action


def main():
    env = SimpleEnvironment()
    agent = SimpleAgent()

    print("Starting Intelligent Agent simulation:")
    for i in range(10):
        observation = env.get_observation()
        print(f"\nStep {i+1}")
        print("Observation:", observation)

        state = agent.perceive(observation)
        action = agent.decide(state)
        print("Chosen action:", action)

        result = env.take_action(action)
        print("Result:", result)

    print("\nSimulation completed.")


if __name__ == "__main__":
    main()
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


class SimpleAgent(IntelligentAgent):
    def perceive(self, observation):
        return int(observation.split(":")[1])

    def decide(self, state):
        return "increment" if state < 5 else "decrement"

    def act(self, action):
        return action


def main():
    env = SimpleEnvironment()
    agent = SimpleAgent()

    print("Starting Intelligent Agent simulation:")
    for i in range(10):
        observation = env.get_observation()
        print(f"\nStep {i+1}")
        print("Observation:", observation)

        state = agent.perceive(observation)
        action = agent.decide(state)
        print("Chosen action:", action)

        result = env.take_action(agent.act(action))
        print("Result:", result)

    print("\nSimulation completed.")


if __name__ == "__main__":
    main()
