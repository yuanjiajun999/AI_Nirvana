import abc
from typing import Any

class IntelligentAgent(abc.ABC):
    @abc.abstractmethod
    def perceive(self, observation: Any) -> Any:
        """Perceive the environment and update the agent's internal state."""
        pass

    @abc.abstractmethod
    def decide(self, state: Any) -> Any:
        """Decide on the next action based on the current state."""
        pass

    @abc.abstractmethod
    def act(self, action: Any) -> Any:
        """Perform the chosen action and update the environment."""
        pass

    def run(self):
        observation = self.perceive(None)
        state = self.decide(observation)
        result = self.act(state)
        return result