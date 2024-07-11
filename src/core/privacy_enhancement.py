from src.config import Config
import differential_privacy as dp

class PrivacyEnhancement:
    def __init__(self, config: Config):
        self.config = config
        self.epsilon = config.get('privacy_epsilon')
        self.delta = config.get('privacy_delta')

    def apply_differential_privacy(self, data):
        return dp.apply_dp(data, self.epsilon, self.delta)