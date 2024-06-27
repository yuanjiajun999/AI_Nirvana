import yaml
import os
import codecs

class Config:
    def __init__(self, config_path='config/default_config.yaml'):
        self.config_path = config_path
        self.load_config()

    def load_config(self):
        if os.path.exists(self.config_path):
            try:
                with codecs.open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
            except UnicodeDecodeError:
                # 如果 UTF-8 解码失败，尝试使用 'utf-8-sig' 编码（处理带 BOM 的 UTF-8 文件）
                with codecs.open(self.config_path, 'r', encoding='utf-8-sig') as f:
                    self.config = yaml.safe_load(f)
            except Exception as e:
                print(f"Error loading config: {e}")
                self.config = {}
        else:
            print(f"Config file not found: {self.config_path}")
            self.config = {}

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value
        self.save_config()

    def save_config(self):
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f)