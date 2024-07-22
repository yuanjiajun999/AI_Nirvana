import json
from datetime import datetime, timedelta
from typing import Any, Optional


class CacheManager:
    def __init__(self, cache_file: str = "cache.json", expiration_days: int = 7):
        self.cache_file = cache_file
        self.expiration_days = expiration_days
        self.cache = self.load_cache()

    def load_cache(self) -> dict:
        try:
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            item = self.cache[key]
            if datetime.now() < datetime.fromisoformat(item['expiration']):
                return item['value']
            else:
                del self.cache[key]
                self.save_cache()
        return None

    def set(self, key: str, value: Any):
        expiration = (datetime.now() + timedelta(days=self.expiration_days)).isoformat()
        self.cache[key] = {'value': value, 'expiration': expiration}
        self.save_cache()

    def clear_expired(self):
        now = datetime.now()
        expired_keys = [k for k, v in self.cache.items() if now >= datetime.fromisoformat(v['expiration'])]
        for key in expired_keys:
            del self.cache[key]
        self.save_cache()