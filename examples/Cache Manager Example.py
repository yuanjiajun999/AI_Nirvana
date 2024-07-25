from src.utils.cache_manager import CacheManager
import time


def main():
    # 初始化缓存管理器，设置过期时间为5秒（为了演示方便）
    cache_manager = CacheManager(
        cache_file="example_cache.json", expiration_days=5 / 86400
    )

    # 设置缓存项
    cache_manager.set("key1", "value1")
    cache_manager.set("key2", {"nested": "data"})

    # 获取缓存项
    print("Retrieved key1:", cache_manager.get("key1"))
    print("Retrieved key2:", cache_manager.get("key2"))

    # 尝试获取不存在的键
    print("Non-existent key:", cache_manager.get("key3"))

    # 等待一段时间，让缓存过期
    print("Waiting for cache to expire...")
    time.sleep(6)

    # 再次尝试获取缓存项
    print("After expiration - key1:", cache_manager.get("key1"))
    print("After expiration - key2:", cache_manager.get("key2"))

    # 清理过期的缓存项
    cache_manager.clear_expired()

    # 设置新的缓存项
    cache_manager.set("key4", "new_value")
    print("New key4:", cache_manager.get("key4"))


if __name__ == "__main__":
    main()
