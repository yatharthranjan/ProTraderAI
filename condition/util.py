
from datetime import time


class Cache():
    def __init__(self, max_size=1000, **kwargs):
        self.max_size = max_size

    def get(self, key):
        pass

    def set(self, key, value):
        pass

    def clear(self):
        pass


class TimeCache(Cache):

    def __init__(self, max_size=1000, refresh_time=60, **kwargs):
        super().__init__(max_size)
        self.cache = {}
        self.refresh_time = refresh_time

    def get(self, key):
        if key in self.cache:
            if self.cache[key][1] + self.refresh_time > time.time():
                return self.cache[key][0]
            else:
                del self.cache[key]
        return None

    def set(self, key, value):
        if len(self.cache) >= self.max_size:
            self.cache.popitem()
        self.cache[key] = (value, time.time())

    def clear(self):
        self.cache = {}