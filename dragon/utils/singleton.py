import threading


class SingletonType(type):

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        cls._instance = None
        cls._lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__call__(*args, **kwargs)
        return cls._instance
