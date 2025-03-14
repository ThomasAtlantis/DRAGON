import ctypes
import random
import threading
import numpy as np
import torch


def seed_everything(seed):
    torch.manual_seed(seed)       # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)          # Numpy module
    random.seed(seed)             # Python random module
    torch.backends.cudnn.benchmark = False    # Close optimization
    torch.backends.cudnn.deterministic = True # Close optimization
    torch.cuda.manual_seed_all(seed) # All GPU (Optional)

def terminate_thread(thread: threading.Thread):
    if not thread or not thread.is_alive(): return
    if hasattr(thread, 'close'):
        thread.close()
        if thread is not threading.current_thread():
            thread.join()
        return
    thread_id = thread.ident
    exc = ctypes.py_object(SystemExit)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), exc)
    if res == 0:
        raise ValueError("Non-existent thread id")
    elif res != 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), None)
        raise SystemError("PyThreadState_SetAsyncExc failed")