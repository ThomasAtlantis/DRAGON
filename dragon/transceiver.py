import queue
import struct
import pickle
import time
import threading
import socket

from typing import List, Protocol, Tuple
from logging import Logger as PyLogger
from .config import DragonConfig
from .utils.mlogging import Logger
from .utils.stable import terminate_thread

logging_level = "DEBUG"

class Message:
    ACK = 0
    READY_FOR_CONNECTION = 1
    READY_FOR_GENERATION = 2
    BEGIN_GENERATE = 3
    DRAFT_TOKEN = 4
    TARGET_TOKEN = 5
    SHUTDOWN = 6
    EMPTY = 7
    DRAT_SEQUENCE = 8
    PREPARE_COMPLETE = 9
    BEGIN_DECODE = 10
    KV_CACHE = 11
    header = ">B I"

    @staticmethod
    def unpack(data: bytes) -> object:
        return pickle.loads(data)
    
    @staticmethod
    def pack(mtype: int, mbody: object) -> Tuple[bytes, int]:
        if not 0 <= mtype <= 255:
            raise ValueError("mtype should be in range 0-255")
        mbody = pickle.dumps(mbody)
        data = struct.pack(Message.header, mtype, len(mbody)) + mbody
        return data, len(mbody)
    
    @staticmethod
    def empty_message():
        return Message.EMPTY, pickle.dumps(None)

mtype2str = {getattr(Message, attr): attr for attr in dir(Message) if not attr.startswith("_")}

class ReceiveListener(threading.Thread):
    def __init__(
            self, socket: socket.socket, 
            queue: queue.Queue,
            logger: PyLogger):
        threading.Thread.__init__(self, name=__class__.__name__)
        self.stop_event = threading.Event()
        self.queue = queue
        self.socket = socket
        self.socket.listen(1)
        self.logger = logger
        self.header_size = struct.calcsize(Message.header)
        self.conn = None
        self.logger.info(f"Protocol header size: {self.header_size} Bytes")

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def run(self):
        self.conn, addr = self.socket.accept()
        self.logger.info(f"Connection accepted from {addr}")
        while not self.stop_event.is_set():
            try:
                header = self.conn.recv(self.header_size, socket.MSG_WAITALL)
                mtype, body_len = struct.unpack(Message.header, header)
                mbody = self.conn.recv(body_len, socket.MSG_WAITALL)
            except Exception:
                break
            self.queue.put((mtype, mbody))
            if mtype in [Message.DRAFT_TOKEN, Message.TARGET_TOKEN]:
                self.logger.debug(f"Received Message(mtype={mtype2str[mtype]}, len={body_len}, token={pickle.loads(mbody)[0]})")
            else:
                self.logger.debug(f"Received Message(mtype={mtype2str[mtype]}, len={body_len})")
        self.logger.debug("Connection closed.")
        if self.conn: self.conn.close()

class Observer(Protocol):
    def __call__(self, mtype: int, mbody: object) -> bool:
        ...


class ReceiveHandler(threading.Thread):
    def __init__(
            self, queue: queue.Queue,
            observers: List[Observer]):
        threading.Thread.__init__(self, name=__class__.__name__)
        self.stop_event = threading.Event()
        self.queue = queue
        self.observers = observers
        # self.logger = Logger.build(__class__.__name__, level="INFO")

    def close(self):
        self.stop_event.set()
        self.queue.put(Message.empty_message())
    
    def run(self):
        while not self.stop_event.is_set():
            mtype, mbody = self.queue.get()
            mbody = Message.unpack(mbody)
            # self.logger.info(f"Unpacked Message(mtype={mtype2str[mtype]})")
            self.notify(mtype, mbody)

    def notify(self, mtype: int, mbody: object):
        for observer in self.observers:
            notified = observer(mtype, mbody)
            # if notified:
            #     self.logger.info(f"Notified observer `{observer.__name__}`.")


class Transceiver:
    
    def __init__(self, config: DragonConfig):
        self.logger = Logger.build(__class__.__name__, level=logging_level)
        self.config = config
        self.remote_ip = config.trans.tx_host
        self.local_ip = config.trans.rx_host
        self.init_receiver(port=self.config.trans.rx_port)
        self.logger.info("Receiver initialized.")
        self.init_sender(port=self.config.trans.tx_port)
        self.logger.info("Sender initialized.")

    def init_receiver(self, port):
        self.rx_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            try:
                self.rx_socket.bind((self.local_ip, port))
                break
            except OSError:
                time.sleep(3)

        self.receive_queue = queue.Queue(0)
        self.receive_listener = ReceiveListener(
            self.rx_socket, self.receive_queue, self.logger)
        self.receive_listener.start()
    
    def init_sender(self, port):
        self.tx_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            try:
                self.logger.info("Trying to connect to the remote node...")
                self.tx_socket.connect((self.remote_ip, port))
                break
            except ConnectionRefusedError:
                time.sleep(3)
            except OSError as e:
                self.logger.error(e)
                self.tx_socket.close()
                self.tx_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                time.sleep(3)
        self.send_queue = queue.Queue(0)
        self.send_thread = threading.Thread(
            target=self._send_thread, name="SendThread")
        self.send_thread.start()

    def send(self, mtype: int, mbody: object):
        self.send_queue.put((mtype, mbody))

    def _send_thread(self):
        while True:
            mtype, mbody = self.send_queue.get()
            if mtype == Message.EMPTY:
                break
            data, body_len = Message.pack(mtype, mbody)
            self.tx_socket.sendall(data)
            if mtype in [Message.DRAFT_TOKEN, Message.TARGET_TOKEN]:
                self.logger.debug(f"Sent Message(mtype={mtype2str[mtype]}, len={body_len}, token={mbody[0]})")
            else:
                self.logger.debug(f"Sent Message(mtype={mtype2str[mtype]}, len={body_len})")
    
    def register_observers(self, observers: List[Observer]):
        self.receive_handler = ReceiveHandler(
            self.receive_queue, observers)
        self.receive_handler.start()

    def terminate(self):
        self.rx_socket.close()
        self.tx_socket.close()
        self.logger.info("Socket closed.")
        self.send_queue.put(Message.empty_message())
        self.send_thread.join()
        terminate_thread(self.receive_listener)
        terminate_thread(self.receive_handler)
        self.logger.info("Transceiver stopped.")
