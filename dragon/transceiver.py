import queue
import struct
import pickle
import time
import threading
import socket

from typing import List, Protocol, Tuple
from logging import Logger as PyLogger
from dragon.config import DragonConfig
from dragon.utils.mlogging import Logger


class Message:
    ACK = 0
    READY_FOR_CONNECTION = 1
    READY_FOR_GENERATION = 2
    BEGIN_GENERATE = 3
    DRAFT_TOKEN = 4
    TARGET_TOKEN = 5
    SHUTDOWN = 6
    header = "=B I"

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

mtype2str = {getattr(Message, attr): attr for attr in dir(Message) if not attr.startswith("_")}

class ReceiveListener(threading.Thread):
    def __init__(
            self, socket: socket.socket, 
            queue: queue.Queue,
            logger: PyLogger):
        threading.Thread.__init__(self)
        self.stop_event = threading.Event()
        self.queue = queue
        self.socket = socket
        self.socket.listen(1)
        self.logger = logger
        self.header_size = struct.calcsize(Message.header)

    def stop(self):
        self.stop_event.set()

    def run(self):
        conn, addr = self.socket.accept()
        self.logger.info(f"Connection accepted from {addr}")
        while not self.stop_event.is_set():
            header = b''
            while len(header) < self.header_size:
                chunk = conn.recv(self.header_size - len(header))
                if not chunk:
                    self.logger.error("Connection closed prematurely.")
                    conn.close()
                    return
                header += chunk
            mtype, body_len = struct.unpack(Message.header, header)
            mbody = b''
            while len(mbody) < body_len:
                chunk = conn.recv(body_len - len(mbody))
                if not chunk:
                    self.logger.error("Incomplete message body received.")
                    conn.close()
                    return
                mbody += chunk
            self.queue.put((mtype, mbody))
            self.logger.info(f"Received Message(mtype={mtype2str[mtype]}, len={body_len})")
        self.logger.info("Connection closed.")
        conn.close()


class Observer(Protocol):
    def __call__(self, mtype: int, mbody: object) -> bool:
        ...


class ReceiveHandler(threading.Thread):
    def __init__(
            self, queue: queue.Queue,
            observers: List[Observer]):
        threading.Thread.__init__(self)
        self.stop_event = threading.Event()
        self.queue = queue
        self.observers = observers
        # self.logger = Logger.build(__class__.__name__, level="INFO")

    def stop(self):
        self.stop_event.set()
    
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
        self.config = config
        self.rank = config.trans.rank
        self.name = f"Node{config.trans.rank:>02}"
        self.logger = Logger.build(self.name, level='INFO')
        self.init_receiver(port=self.config.trans.rx_port)
        self.logger.info("Receiver initialized.")
        self.init_sender(port=self.config.trans.tx_port)
        self.logger.info("Sender initialized.")

    def init_receiver(self, port):
        self.rx_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            try:
                self.rx_socket.bind(('localhost', port))
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
                self.tx_socket.connect(('localhost', port))
                break
            except ConnectionRefusedError:
                time.sleep(3)

    def send(self, mtype: int, mbody: object):
        data, body_len = Message.pack(mtype, mbody)
        self.tx_socket.send(data)
        self.logger.info(f"Sent Message(mtype={mtype2str[mtype]}, len={body_len})")
    
    def register_observers(self, observers: List[Observer]):
        self.receive_handler = ReceiveHandler(
            self.receive_queue, observers)
        self.receive_handler.start()

    def terminate(self):
        print()
        self.receive_listener.stop()
        if self.receive_listener.is_alive():
            self.receive_listener.join()
        self.receive_handler.stop()
        if self.rx_socket.fileno() != -1:
            self.rx_socket.close()
        if self.tx_socket.fileno() != -1:
            self.tx_socket.close()
        self.logger.info("Transceiver stopped.")
