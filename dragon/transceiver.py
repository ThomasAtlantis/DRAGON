import queue
import struct
import time
from typing import List, Tuple, Protocol
from dragon.config import DragonConfig
from dragon.utils.mlogging import Logger
from logging import Logger as PyLogger
import threading
import socket


class Message:
    ACK = 0
    READY_FOR_CONNECTION = 1
    READY_FOR_GENERATION = 2
    GENERATE = 3

    @staticmethod
    def unpack(data: bytes) -> Tuple[int, bytes]:
        header_size = struct.calcsize("B I")
        if len(data) < header_size:
            raise ValueError("Incomplete message")
        mtype, body_len = struct.unpack("B I", data[:header_size])
        total_size = header_size + body_len
        if len(data) < total_size:
            raise ValueError("Incomplete message")
        return mtype, data[header_size:total_size]
    
    @staticmethod
    def pack(mtype: int, mbody: bytes) -> bytes:
        if not 0 <= mtype <= 255:
            raise ValueError("mtype should be in range 0-255")
        return struct.pack("B I", mtype, len(mbody)) + mbody


class ReceiveListener(threading.Thread):
    def __init__(
            self, socket: socket.socket, 
            queue: queue.Queue,
            buff_size: int, logger: PyLogger):
        threading.Thread.__init__(self)
        self.stop_event = threading.Event()
        self.queue = queue
        self.socket = socket
        self.socket.listen(1)
        self.buff_size = buff_size
        self.logger = logger

    def stop(self):
        self.stop_event.set()

    def run(self):
        conn, addr = self.socket.accept()
        self.logger.info(f"Connection accepted from {addr}")
        while not self.stop_event.is_set():
            if data := conn.recv(self.buff_size):
                self.logger.info(f"Received message of size {len(data)}")
                self.queue.put(data)
        self.logger.info("Connection closed.")
        conn.close()


class Observer(Protocol):
    def __call__(self, mtype: int, mbody: bytes):
        ...


class ReceiveHandler(threading.Thread):
    def __init__(
            self, queue: queue.Queue,
            observers: List[Observer], logger: PyLogger):
        threading.Thread.__init__(self)
        self.stop_event = threading.Event()
        self.queue = queue
        self.observers = observers
        self.logger = logger

    def stop(self):
        self.stop_event.set()
    
    def run(self):
        while not self.stop_event.is_set():
            if not self.queue.empty():
                message = self.queue.get()
                self.notify(message)
            time.sleep(0.0001)

    def notify(self, message: bytes):
        mtype, mbody = Message.unpack(message)
        for observer in self.observers:
            self.logger.info(f"Notifying observer `{observer.__name__}` ...")
            observer(mtype, mbody)


class Transceiver:
    
    def __init__(self, config: DragonConfig):
        self.config = config
        self.rank = config.trans.rank
        self.name = f"Node{config.trans.rank:>02}"
        self.logger = Logger.build(self.name, level='INFO')
        self.buff_size, self.timeout_ms = 1024, 500
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
            self.rx_socket, self.receive_queue, self.buff_size, self.logger)
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

    def send(self, mtype: int, mbody: bytes):
        message = Message.pack(mtype, mbody)
        self.tx_socket.send(message)
        self.logger.info(f"Sent message of type {mtype}")
    
    def register_observers(self, observers: List[Observer]):
        self.receive_handler = ReceiveHandler(
            self.receive_queue, observers, self.logger)
        self.receive_handler.start()

    def terminate(self):
        print()
        self.receive_listener.stop()
        self.receive_listener.join()
        self.receive_handler.stop()
        self.receive_handler.join()
        if self.rx_socket.fileno() != -1:
            self.rx_socket.close()
        if self.tx_socket.fileno() != -1:
            self.tx_socket.close()
        self.logger.info("Transceiver stopped.")
