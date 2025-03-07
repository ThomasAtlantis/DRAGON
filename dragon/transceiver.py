import queue
import struct
import time
from typing import List, Tuple, Protocol
from dragon.config import DragonConfig
from dragon.utils.mlogging import Logger
import threading
import zmq


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
            self, socket: zmq.SyncSocket, 
            queue: queue.Queue,
            s_buff: int = 1024):
        threading.Thread.__init__(self)
        self.stop_event = threading.Event()
        self.queue = queue
        self.socket = socket
        self.s_buff = s_buff

    def stop(self):
        self.stop_event.set()

    def run(self):
        while not self.stop_event.is_set():
            try:
                data = self.socket.recv(self.s_buff)
                self.queue.put(data)
                ack_message = Message.pack(Message.ACK, b"")
                self.socket.send(ack_message)
            except zmq.Again:
                pass


class Observer(Protocol):
    def __call__(self, mtype: int, mbody: bytes):
        ...


class ReceiveHandler(threading.Thread):
    def __init__(
            self, queue: queue.Queue,
            observers: List[Observer]):
        threading.Thread.__init__(self)
        self.stop_event = threading.Event()
        self.queue = queue
        self.observers = observers

    def stop(self):
        self.stop_event.set()
    
    def run(self):
        while not self.stop_event.is_set():
            if not self.queue.empty():  # non-blocking
                message = self.queue.get()
                self.notify(message)
            time.sleep(0.0001)

    def notify(self, message: bytes):
        mtype, mbody = Message.unpack(message)
        for observer in self.observers:
            observer(mtype, mbody)


class Transceiver:
    
    def __init__(self, config: DragonConfig):
        self.config = config
        self.rank = config.trans.rank
        self.name = f"Node{config.trans.rank:>02}"
        self.logger = Logger.build(self.name, level='INFO')
        self.context = zmq.Context()
        self.init_receiver(port=self.config.trans.tx_port)
        self.init_sender(port=self.config.trans.rx_port)

    def init_receiver(self, port):
        self.rx_socket = self.context.socket(zmq.REP)
        self.rx_socket.bind(f"tcp://localhost:{port}")
        self.rx_socket.setsockopt(zmq.RCVTIMEO, 100) 

        self.receive_queue = queue.Queue(0)
        self.receive_listener = ReceiveListener(self.rx_socket, self.receive_queue)
        self.receive_listener.start()
        self.logger.info("Receiver initialized.")
    
    def init_sender(self, port):
        self.tx_socket = self.context.socket(zmq.REQ)
        self.tx_socket.connect(f"tcp://localhost:{port}")
        self.tx_socket.setsockopt(zmq.SNDTIMEO, 100)
        self.logger.info("Sender initialized.")

    def send(self, mtype: int, mbody: bytes):
        message = Message.pack(mtype, mbody)
        self.tx_socket.send(message)

    def send_with_retry(self, mtype, mbody, max_retry=-1, sleep_time=1):
        attempts = 0
        message = Message.pack(mtype, mbody)
        while max_retry == -1 or attempts < max_retry:
            try:
                self.tx_socket.send(message)
                return True
            except zmq.Again:
                time.sleep(sleep_time)
                attempts += 1
        return False
    
    def register_observers(self, observers: List[Observer]):
        self.receive_handler = ReceiveHandler(self.receive_queue, observers)
        self.receive_handler.start()

    def terminate(self):
        print()
        self.receive_listener.stop()
        self.receive_listener.join()
        self.receive_handler.stop()
        self.receive_handler.join()
        self.rx_socket.close()
        self.tx_socket.close()
        self.context.term()
        self.logger.info("Transceiver stopped.")
