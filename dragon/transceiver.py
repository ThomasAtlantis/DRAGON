import queue
import time
from dragon.utils.configure import Configure, Field as F
from dragon.utils.mlogging import Logger
import threading
import zmq


class TransceiverConfig(Configure):
    class trans:
        rank = F(int, default=0, help="Index of the transceiver")
        tx_port = F(int, default=5555, help="Port for sending data")
        rx_port = F(int, default=5556, help="Port for receiving data")


class ReceiveThread(threading.Thread):
    def __init__(
            self, socket: zmq.SyncSocket, 
            queue: queue.Queue, 
            event: threading.Event,
            s_buff: int = 1024):
        threading.Thread.__init__(self)
        self.queue = queue
        self.socket = socket
        self.event = event
        self.s_buff = s_buff

    def run(self):
        while not self.event.is_set():
            try:
                data = self.socket.recv_pyobj(self.s_buff)
                self.queue.put(data)
                self.socket.send_pyobj("1")
            except zmq.Again:
                pass


class Transceiver:
    
    def __init__(self, config: TransceiverConfig):
        self.rank = config.trans.rank
        self.name = f"{self.__class__.__name__}{config.trans.rank:>02}"
        self.logger = Logger.build(self.name, level='INFO')

        self.context = zmq.Context()
        self.init_receiver()
        self.init_sender()
        self.send_with_retry(f"Synchronization Test from {self.name}")

    def init_receiver(self):
        self.rx = self.context.socket(zmq.REP)
        self.rx.bind(f"tcp://localhost:{config.trans.tx_port}")
        self.rx.setsockopt(zmq.RCVTIMEO, 100) 

        self.q_receiver = queue.Queue(0)
        self.stop_event = threading.Event()
        self.receive_thread = ReceiveThread(self.rx, self.q_receiver, self.stop_event)
        self.receive_thread.start()
        self.logger.info("Receiver initialized.")
    
    def init_sender(self):
        self.tx = self.context.socket(zmq.REQ)
        self.tx.connect(f"tcp://localhost:{config.trans.rx_port}")
        self.tx.setsockopt(zmq.SNDTIMEO, 100)
        self.logger.info("Sender initialized.")

    def send(self, message):
        self.tx.send_pyobj(message)

    def send_with_retry(self, message):
        while True:
            try:
                self.tx.send_pyobj(message)
                return True
            except zmq.Again:
                time.sleep(1)

    def terminate(self):
        print()
        self.stop_event.set()
        self.receive_thread.join()
        self.rx.close()
        self.tx.close()
        self.context.term()
        self.logger.info("Transceiver stopped.")

    def handle_receive_message(self):
        self.is_running = True
        while self.is_running:
            if self.q_receiver.qsize() > 0:
                msg_params = self.q_receiver.get()
                self.logger.info(msg_params)
            time.sleep(0.0001)

if __name__ == '__main__':
    config = TransceiverConfig()
    config.parse_sys_args()
    transceiver = Transceiver(config)
    try:
        transceiver.handle_receive_message()
    except KeyboardInterrupt:
        transceiver.terminate()
