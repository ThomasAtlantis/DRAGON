import pickle
import socket
from dragon.utils.configure import Configure, Field as F
import threading


class TransceiverConfig(Configure):
    class trans:
        mode = F(str, default='server', help='Server or client mode')
        host = F(str, default='localhost', help='Host address')
        port = F(int, default=12345, help='Port number')


class Transceiver:
    
    def __init__(self, config: TransceiverConfig):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if config.trans.mode == 'server':    
            self.socket.bind((config.trans.host, config.trans.port))
            self.socket.listen(1)
        else:
            self.socket.connect((config.trans.host, config.trans.port))

    def receive_messages(self):
        while True:
            try:
                data = self.socket.recv(1024).decode('utf-8')
                self.handle_received_data(data)
                if not data:
                    break
                print("Client says:", data)
            except:
                break
        print("Connection closed by peer.")
        self.socket.close()

    def handle_received_data(self, obj):
        print(f'Received data: {obj}')

    def listen(self):
        self.socket.listen(1)
        while True:
            connection, address = self.socket.accept()
            buf = connection.recv(64)
            if len(buf) > 0:
                obj = pickle.loads(buf)
                self.handle_received_data(obj)
            if len(buf) > 0:
                print(obj)

class ClientTransceiver:
    
    def __init__(self, config: TransceiverConfig):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((config.trans.host, config.trans.port))
        self.send({'hello': 'world'})
    
    def send(self, message):
        message = pickle.dumps(message)
        self.socket.sendall(message)
