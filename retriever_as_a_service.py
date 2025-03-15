import json
import socket
import struct
from dragon.retriever import DPRRetriever
from dragon.utils.mlogging import Logger


class Config:
    class retriever:
        n_docs = 4

HOST, PORT = "localhost", 8765

if __name__ == "__main__":
    config = Config()
    logger = Logger.build("RaaS", "INFO")
    retriever = DPRRetriever(config)
    retriever.prepare_retrieval(config)
    logger.info("Retriever is ready!")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(1)
    protocol = struct.Struct("I")
    header_size = struct.calcsize("I")
    while True:
        conn, addr = server.accept()
        try:
            header = b''
            while len(header) < header_size:
                chunk = conn.recv(header_size - len(header))
                header += chunk
            body_len = protocol.unpack(header)[0]
            mbody = b''
            while len(mbody) < body_len:
                chunk = conn.recv(body_len - len(mbody))
                mbody += chunk
            data = json.loads(mbody.decode())
            queries = data.get("queries")
            n_docs = data.get("n_docs")
            retriever.n_docs = n_docs
            result = retriever.retrieve_passages(queries)
            mbody = json.dumps(result).encode()
            conn.send(protocol.pack(len(mbody)) + mbody)
        except Exception as e:
            conn.close()
