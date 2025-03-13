import json
from dragon.retriever.retriever import DPRRetriever
from dragon.utils.mlogging import Logger
# from flask import Flask, request, jsonify
import json
import socket
import struct


class Config:
    class retriever:
        n_docs = 4

HOST, PORT = "localhost", 8765

if __name__ == "__main__":
    # app = Flask(__name__)
    config = Config()
    logger = Logger.build("RaaS", "INFO")
    retriever = DPRRetriever(config, logger)
    retriever.prepare_retrieval(config)
    logger.info("Retriever is ready!")
    # @app.route("/quit", methods=["POST"])
    # def quit(): exit(0)

    # @app.route("/test", methods=["POST"])
    # def test():
    #     queries = request.json.get("queries")
    #     n_docs = request.json.get("n_docs")
    #     return {"queries": queries, "n_docs": n_docs}

    # @app.route("/retrieve", methods=["POST"])
    # def evaluate_endpoint():
    #     queries = request.json.get("queries")
    #     n_docs = request.json.get("n_docs")
    #     retriever.n_docs = n_docs
    #     result = retriever.retrieve_passages(queries)
    #     return jsonify(result)

    # app.run(host=HOST, port=PORT)
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
