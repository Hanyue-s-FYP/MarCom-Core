from concurrent import futures
import logging
import os

import grpc
from MarcomCoreServicer import MarcomCoreServicer
from agent import Agent, AgentAttribute
from product import Product
from simulation import Simulation
from dotenv import load_dotenv
from proto import marcom_core_pb2_grpc

from db import *

def main():
    # configure environment variables
    load_dotenv()
    print(f"Env loaded: DB_FILE={os.getenv('DB_FILE')};")

    # initialize the db
    db.connect()
    db.create_tables([AgentInfo, AgentMemory, SimulationEvent])
    print(f"Database initialized")

    # start grpc server
    print("Initialise grpc simulation servicer")
    init_core_servicer()

def init_core_servicer():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    marcom_core_pb2_grpc.add_MarcomServiceServicer_to_server(MarcomCoreServicer(), server)
    server.add_insecure_port(f"{os.getenv('GRPC_CONNECTION_HOST')}:{os.getenv('GRPC_CONNECTION_PORT')}")
    print(f"Connecting to {os.getenv('GRPC_CONNECTION_HOST')}:{os.getenv('GRPC_CONNECTION_PORT')}")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    main()
