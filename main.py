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
    print(f"Env loaded: DB_FILE={os.getenv('DB_FILE')}; MODEL={os.getenv('MODEL')}")

    # initialize the db
    db.connect()
    db.create_tables([AgentInfo, AgentMemory, SimulationEvent])
    print(f"Database initialized")

    # start grpc server
    print("Initialise grpc simulation servicer")
    init_simulation_servicer()

    testAttrs = [
        AgentAttribute("Priority", "Scoring top marks"),
        AgentAttribute("Subjects", "Focus on core subjects (Math, Science, English)"),
        AgentAttribute("Budget", "Moderate-High (Willing to invest for quality tuition)"),
    ]
    testAttrs2 = [
        AgentAttribute("Priority", "Scoring top marks"),
        AgentAttribute("Subjects", "Focus on core subjects (Math, Science, English)"),
        AgentAttribute("Budget", "Low"),
    ]
    testAgent = Agent(id=1, name="Bob", desc="Highly motivated student prioritizing top marks in PT3/SPM exams.", attrs=testAttrs, simulation_id=1)
    testAgent2 = Agent(id=2, name="Bobby", desc="Highly motivated student prioritizing top marks in PT3/SPM exams. Likes to obtain suggestions from others", attrs=testAttrs2, simulation_id=1)
    testEnv = "The Malaysian education system places high emphasis on standardized national exams like PT3 (Form 3) and SPM (Form 5) as crucial factors in determining university placement and future career opportunities. This creates a high-pressure environment for students and parents in the Klang Valley, a densely populated urban area. Here, a diverse range of tuition centers cater to various student needs and budgets, offering support for these crucial exams. However, intense competition necessitates effective marketing strategies and targeted offerings from tuition centers to attract students and their parents seeking academic success."
    product1 = Product(
        id=1,
        name="Intensive Exam Preparation Course (PT3/SPM)",
        desc="Exam-oriented teaching, Past year paper analysis, Experienced teachers. Subjects include Chinese, Moral",
        price=350.00,  # Assuming a price in MYR
        cost=200.00,  # Assuming a cost for the tuition center
        simulation_id=1
    )
    product2 = Product(
        id=2,
        name="Small Group Subject Tutorials (PT3/SPM)",
        desc="Personalized attention, Addressing individual weaknesses, Targeted practice exercises. Subjects include Maths",
        price=180.00,  # Assuming a price in MYR
        cost=120.00,  # Assuming a cost for the tuition center
        simulation_id=1
    )
    simulation = Simulation(id=1, env_desc=testEnv, agents=[testAgent, testAgent2], products=[product1, product2], total_cycle=5)
    simulation.init_simulation()
    for event in simulation.run_simulation():
        print(event, f"Agent ID: {event.agent.agent_id if event.agent is not None else -1}, Sim ID: {event.sim_id}, Cycle: {event.cycle}, Type: {event.type}, Content: {event.content}, Time: {event.time_created}")

def init_simulation_servicer():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    marcom_core_pb2_grpc.add_MarcomServiceServicer_to_server(MarcomCoreServicer(), server)
    server.add_insecure_port(f"[::]:{os.getenv('GRPC_CONNECTION_PORT')}")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    main()
