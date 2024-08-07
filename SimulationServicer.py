from agent import Agent, AgentAttribute
from product import Product
import proto_simulation.simulation_pb2_grpc as simulation_pb2_grpc
import proto_simulation.simulation_pb2 as simulation_pb2
from simulation import Simulation

current_simulations = [] # when start simulation, create and put in this array, so when try to stream updates can get from here

class SimulationServicer(simulation_pb2_grpc.SimulationServiceServicer):
    def StartSimulation(self, request, context):
        # build agents from grpc request
        agents = []
        for agent in request.agents:
            agent_attrs = []
            for attr in agent.attrs:
                agent_attrs.append(AgentAttribute(key=attr.key, value=attr.value))
            agents.append(Agent(id=agent.id, name=agent.name, desc=agent.desc, attrs=agent_attrs, simulation_id=request.id))
        # build products from grpc request
        products = []
        for product in request.products:
            products.append(Product(id=product.id, name=product.name, desc=product.desc, price=product.price, cost=product.cost, simulation_id=request.id))
        sim = Simulation(id=request.id, env_desc=request.env_desc, agents=agents, products=products, total_cycle=request.total_cycles)
        current_simulations.append(sim)
        return simulation_pb2.SimulationResponse(message="Simulation created, calling stream to initialise and run simulation")

    def PauseSimulation(self, request, context):
        print(request)

    def StreamUpdates(self, request, context):
        for i in range(6):
            yield simulation_pb2.SimulationUpdate(agent_id=1, action="BUY", content="1:asd", cycle=i, simulation_id=request.simulation_id)