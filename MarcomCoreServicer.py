from typing import Generator
from agent import Agent, AgentAttribute
from db import SimulationEvent
from product import Product
from proto import marcom_core_pb2, marcom_core_pb2_grpc
from researcher import (
    do_web_search,
    get_product_comp_report,
    reconstruct_query_with_product,
)
from simulation import Simulation


class MarcomCoreServicer(marcom_core_pb2_grpc.MarcomServiceServicer):
    current_simulations: list[Simulation] = (
        []
    )  # when start simulation, create and put in this array, so when try to stream updates can get from here
    # stores the generator for pause capabilities pair id -> the generator
    simulation_generators: dict[int, Generator[SimulationEvent, None, None]] = {}

    def StartSimulation(self, request, context):
        print(request)
        # check if the simulation has already been created (frontend limits that completed simulation wont be able to hit the run button)
        in_curr_sim = [
            sim for sim in self.current_simulations if int(sim.id) == int(request.id)
        ]
        if len(in_curr_sim) > 0:  # should only be 1 though
            # resume the simulation, since is reference, can direct make changes
            in_curr_sim[0].resume_simulation()
            return marcom_core_pb2.SimulationResponse(
                message="Simulation exist in core, calling stream to listen to updates"
            )  # give the client such message, but backend is the one responsible for calling stream
        else:
            # build agents from grpc request
            agents = []
            for agent in request.agents:
                agent_attrs = []
                for attr in agent.attrs:
                    agent_attrs.append(AgentAttribute(key=attr.key, value=attr.value))
                agents.append(
                    Agent(
                        id=agent.id,
                        name=agent.name,
                        desc=agent.desc,
                        attrs=agent_attrs,
                        simulation_id=request.id,
                    )
                )
            # build products from grpc request
            products = []
            for product in request.products:
                products.append(
                    Product(
                        id=product.id,
                        name=product.name,
                        desc=product.desc,
                        price=product.price,
                        cost=product.cost,
                        simulation_id=request.id,
                    )
                )
            sim = Simulation(
                id=int(request.id),
                env_desc=request.env_desc,
                agents=agents,
                products=products,
                total_cycle=request.total_cycles,
            )
            self.current_simulations.append(sim)
            self.simulation_generators[sim.id] = (
                sim.run_simulation()
            )  # create the generator
            return marcom_core_pb2.SimulationResponse(
                message="Simulation added, calling stream to initialise and run simulation"
            )

    def PauseSimulation(self, request, context):
        in_curr_sim = [
            sim
            for sim in self.current_simulations
            if int(sim.id) == int(request.simulation_id)
        ]
        if len(in_curr_sim) > 0:  # should only be 1 though
            # resume the simulation, since is reference, can direct make changes
            in_curr_sim[0].pause_simulation()
            return marcom_core_pb2.PauseResponse(
                message="Pausing the simulation gracefully..."
            )  # give the client such message, but backend is the one responsible for calling stream
        return marcom_core_pb2.PauseResponse(
            message="No such simulation in the system, is StartSimulation called?"
        )

    def StreamSimulationUpdates(self, request, context):
        in_curr_sim = [
            sim
            for sim in self.current_simulations
            if int(sim.id) == int(request.simulation_id)
        ]
        if len(in_curr_sim) <= 0:
            # simulation does not exist or is completed (completed simulations will be removed), directly end ba
            return
        # get the specific generator
        gen = self.simulation_generators[int(request.simulation_id)]
        while not in_curr_sim[0].paused:
            try:
                sim_event = next(gen)
                yield marcom_core_pb2.SimulationUpdate(
                    agent_id=(
                        sim_event.agent.agent_id if sim_event.agent is not None else 0
                    ),
                    action=sim_event.type,
                    content=sim_event.content,
                    cycle=sim_event.cycle,
                    simulation_id=sim_event.sim_id,
                )
            except StopIteration:
                # the simulation ended, remove it from the list
                current_simulations = [
                    sim
                    for sim in current_simulations
                    if int(sim.id) != int(request.simulation_id)
                ]
                del self.simulation_generators[int(request.simulation_id)]
                break

    def ResearchProductCompetitor(self, request, context):
        p = Product(
            id=int(request.id),
            name=request.name,
            desc=request.desc,
            price=float(request.price),
            cost=float(request.cost),
            simulation_id=0,
        )  # not associated to a specific simulation
        reconstructed_query = reconstruct_query_with_product(p)
        search_results = do_web_search(reconstructed_query["query"])
        report = get_product_comp_report(
            p, reconstructed_query["query"], search_results
        )
        return marcom_core_pb2.ProductCompetitorResponse(
            query=reconstructed_query["query"], report=report
        )
