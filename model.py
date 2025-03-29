from mesa import Model
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
import time
import networkx as nx
from agent import OpinionAgent
import numpy as np

class OpinionModel(Model):
    def __init__(self, num_agents=100, avg_degree=3):
        super().__init__()
        self.num_agents = num_agents
        self.G = nx.watts_strogatz_graph(n=num_agents, k=avg_degree, p=0.1)
        self.grid = NetworkGrid(self.G)
        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(
            model_reporters={
                "Positive": lambda m: self.count_opinions(1),
                "Negative": lambda m: self.count_opinions(-1),
                "Neutral": lambda m: self.count_opinions(0)
            }
        )

        for i, node in enumerate(self.G.nodes()):
            knowledge = np.clip(np.random.normal(loc=0.5, scale=0.2), 0, 1)
            opinion = 0
            agent = OpinionAgent(i, self, knowledge, opinion)

            # Initialize one or two agents with strong opinion
            if i < 2:
                agent.opinion = 1
                agent.has_heard = True

            self.schedule.add(agent)
            self.grid.place_agent(agent, node)

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)

    def count_opinions(self, value):
        return sum([1 for a in self.schedule.agents if a.opinion == value])