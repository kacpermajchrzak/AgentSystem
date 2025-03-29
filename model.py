from mesa import Model
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
import networkx as nx
from agent import OpinionAgent
import numpy as np
import matplotlib.pyplot as plt
from graph_network import generate_graph

class OpinionModel(Model):
    def __init__(self, num_agents=100, avg_degree=3):
        super().__init__()
        self.num_agents = num_agents
        self.G = generate_graph(num_agents)
        self.grid = NetworkGrid(self.G)
        self.datacollector = DataCollector(
            model_reporters={
                "Positive": lambda m: self.count_opinions(1),
                "Negative": lambda m: self.count_opinions(-1),
                "Neutral": lambda m: self.count_opinions(0)
            }
        )
        self.iteration_counter = 0
        node_colors = []
        for i, node in enumerate(self.G.nodes()):
            knowledge = np.random.triangular(left=0, mode=0.05, right=1)
            opinion = 0
            agent = OpinionAgent(self, knowledge, opinion)
            if i == 30:
                agent.opinion = 1
                agent.involvement_threshold = 0.0
                agent.knowledge = 0.0
                agent.has_heard = True
                agent.opinion_raw = 1.0
                node_colors.append('red')
            else:
                node_colors.append('gold')
            self.grid.place_agent(agent, node)
            
        pos = nx.spring_layout(self.G, seed=42)
        nx.draw_networkx_nodes(self.G, pos, node_size=20, node_color=node_colors, alpha=0.8)
        nx.draw_networkx_edges(self.G, pos, width=0.5, edge_color='skyblue', alpha=0.5)
        plt.show()

    def step(self):
        self.iteration_counter += 1
        self.agents.shuffle_do('spread_step')
        self.datacollector.collect(self)
        self.agents.shuffle_do('update_step')

    def count_opinions(self, value):
        return sum([1 for a in self.agents if a.opinion == value])