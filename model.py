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
        self.G = generate_graph(num_agents, m_=avg_degree)
        self.grid = NetworkGrid(self.G)
        self.datacollector = DataCollector(
            model_reporters={
                "Positive": lambda m: self.count_opinions(1),
                "Negative": lambda m: self.count_opinions(-1),
                "Neutral": lambda m: self.count_opinions(0)
            }
        )
        self.iteration_counter = 0
        self.pos = nx.spring_layout(self.G, seed=42)

        for i, node in enumerate(self.G.nodes()):
            knowledge = np.random.triangular(left=0, mode=0.05, right=1)
            opinion = 0
            
            agent = OpinionAgent(self, knowledge, opinion)

            if i == 30:
                agent.opinion = 1
                if hasattr(agent, 'involvement_threshold'):
                    agent.involvement_threshold = 0.0
                agent.knowledge = 0.0
                if hasattr(agent, 'has_heard'):
                    agent.has_heard = True
                if hasattr(agent, 'opinion_raw'):
                    agent.opinion_raw = 1.0
            
            self.grid.place_agent(agent, node)

    def visualize_network_opinions(self, step_number):
        plt.figure(figsize=(12, 10))
        node_colors = []
        
        for node_id in self.G.nodes():
            cell_contents = self.grid.get_cell_list_contents([node_id])
            if cell_contents:
                agent = cell_contents[0]
                if agent.opinion == 1:
                    node_colors.append('red')
                elif agent.opinion == -1:
                    node_colors.append('blue')
                else:
                    node_colors.append('gold')
            else:
                node_colors.append('grey')

        nx.draw_networkx_nodes(self.G, self.pos, node_size=50, node_color=node_colors, alpha=0.9)
        nx.draw_networkx_edges(self.G, self.pos, width=0.5, edge_color='lightgrey', alpha=0.7)
        plt.title(f"Network Opinions at Step {step_number}", fontsize=16)
        plt.xlabel("X-coordinate")
        plt.ylabel("Y-coordinate")
        plt.show()

    def step(self):
        self.iteration_counter += 1
        self.agents.shuffle_do('spread_step')
        self.datacollector.collect(self)
        self.agents.shuffle_do('update_step')
        self.visualize_network_opinions(self.iteration_counter)

    def count_opinions(self, opinion_value):
        return sum([1 for a in self.agents if a.opinion == opinion_value])