from mesa import Model
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from agents.agent import OpinionAgent
from agents.llm_agent import LLMOpinionAgent
from environment.community_graph import generate_graph
from agents.utils.llm import LLM


class NetworkModel(Model):
    def __init__(
        self,
        num_agents=100,
        n_communities=3,
        use_llm=False,
        c=0.05,
        a_rep=4,
        b_rep=10,
        low_involv=0.0,
        high_involv=1.0
    ):
        super().__init__()

        # Parameters
        self.num_agents = num_agents
        self.use_llm = use_llm
        self.llm = LLM() if use_llm else None
        self.fake_news = "There was an earthquake in Asia this month."
        self.fact = "There was no earthquake in Asia this month."

        # Network and grid
        self.G = generate_graph(self.num_agents, m_=n_communities)
        self.grid = NetworkGrid(self.G)
        self.pos = nx.spring_layout(self.G, seed=42)

        # Step counter
        self.iteration_counter = 0

        # Data collection
        self.datacollector = DataCollector(model_reporters={
            "Positive": lambda m: self.count_opinions(1),
            "Negative": lambda m: self.count_opinions(-1),
            "Neutral": lambda m: self.count_opinions(0),
        })

        # Initialize agents
        self._initialize_agents(a_rep, b_rep, low_involv, high_involv, c)

    def _initialize_agents(self, a_rep, b_rep, low_involv, high_involv, c):
        """Create and place agents on the network grid."""
        for i, node in enumerate(self.G.nodes()):
            knowledge = np.random.triangular(left=0, mode=c, right=1)
            opinion = 0

            if self.use_llm:
                agent = LLMOpinionAgent(
                    model=self,
                    knowledge=knowledge,
                    opinion=opinion,
                    llm=self.llm,
                    fake_news=self.fake_news,
                    fact=self.fact
                )
            else:
                agent = OpinionAgent(
                    model=self,
                    knowledge=knowledge,
                    opinion=opinion,
                    a_rep=a_rep,
                    b_rep=b_rep,
                    low_involv=low_involv,
                    high_involv=high_involv
                )

            # Seed fake news in a specific agent (e.g., node index 30)
            if i == 30:
                self._seed_fake_news(agent)

            self.grid.place_agent(agent, node)

    def _seed_fake_news(self, agent):
        """Initialize an agent to spread fake news strongly."""
        agent.opinion = 1
        agent.involvement_threshold = 0.0
        agent.knowledge = 0.0
        agent.has_heard = True
        agent.opinion_raw = 1.0
        agent.news = self.fake_news

    def step(self):
        """Advance the model by one step."""
        self.iteration_counter += 1

        self.agents.shuffle_do("spread_step")
        self.datacollector.collect(self)
        self.agents.shuffle_do("update_step")

        self.visualize_network_opinions(self.iteration_counter)

    def count_opinions(self, opinion_value):
        """Count the number of agents holding a specific opinion."""
        return sum(1 for agent in self.agents if agent.opinion == opinion_value)

    def visualize_network_opinions(self, step_number):
        """Draw the current state of the network with agent opinions."""
        plt.figure(figsize=(12, 10))

        node_colors = [
            self._get_node_color(node)
            for node in self.G.nodes()
        ]

        nx.draw_networkx_nodes(
            self.G, self.pos, node_size=50,
            node_color=node_colors, alpha=0.9
        )
        nx.draw_networkx_edges(
            self.G, self.pos, width=0.5,
            edge_color="lightgrey", alpha=0.7
        )

        plt.title(f"Network Opinions at Step {step_number}", fontsize=16)
        plt.xlabel("X-coordinate")
        plt.ylabel("Y-coordinate")
        plt.show()

    def _get_node_color(self, node_id):
        """Return color representing agent opinion for a node."""
        cell_contents = self.grid.get_cell_list_contents([node_id])
        if cell_contents:
            agent = cell_contents[0]
            if agent.opinion == 1:
                return "red"
            elif agent.opinion == -1:
                return "blue"
            else:
                return "gold"
        return "grey"