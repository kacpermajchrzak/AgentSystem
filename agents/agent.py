from mesa import Agent
import math
import numpy as np

class OpinionAgent(Agent):
    def __init__(self,model, knowledge, opinion, a_rep=4, b_rep=10, low_involv=0.0, high_involv=1.0):
        super().__init__(model)
        self.knowledge = knowledge           # ki ∈ [0, 1]
        self.opinion = opinion               # oi ∈ {-1, 0, 1}
        self.opinion_raw = 0.0               # li (information load)
        self.reputation = np.random.beta(a=a_rep, b=b_rep)
        self.involvement = 1.0               # ci
        self.involvement_threshold = np.random.uniform(low_involv, high_involv)
        self.has_heard = False               # Controls involvement decay
        self.time_since_heard = 0

    def spread_step(self):
        if self.involvement >= self.involvement_threshold and self.opinion != 0:
            if self.pos is not None:
                neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)
                if neighbors:
                    neighbor_agent_object = self.random.choice(neighbors)
                    payload = self.opinion * self.reputation
                    neighbor_agent_object.receive_payload(payload)

    def update_step(self):
        if self.has_heard:
            self.time_since_heard += 1
            self.involvement = max(0.0, 1.0 - 0.05 * self.time_since_heard)

        self.update_opinion()

    def receive_payload(self, payload):
        self.opinion_raw += payload
        if not self.has_heard:
            self.has_heard = True
            self.time_since_heard = 0

    def update_opinion(self):
        if self.has_heard:
            o_hat = 2 * (1 / (1 + math.exp(-5 * self.opinion_raw))) - 1
            if o_hat > self.knowledge + 0.1:
                self.opinion = 1
            elif o_hat < self.knowledge - 0.1:
                self.opinion = -1
            else:
                self.opinion = 0
            
            if self.involvement == 0.0: 
                self.opinion = 0
    
    def step(self):
        """Defines the agent's actions at each step of the simulation."""
        self.spread_step()
        self.update_step()