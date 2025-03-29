from mesa import Agent
import random
import math
import numpy as np

class OpinionAgent(Agent):
    def __init__(self, unique_id, model, knowledge, opinion):
        super().__init__(model)
        self.knowledge = knowledge           # ki ∈ [0, 1]
        self.opinion = opinion               # oi ∈ {-1, 0, 1}
        self.opinion_raw = 0.0               # li (information load)
        self.reputation = np.random.beta(a=4, b=10)  # ri ∈ [0.4, 1.0]
        self.involvement = 1.0               # ci
        self.involvement_threshold = random.uniform(0.0, 1.0)  # cti
        self.has_heard = False               # Controls involvement decay
        self.time_since_heard = 0

    def spread_step(self):
        # Share message if involved
        if self.involvement >= self.involvement_threshold and self.opinion != 0:
            neighbor = self.random.choice(self.model.grid.get_neighbors(self.pos, include_center=False))
            payload = self.opinion * self.reputation
            neighbor.receive_payload(payload)

    def update_step(self):
        if self.has_heard:
            self.time_since_heard += 1
            self.involvement = max(0.0, self.involvement - 0.05 * self.time_since_heard)  # Decrease by 0.05 each step

        # Update opinion based on opinion_raw (li)
        self.update_opinion()

    def receive_payload(self, payload):
        self.opinion_raw += payload
        if not self.has_heard:
            self.has_heard = True
            self.time_since_heard = 0

    def update_opinion(self):
        o_hat = 2 * (1 / (1 + math.exp(-5 * self.opinion_raw))) - 1
        if o_hat > self.knowledge + 0.1:
            self.opinion = 1
        elif o_hat < self.knowledge - 0.1:
            self.opinion = -1
        else:
            self.opinion = 0