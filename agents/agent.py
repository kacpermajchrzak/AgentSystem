from mesa import Agent
import math
import numpy as np


class OpinionAgent(Agent):
    def __init__(
        self,
        model,
        knowledge,
        opinion,
        a_rep=4,
        b_rep=10,
        low_involv=0.0,
        high_involv=1.0
    ):
        super().__init__(model)
        
        # Agent properties
        self.knowledge = knowledge          # A float [0.0–1.0] representing how informed the agent is
        self.opinion = opinion              # -1 (against), 0 (neutral), 1 (in favor)
        self.opinion_raw = 0.0              # Aggregated influence from received payloads
        
        # Reputation and involvement
        self.reputation = np.random.beta(a=a_rep, b=b_rep)
        self.involvement = 1.0
        self.involvement_threshold = np.random.uniform(low_involv, high_involv)

        # Message tracking
        self.has_heard = False
        self.time_since_heard = 0

    def step(self):
        """Main function called each simulation step."""
        self.spread_step()
        self.update_step()

    def spread_step(self):
        """Agent attempts to influence a neighbor with their opinion."""
        if not self.should_spread():
            return

        neighbor = self.get_random_neighbor()
        if neighbor:
            payload = self.opinion * self.reputation
            neighbor.receive_payload(payload)

    def update_step(self):
        """Update the internal state of the agent each step."""
        if self.has_heard:
            self.update_involvement()
        self.update_opinion()

    def receive_payload(self, payload):
        """Receive influence from another agent."""
        self.opinion_raw += payload
        if not self.has_heard:
            self.has_heard = True
            self.time_since_heard = 0

    def update_opinion(self):
        """Update the agent's opinion based on accumulated influence."""
        if not self.has_heard:
            return

        o_hat = self.sigmoid_opinion(self.opinion_raw)
        
        if o_hat > self.knowledge + 0.1:
            self.opinion = 1
        elif o_hat < self.knowledge - 0.1:
            self.opinion = -1
        else:
            self.opinion = 0

        # Agents lose opinion if no longer involved
        if self.involvement == 0.0:
            self.opinion = 0

    def update_involvement(self):
        """Decay agent's involvement over time."""
        self.time_since_heard += 1
        decay_rate = 0.05
        self.involvement = max(0.0, 1.0 - decay_rate * self.time_since_heard)

    def should_spread(self):
        """Determine if the agent is both involved and opinionated enough to spread their view."""
        return self.involvement >= self.involvement_threshold and self.opinion != 0

    def get_random_neighbor(self):
        """Return a random neighbor from the agent’s Moore neighborhood, or None."""
        if self.pos is None:
            return None

        neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)
        if not neighbors:
            return None

        return self.random.choice(neighbors)

    @staticmethod
    def sigmoid_opinion(raw_opinion, scale=5.0):
        """Map raw opinion to bounded [-1, 1] using a scaled sigmoid function."""
        return 2 * (1 / (1 + math.exp(-scale * raw_opinion))) - 1