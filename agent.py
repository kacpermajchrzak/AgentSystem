from mesa import Agent
import random
import math
import numpy as np

class OpinionAgent(Agent):
    def __init__(self,model, knowledge, opinion): # Added unique_id, model is now second
        super().__init__(model) # Pass unique_id and model to super
        self.knowledge = knowledge           # ki ∈ [0, 1]
        self.opinion = opinion               # oi ∈ {-1, 0, 1}
        self.opinion_raw = 0.0               # li (information load)
        # Ensure self.pos is available for get_neighbors. It's set when agent is placed.
        # self.reputation = np.random.beta(a=4, b=10) # Original
        # To ensure reproducibility if model seed is used, pass model.random
        self.reputation = self.model.random.beta(a=4, b=10) if hasattr(self.model, 'random') and hasattr(self.model.random, 'beta') else np.random.beta(a=4, b=10)
        self.involvement = 1.0               # ci
        # self.involvement_threshold = np.random.beta(a=2, b=10) # Original
        self.involvement_threshold = self.model.random.beta(a=2, b=10) if hasattr(self.model, 'random') and hasattr(self.model.random, 'beta') else np.random.beta(a=2, b=10)
        self.has_heard = False               # Controls involvement decay
        self.time_since_heard = 0

    def spread_step(self):
        # Share message if involved
        if self.involvement >= self.involvement_threshold and self.opinion != 0:
            # Get neighbors from the grid using agent's position (self.pos)
            # self.pos is automatically set by model.grid.place_agent(agent, node_id)
            if self.pos is not None:
                neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)
                if neighbors: # Check if there are any neighbors
                    neighbor_agent_object = self.random.choice(neighbors) # This returns an agent object
                    payload = self.opinion * self.reputation
                    neighbor_agent_object.receive_payload(payload) # Call receive_payload on the agent object

    def update_step(self):
        if self.has_heard:
            self.time_since_heard += 1
            self.involvement = max(0.0, 1.0 - 0.05 * self.time_since_heard)

        self.update_opinion()

    def receive_payload(self, payload):
        self.opinion_raw += payload
        if not self.has_heard:
            self.has_heard = True
            self.time_since_heard = 0 # Reset time since heard

    def update_opinion(self):
        if self.has_heard: # Only update if agent has heard something
            o_hat = 2 * (1 / (1 + math.exp(-5 * self.opinion_raw))) - 1
            # Using a small epsilon for floating point comparisons might be safer if knowledge can be very close to 0.1
            # For now, direct comparison as per original logic.
            if o_hat > self.knowledge + 0.1:
                self.opinion = 1
            elif o_hat < self.knowledge - 0.1: # Corrected from self.knowledge - 0.1 to -self.knowledge + 0.1 if that was intended, but original seems fine.
                self.opinion = -1
            else:
                self.opinion = 0
            
            # If involvement drops to zero, opinion becomes neutral
            if self.involvement == 0.0: # Consider using <= 0.0 for safety with float precision
                self.opinion = 0
    
    def step(self):
        """Defines the agent's actions at each step of the simulation."""
        # The order of spread and update might matter depending on desired model dynamics
        self.spread_step()
        self.update_step()