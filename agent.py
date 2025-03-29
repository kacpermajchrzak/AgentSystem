from mesa import Agent
import random
import math
import numpy as np

class OpinionAgent(Agent):
    def __init__(self, model, knowledge, opinion, llm, fake_news, fact):
        super().__init__(model)
        self.knowledge = knowledge           # ki ∈ [0, 1]
        self.opinion = opinion               # oi ∈ {-1, 0, 1}
        self.opinion_raw = 0.0               # li (information load)
        self.reputation = np.random.beta(a=4, b=10)  # ri ∈ [0.4, 1.0]
        self.involvement = 1.0               # ci
        self.involvement_threshold = random.uniform(0.0, 1.0)  # cti
        self.has_heard = False               # Controls involvement decay
        self.time_since_heard = 0
        self.llm = llm
        self.news = None
        self.fake_news = fake_news
        self.fact = fact

    def spread_step(self):
        # Share message if involved
        if self.involvement >= self.involvement_threshold and self.opinion != 0:
            neighbor = self.random.choice(self.model.grid.get_neighbors(self.pos, include_center=False))
            payload = self.opinion * self.reputation
            model_message = ""
            if self.opinion == 1:
                model_message = self.llm.spread_the_news(self.fake_news)
            else:
                model_message = self.llm.spread_the_news(self.fact)

            neighbor.receive_payload(model_message)

    def update_step(self):
        if self.has_heard:
            self.time_since_heard += 1
            self.involvement = max(0.0, 1.0 - 0.1 * self.time_since_heard)

        self.update_opinion()

    def receive_payload(self, message):
        if not self.has_heard:
            self.has_heard = True
            self.time_since_heard = 0
            self.news = message

    def update_opinion(self):
        if self.has_heard:
            if self.involvement == 0.0:
                self.opinion = 0
                return
            
            if self.knowledge > 0.5:
                model_decision = self.llm.check_if_news_is_fake (self.news, fact=self.fact)
            else:
                model_decision = self.llm.check_if_news_is_fake (self.news)

            if model_decision:
                self.opinion = 1
            else:
                self.opinion = -1