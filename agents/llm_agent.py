from mesa import Agent
import random

class LLMOpinionAgent(Agent):
    def __init__(
        self,
        model,
        knowledge,
        opinion,
        llm,
        fake_news,
        fact,
        a_rep=4,
        b_rep=10,
        low_involv=0.0,
        high_involv=1.0
    ):
        super().__init__(model)
        self.knowledge = knowledge             # Agent's background knowledge level [0.0 - 1.0]
        self.opinion = opinion                 # 1 = believes fake news, -1 = believes fact, 0 = undecided
        self.involvement = 1.0
        self.involvement_threshold = random.uniform(low_involv, high_involv)
        self.has_heard = False
        self.time_since_heard = 0
        self.llm = llm                         # Language model used to evaluate news
        self.news = None
        self.fake_news = fake_news
        self.fact = fact

    def spread_step(self):
        """If the agent is involved and has an opinion, spread it to a neighbor."""
        if not self.should_spread():
            return

        neighbor = self.get_random_neighbor()
        message = self.compose_news_message()
        neighbor.receive_payload(message)

    def update_step(self):
        """Update involvement and opinion state based on heard news."""
        if self.has_heard:
            self.update_involvement()
        self.update_opinion()

    def receive_payload(self, message):
        """Receive a news message, store it if the agent hasn't heard it before."""
        if not self.has_heard:
            self.has_heard = True
            self.time_since_heard = 0
            self.news = message

    def update_opinion(self):
        """Update opinion based on the credibility of the received message."""
        if not self.has_heard:
            return

        if self.involvement == 0.0:
            self.opinion = 0  # Becomes neutral due to lack of involvement
            return

        is_fake = self.is_news_fake()
        self.opinion = 1 if is_fake else -1

    def update_involvement(self):
        """Reduce involvement over time after hearing the news."""
        self.time_since_heard += 1
        decay = 0.1 * self.time_since_heard
        self.involvement = max(0.0, 1.0 - decay)

    def should_spread(self):
        """Determine whether the agent is engaged enough to spread its opinion."""
        return self.involvement >= self.involvement_threshold and self.opinion != 0

    def get_random_neighbor(self):
        """Get a random neighbor (not including self)."""
        return self.random.choice(
            self.model.grid.get_neighbors(self.pos, include_center=False)
        )

    def compose_news_message(self):
        """Use the LLM to generate a piece of fake or factual news."""
        original_news = self.fake_news if self.opinion == 1 else self.fact
        return self.llm.spread_the_news(original_news)

    def is_news_fake(self):
        """Determine whether the current news is fake using LLM, optionally with fact support."""
        if self.knowledge > 0.5:
            return self.llm.check_if_news_is_fake(self.news, fact=self.fact)
        return self.llm.check_if_news_is_fake(self.news)