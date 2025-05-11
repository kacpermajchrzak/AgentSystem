import solara
from matplotlib import pyplot as plt
from mesa.visualization.utils import update_counter

@solara.component
def HistogramComponent(model, attribute, title, bins=10):
    """
    Generic histogram component for visualizing agent attributes.

    Args:
        model: The model instance.
        attribute: The attribute of agents to visualize (e.g., 'reputation').
        title: Title of the histogram.
        bins: Number of bins for the histogram.
    """
    update_counter.get()
    fig = plt.Figure()
    ax = fig.subplots()
    ax.set_xlim(0, 1)
    values = [getattr(agent, attribute, None) for agent in model.agents]
    ax.hist(values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(attribute.capitalize())
    ax.set_ylabel("Frequency")
    solara.FigureMatplotlib(fig)

@solara.component
def HistogramReputation(model):
    return HistogramComponent(model, "reputation", "Reputation Histogram")

@solara.component
def HistogramKnowledge(model):
    return HistogramComponent(model, "knowledge", "Knowledge Histogram")

@solara.component
def HistogramCommitment(model):
    return HistogramComponent(model, "involvement_threshold", "Commitment Threshold Histogram")