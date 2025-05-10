from mesa.visualization import SolaraViz, make_plot_component, make_space_component
from model import OpinionModel
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

@solara.component
def NetworkLegend(model):
    """Displays a legend for the network visualization."""
    with solara.Div(style={"padding": "10px", "border": "1px solid #ccc", "margin-top": "10px"}):
        solara.Markdown("### Network Legend")
        with solara.Row(style={"margin-bottom": "5px"}):
            solara.Div(style={"width": "20px", "height": "20px", "background-color": "blue", "margin-right": "10px", "border-radius": "50%"})
            solara.Text("Positive Opinion (P)")
        with solara.Row( style={"margin-bottom": "5px"}):
            solara.Div(style={"width": "20px", "height": "20px", "background-color": "orange", "margin-right": "10px", "border-radius": "50%"})
            solara.Text("Negative Opinion (N)")
        with solara.Row():
            solara.Div(style={"width": "20px", "height": "20px", "background-color": "green", "margin-right": "10px", "border-radius": "50%"})
            solara.Text("Neutral Opinion (Nu)")

def agent_portrayal(agent):
    portrayal = {"size": 15,
                 "text_color": "black"}

    if agent.opinion == 1:
        portrayal["color"] = "blue"
        portrayal["text"] = "P"
    elif agent.opinion == -1:
        portrayal["color"] = "orange"
        portrayal["text"] = "N"
    else:
        portrayal["color"] = "green"
        portrayal["text"] = "Nu"
    return portrayal


model_params = {
    "num_agents": {
        "type": "SliderInt",
        "value": 500,
        "label": "Number of Agents:",
        "min": 100,
        "max": 1000,
        "step": 1,
    },
    "n_communities": {
        "type": "SliderInt",
        "value": 3,
        "label": "Average Degree (m for SBM):",
        "min": 1,
        "max": 10,
        "step": 1,
    },
    "use_llm": {
        "type": "Checkbox",
        "value": False,
        "label": "Use LLM"
    },
    "c": {
        "type": "SliderFloat",
        "value": 0.05,
        "label": "c param for knowledge distribution:",
        "min": 0,
        "max": 1,
        "step": 0.01,
    },
    "a_rep": {
        "type": "SliderInt",
        "value": 4,
        "label": "a param for reputation distribution:",
        "min": 1,
        "max": 10,
        "step": 1,
    },
    "b_rep": {
        "type": "SliderInt",
        "value": 10,
        "label": "b param for reputation distribution:",
        "min": 1,
        "max": 10,
        "step": 1,
    },
    "low_involv": {
        "type": "SliderFloat",
        "value": 0.0,
        "label": "low param for involvement distribution:",
        "min": 0,
        "max": 1,
        "step": 0.01,
    },
    "high_involv": {
        "type": "SliderFloat",
        "value": 1.0,
        "label": "high param for involvement distribution:",
        "min": 0,
        "max": 1,
        "step": 0.01,
    },
}

space_viz = make_space_component(
    agent_portrayal=agent_portrayal,
)

plot_opinions = make_plot_component(
    measure=["Positive", "Negative", "Neutral"]
)


initial_model = OpinionModel(num_agents=model_params["num_agents"]["value"],
                             n_communities=model_params["n_communities"]["value"],
                             use_llm=model_params["use_llm"]["value"])

page = SolaraViz(
    model=initial_model,
    model_params=model_params,
    components=[
        space_viz,
        plot_opinions,
        NetworkLegend,
        HistogramReputation,
        HistogramKnowledge,
        HistogramCommitment,
    ],
    name="Opinion Model Visualization",
)

if __name__ == "__main__":
    print("To run this visualization, save it as a Python file (e.g., server.py)")
    print("Then, in your terminal, run: solara run server.py")