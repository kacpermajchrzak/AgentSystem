# server.py (or your visualization script)

import mesa
print(f"Mesa version: {mesa.__version__}") # Good to check!

# Core imports for the new visualization system
from mesa.visualization import SolaraViz, make_plot_component, make_space_component

# Import your model (assuming it's in model.py)
from model import OpinionModel # Your OpinionModel class

# 1. Agent Portrayal Function (this remains conceptually similar)
# This function tells SolaraViz how to draw each agent.
# It should return a dictionary with keys like "size", "color", "shape", "text", etc.
# The available keys and their effects might differ slightly from the old system,
# so refer to `make_space_component` documentation if needed.
def agent_portrayal(agent):
    portrayal = {"size": 15, # Default size
                 "text_color": "black"}

    if agent.opinion == 1:
        portrayal["color"] = "red"
        portrayal["text"] = "P" # Display 'P' on the agent
    elif agent.opinion == -1:
        portrayal["color"] = "blue"
        portrayal["text"] = "N"
    else:
        portrayal["color"] = "gold" # Gold was used in your Matplotlib viz
        portrayal["text"] = "Nu"
    return portrayal

# 2. Model Parameters for User Input
# These are parameters for your OpinionModel that users can adjust in the UI.
model_params = {
    "num_agents": {
        "type": "SliderInt", # Use "SliderInt", "SliderFloat", "Checkbox", "NumberInput" etc.
        "value": 100,
        "label": "Number of Agents:",
        "min_value": 10, # Note: min_value, max_value for Solara
        "max_value": 200,
        "step": 1,
    },
    "avg_degree": {
        "type": "SliderInt",
        "value": 3,
        "label": "Average Degree (m for SBM):",
        "min_value": 1,
        "max_value": 10,
        "step": 1,
    }
    # Add other parameters of your OpinionModel.__init__ here
    # e.g., if you wanted to make the initial "infected" agent configurable
}

# 3. Create Visualization Components

# Space Component (for your NetworkGrid)
# make_space_component handles drawing agents on the space.
# For a NetworkGrid, it will try to draw the network.
# You might need to pass your self.pos from the model for static layouts,
# or ensure the NetworkGrid is compatible with how make_space_component handles it.
# The `space_drawing_kwargs` can be used for backend-specific drawing options.
# For networkx graphs, ensure your model has 'G' attribute and 'grid' (NetworkGrid).
# The portrayal function is passed here.
# SolaraViz/make_space_component should handle NetworkGrid by trying to draw the graph.
# It uses `model.grid` and the `agent_portrayal`.
# You can use 'space_type': 'network' if it helps disambiguate, or pass drawing_kwargs.
# By default, `make_space_component` will try to infer the space and draw it.
# If your `model.pos` (nx.spring_layout) is used for positions, ensure it's accessible
# or handled correctly by the drawing backend.
# The new system might use different ways to fix node positions than the old NetworkModule.
space_viz = make_space_component(
    agent_portrayal=agent_portrayal,
    # Optional: You might need to specify space_type or provide drawing_kwargs
    # for NetworkX graphs if the default doesn't work as expected.
    # Refer to Mesa's documentation for `make_space_component` with NetworkGrid.
    # E.g., space_drawing_kwargs={"graph_layout_func": lambda G: model_instance.pos} (pseudo-code, needs checking)
)


# Plot Component (for your DataCollector data)
# This creates charts from your model's DataCollector.
# List the model-level reporters you want to plot.
plot_opinions = make_plot_component(
    measure=["Positive", "Negative", "Neutral"]
    # You can customize colors and chart types here if needed,
    # e.g., measurements=[{"label": "Positive", "color": "red"}, ...]
    # Refer to `make_plot_component` documentation for options.
)

# 4. Create and Configure SolaraViz Page
# SolaraViz takes the model class, the list of components, model parameters,
# and other optional arguments.
# Create an initial model instance to pass to SolaraViz.
# This is a slight change from ModularServer which took the class directly for instantiation.
initial_model = OpinionModel(num_agents=model_params["num_agents"]["value"],
                             avg_degree=model_params["avg_degree"]["value"])

page = SolaraViz(
    model=initial_model, # Pass the class for re-instantiation
    model_params=model_params,
    components=[
        space_viz,
        plot_opinions,
        # You can add custom text components too:
        # lambda m: f"Step: {m.iteration_counter} | Positive: {m.datacollector.get_model_vars_dataframe()['Positive'].iloc[-1] if m.iteration_counter > 0 else 0}"
    ],
    name="Opinion Model Visualization",
    # agent_view=True, # If you want to inspect individual agents
    # play_interval=500, # Milliseconds between steps when playing
)

# 5. To run this (typically in a separate file like `run.py` or at the end of this script):
# This setup is for running with Solara. You'd typically run `solara run your_script_name.py`
# If you are in a Jupyter environment, just having `page` as the last line might render it.

if __name__ == "__main__":
    # This part is tricky. Solara apps are usually launched with `solara run <filename>.py`.
    # To make it runnable with `python <filename>.py`, you might need to embed it
    # differently or instruct the user to use the Solara CLI.
    # For direct execution, one might try to use solara.server.starlette.Server but it's more involved.
    # The simplest for development is `solara run your_script_name.py`
    print("To run this visualization, save it as a Python file (e.g., server.py)")
    print("Then, in your terminal, run: solara run server.py")
    # In a Jupyter Notebook, displaying 'page' usually works.
    # If not in Jupyter, the below won't directly launch the server correctly without the Solara CLI.
    # page # This line is for Jupyter auto-rendering.