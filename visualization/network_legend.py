import solara

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