import dash

import plotly.express as px
import pandas as pd

from dash import dcc, html, Output, Input

df = pd.read_csv("taxonomy_analysis.csv")
df["NumTasks_logsafe"] = df["NumTasks"].replace(0, 5e-1)


app = dash.Dash(__name__)

fig = px.scatter(df, x="NumJobs", y="NumTasks_logsafe", size="Population", color="BaseTaskName", hover_name="BaseTaskName", log_y=True, size_max=60)

fig.update_layout(height=1200)

app.layout = html.Div([
    dcc.Graph(id="bubble-plot", figure=fig, style={"height": "1200px"}),
    html.Div(id="click-output")
])

@app.callback(
    Output("click-output", "children"),
    Input("bubble-plot", "clickData")
)
def display_click(clickData):
    if clickData:
        city = clickData["points"][0]["hovertext"]
        return f"You clicked on {city}!"
    return "Click on a bubble to see details."

if __name__ == "__main__":
    app.run(debug=True)
