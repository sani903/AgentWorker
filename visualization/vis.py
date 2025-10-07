import dash
import json

import plotly.express as px
import pandas as pd

from dash import dcc, html, Output, Input

df = pd.read_csv("taxonomy_analysis.csv")
df["NumTasks_logsafe"] = df["NumTasks"].replace(0, 5e-1)
taxonomy = json.load(open("../taxonomy_with_similarity_updated.json"))

app = dash.Dash(__name__)

fig = px.scatter(df, x="NumJobs", y="NumTasks_logsafe", size="Population", color="BaseTaskName", hover_name="BaseTaskName", custom_data="TaxonomyPath", log_y=True, size_max=60)

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
    # If nothing clicked yet
    if not clickData:
        return "Click on a bubble to see details."

    # Extract and traverse the taxonomy path
    path_str = clickData["points"][0]["customdata"][0]
    path_parts = [p.strip() for p in path_str.split(" -> ") if p.strip()]
    node = taxonomy
    try:
        for part in path_parts:
            node = node[part]
    except (KeyError, TypeError):
        return f"Path not found in taxonomy: {path_str}"

    # Extract expected lists
    jobs = node.get("jobs", []) if isinstance(node, dict) else []
    tasks = node.get("tasks", []) if isinstance(node, dict) else []

    # Index jobs by job_task_id for quick lookups when enriching task job_similarities
    job_index = {}
    for j in jobs:
        # job_task_id might be str or int in JSON; normalize to str
        key = str(j.get("job_task_id"))
        if key:
            job_index[key] = j

    def format_scalar(val):
        if isinstance(val, (int, float)):
            return str(val)
        if val is None:
            return "—"
        return str(val)

    # Helper to compute indentation style per nesting level
    def indent_style(level, base_margin_bottom=8):
        return {
            "marginLeft": f"{level * 14}px" if level > 0 else "0px",
            "marginBottom": f"{base_margin_bottom}px"
        }

    # Build job blocks (simple key/value listing)
    job_blocks = []
    for i, job in enumerate(jobs):
        header = f"{job.get('job_task_id', i+1)}: {job.get('occupation', 'Occupation')}"
        kv_items = []
        for k, v in job.items():
            kv_items.append(html.Li([
                html.Strong(f"{k}: "),
                html.Span(format_scalar(v))
            ]))
        job_blocks.append(
            html.Details([
                html.Summary(header),
                html.Ul(kv_items, style={"marginTop": "4px", "marginBottom": "4px"})
            ], open=False, style=indent_style(1))
        )

    jobs_section = html.Details([
        html.Summary(f"Jobs ({len(jobs)})"),
        html.Div(job_blocks) if job_blocks else html.Em("No jobs listed.")
    ], open=False, style=indent_style(0, 18))

    # Build task blocks with nested job_similarities enriched with job data
    task_blocks = []
    for i, task in enumerate(tasks):
        task_id = task.get("task_id") or f"Task {i+1}"
        benchmark = task.get("benchmark", "")
        summary_label = f"{task_id}{' • ' + benchmark if benchmark else ''}"

        # Instruction (truncate long text in summary view only)
        instruction = task.get("instruction", "")
        truncated = (instruction[:140] + "…") if len(instruction) > 140 else instruction

        # job_similarities list
        sims = task.get("job_similarities", []) or []
        sim_blocks = []
        for s_i, sim in enumerate(sims):
            jt_id = str(sim.get("job_task_id"))
            matched_job = job_index.get(jt_id)
            score = sim.get("score")
            reasoning = sim.get("reasoning")

            # Combine similarity + job fields
            detail_items = []
            if score is not None:
                detail_items.append(html.Li([html.Strong("score: "), html.Span(format_scalar(score))]))
            if reasoning:
                detail_items.append(html.Li([html.Strong("reasoning: "), html.Span(reasoning)]))
            if matched_job:
                # Add a nested UL for job metadata
                job_meta_items = []
                for k, v in matched_job.items():
                    job_meta_items.append(html.Li([html.Strong(f"{k}: "), html.Span(format_scalar(v))]))
                detail_items.append(html.Li([
                    html.Details([
                        html.Summary("Job metadata"),
                        html.Ul(job_meta_items, style={"marginTop": "4px"})
                    ], open=False, style=indent_style(4))
                ]))
            sim_blocks.append(
                html.Details([
                    html.Summary(f"Similarity {s_i+1} • job_task_id={jt_id}"),
                    html.Ul(detail_items, style={"marginTop": "4px"})
                ], open=False, style=indent_style(3, 6))
            )

        # Key/values for the task itself (excluding big fields handled separately)
        task_kv = []
        for k, v in task.items():
            if k in {"instruction", "job_similarities"}:
                continue
            task_kv.append(html.Li([html.Strong(f"{k}: "), html.Span(format_scalar(v))]))

        # Assemble task details
        task_blocks.append(
            html.Details([
                html.Summary(summary_label),
                html.Div([
                    html.Details([
                        html.Summary("Instruction"),
                        html.Pre(instruction, style={"whiteSpace": "pre-wrap", "fontSize": "12px", "margin": 0})
                    ], open=False, style=indent_style(2, 6)),
                    html.Details([
                        html.Summary(f"Job similarities ({len(sim_blocks)})"),
                        html.Div(sim_blocks) if sim_blocks else html.Em("No job similarities provided")
                    ], open=False, style=indent_style(2, 8)),
                    html.Details([
                        html.Summary("Metadata"),
                        html.Ul(task_kv, style={"marginTop": "4px"}) if task_kv else html.Em("No additional metadata")
                    ], open=False, style=indent_style(2))
                ], style=indent_style(1, 0))
            ], open=False, style=indent_style(1, 10))
        )

    tasks_section = html.Details([
        html.Summary(f"Tasks ({len(tasks)})"),
        html.Div(task_blocks, style=indent_style(0, 0)) if task_blocks else html.Em("No tasks listed.")
    ], open=False, style=indent_style(0, 18))

    # Fallback note
    note = None
    if not jobs and not tasks:
        note = html.Div("No 'jobs' or 'tasks' found at this node.", style={"color": "#aa0000"})

    return html.Div([
        html.H4("Details"),
        html.Div([
            html.Strong("Path: "),
            html.Span(" / ".join(path_parts))
        ], style={"marginBottom": "12px"}),
        tasks_section,
        jobs_section,
        note
    ], style={"fontFamily": "Arial, sans-serif", "fontSize": "14px"})


if __name__ == "__main__":
    app.run(debug=True)
