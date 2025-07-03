import os
from collections import deque
import pathlib
import logging
import datetime

import dash
from dash import Input, Output, State, html, dcc
from dash_canvas.utils import array_to_data_url
import plotly.graph_objs as go
import dash_mantine_components as dmc
from dash_extensions import EventListener
import pandas as pd
import cv2
import numpy as np
import base64
from io import BytesIO

import torch
from torchvision import transforms
from torch.serialization import add_safe_globals
from PIL import Image
from torch._dynamo.eval_frame import OptimizedModule

from visualize import get_level, get_conf

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    # suppress_callback_exceptions=True
)
app.title = "Oilsands Level Tracker"
server = app.server

APP_PATH = str(pathlib.Path(__file__).parent.resolve())
df = pd.read_csv(os.path.join(APP_PATH, os.path.join("data", "spc_data.csv")))
# TODO : add confidence level to the params ie the csv
params = list(df)

max_length = len(df)

suffix_row = "_row"
suffix_button_id = "_button"
suffix_sparkline_graph = "_sparkline_graph"
suffix_count = "_count"
suffix_ooc_n = "_OOC_number"
suffix_ooc_g = "_OOC_graph"
suffix_indicator = "_indicator"


model = None
device = "cuda"
transform = transforms.Compose(
            [
        transforms.Resize((720, 1280)),
        transforms.ToTensor(),
            ]
)
# TODO: move to persistent store
level_queue = deque(maxlen=100)
qual_queue = deque(maxlen=100)
date_queue = deque(maxlen=100)

logger = logging.getLogger(__name__)


def get_next_img(idx):
    cap = cv2.VideoCapture("assets/crusher.mp4")
    tot_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.set(cv2.CAP_PROP_POS_FRAMES, (idx * 1) % tot_frames + 1) # this sets the frame position to the next frame based on the interval
    ret, frame = cap.read()
    if ret:
        url = array_to_data_url(np.asarray(frame))

    else:
        url = ""

    cap.release()
    return url
def load_model_if_not_loaded():
    global model # Declare intent to modify global 'model'
    if model is None:
        print("Loading model for the first time...")
        add_safe_globals([OptimizedModule]) # Keep this where torch.load is used
        model_path = "../best_mdl_wts.pt" # Or pass it in
        model = torch.load(model_path, weights_only=False)
        model = model.to(device)
        model.eval()

        print("Model loaded.")
    return model, transform # Return model and transform

# ##################### UI Elements #####################

def build_banner():
    """
    Build the banner UI component.
    """
    return html.Div(
        id="banner",
        className="banner",
        children=[
            html.Div(
                id="banner-text",
                children=[
                    html.H5("Oil Sand Monitoring Dashboard"),
                    html.H6("Process Control and Exception Reporting"),
                ],
            ),
        ],
    )


def build_tabs():
    """
    Build the tabs UI component.
    """
    return html.Div(
        id="tabs",
        className="tabs",
        children=[
            dcc.Tabs(
                id="app-tabs",
                value="tab2",
                className="custom-tabs",
                children=[
                    dcc.Tab(
                        id="Control-chart-tab",
                        label="Control Charts Dashboard",
                        value="tab2",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                ],
            )
        ],
    )


def build_bar_figure(values=None):
    """
    Build the bar chart figure for the dashboard.
    """
    labels = ["Oil Sand Level", "Confidence Level"]
    values = [45, 88] if not values else values
    error_y_data = [None,None]
    error_data = values[2]

    error_y_data[0] = dict(
            type='data',
            array=error_data[0],      # Error value for the top side
            arrayminus=error_data[1], # Error value for the bottom side
            visible=True,
            color='red',            # Color of the error bar
            thickness=1.5,            # Thickness of the error bar line
            width=3                   # Width of the caps on the error bar
        )
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                text=[f"{v:.1f}%" for v in values],
                textposition="outside",
                textfont=dict(
                    color="white",
                    size=16
                ),
                marker_color=['#1f77b4', '#ff7f0e'],
                error_y= error_y_data[0] # Apply error_y to the first bar (Level)
            )
        ]
    )

    fig.update_layout(
        yaxis=dict(
            range=[0, 100],
            ticksuffix="%"
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        autosize=True,
        template="plotly_dark",
        font=dict(size=18),
    )
    return fig



def build_barchart_panel():
    """
    Build the bar chart panel for the dashboard.
    """
    return html.Div(
        id="bar-chart-container",
        children=[dcc.Graph(id="bar-chart-live-graph", figure=build_bar_figure())],
    )

def build_chart_panel():
    """
    builds the control chart panel with two line charts for Level and Confidence.
    """
    return html.Div(
        id="control-chart-container",
        className="twelve columns",
        children=[
            generate_section_banner("Live SPC Chart"),
            dcc.Graph(
                id="control-chart-live",
                figure=go.Figure(
                    {
                        "data": [
                            # {
                            #     "x": [],
                            #     "y": [],
                            #     "mode": "lines+markers",
                            #     "name": params[1],
                            # }
                            {
                                "x": [],
                                "y": [],
                                "mode": "lines+markers",
                                "name": "Level",  # First line chart for Level
                                "line": {"color": "#1f77b4"} 
                            },
                            {
                                "x": [],
                                "y": [],
                                "mode": "lines+markers",
                                "name": "Confidence", # Second line chart for Confidence
                                "line": {"color": "#ff7f0e"} 
                            },
                        ],
                        "layout": {
                            "paper_bgcolor": "rgba(0,0,0,0)",
                            "plot_bgcolor": "rgba(0,0,0,0)",
                            "xaxis": dict(
                                showline=False, showgrid=False, zeroline=False
                            ),
                            "yaxis": dict(
                                showgrid=False, showline=False, zeroline=False, ticksuffix="%"
                            ),
                            "autosize": True,
                        },
                    }
                ),
            ),
        ],
    )


def generate_section_banner(title):
    return html.Div(className="section-banner", children=title)


def build_top_panel(stopped_interval):
    """
    Build the top panel of the dashboard, which includes a metrics summary section.
    """
    return html.Div(
        id="top-section-container",
        className="row",
        children=[
            # Metrics summary
            html.Div(
                id="metric-summary-session",
                className="eight columns",
                children=[
                    generate_section_banner("Process Control Metrics Summary"),
                ],
            ),
        ],
    )

def build_video_feed():
    """
    Build the video feed component for the dashboard.
    """
    event_spec = {"event": "click", "props": ["offsetX", "offsetY"]}
    return html.Div(
        id="video-container",
        children=[
            html.Div(
                [
                    html.Img(
                        id="point-canvas",
                        width="1500px",
                        height="720px",
                        src="",
                        # style={"position": "absolute", "pointerEvents": "all"},
                    ),
                    EventListener(
                        html.Canvas(
                            id="overlay-canvas",
                            width="1500px",
                            height="720px",
                            style={
                                "position": "absolute",
                                "left": "0",
                                "top": "0",
                            },
                        ),
                        events=[event_spec],
                        id="canvas-listener",
                    ),
                ],
                style={
                    "position": "relative",
                    "width": "100%",
                    "height": "720px",
                    "display": "inline-block",
                },
            ),
            html.Div(id="coords-output"),
            html.Div(
                children=[
                    dmc.Button(
                        "Top Left",
                        id="roi-top-left",
                        color="blue",
                        variant="filled",
                        size="xl",
                        style={"margin": "0.5%"},
                    ),
                    dmc.Button(
                        "Bottom Right",
                        id="roi-bot-right",
                        color="red",
                        variant="filled",
                        size="xl",
                        style={"margin": "0.5%"},
                    ),
                ],
                style={
                    "alignItems": "center",
                    "justifyContent": "center",
                    "display": "flex",
                },
            ),
        ],
    )


def init_df():
    ret = {}
    for col in list(df[1:]):
        data = df[col]
        stats = data.describe()

        std = stats["std"].tolist()
        ucl = (stats["mean"] + 3 * stats["std"]).tolist()
        lcl = (stats["mean"] - 3 * stats["std"]).tolist()
        usl = (stats["mean"] + stats["std"]).tolist()
        lsl = (stats["mean"] - stats["std"]).tolist()

        ret.update(
            {
                col: {
                    "count": stats["count"].tolist(),
                    "data": data,
                    "mean": stats["mean"].tolist(),
                    "std": std,
                    "ucl": round(ucl, 3),
                    "lcl": round(lcl, 3),
                    "usl": round(usl, 3),
                    "lsl": round(lsl, 3),
                    "min": stats["min"].tolist(),
                    "max": stats["max"].tolist(),
                    "ooc": populate_ooc(data, ucl, lcl),
                }
            }
        )

    return ret


def populate_ooc(data, ucl, lcl):
    ooc_count = 0
    ret = []
    for i in range(len(data)):
        if data[i] >= ucl or data[i] <= lcl:
            ooc_count += 1
            ret.append(ooc_count / (i + 1))
        else:
            ret.append(ooc_count / (i + 1))
    return ret


state_dict = init_df()


def init_value_setter_store():
    # Initialize store data
    
    state_dict = init_df()
    return state_dict


app.layout = dmc.MantineProvider(
    html.Div(
        id="big-app-container",
        children=[
            build_banner(),
            dcc.Interval(
                id="interval-component",
                interval=1000,  # in milliseconds
                n_intervals=5,  # start at batch 50
            ),
            dcc.Interval(
                id="plot-interval-component",
                interval=1000,  # in milliseconds
                n_intervals=5,  # start at batch 50
            ),
            dcc.Interval(
                id="img-interval-component",
                interval=1000,  # in milliseconds
                n_intervals=5,  # start at batch 50
            ),
            html.Div(
                id="app-container",
                children=[
                    build_tabs(),
                    # Main app
                    html.Div(id="app-content"),
                ],
            ),
            dcc.Store(id="value-setter-store", data=init_value_setter_store()),
            dcc.Store(id="n-interval-stage", data=50),
            # NOTE: donot change the local storage flag otherwise the js callbacks will fail
            dcc.Store(id="roi-coords-store", storage_type="local", data =[{"x": 0, "y": 0}, {"x": 1500, "y": 720}] ),
            dcc.Store(id="roi-selection-mode", storage_type="local"),
            dcc.Store(id="img-url-store"),
            dcc.Store(id="model_output"),
            dcc.Store(id="plot-queues")
            # generate_modal(),
        ],
    )
)


# ##################### Callbacks #####################



@app.callback(
    [Output("app-content", "children"), Output("interval-component", "n_intervals")],
    [Input("app-tabs", "value")],
    [State("n-interval-stage", "data")],
)
def render_tab_content(tab_switch, stopped_interval):
    return (
        html.Div(
            id="status-container",
            children=[
                # build_quick_stats_panel(),
                html.Div(
                    id="video-stats-container",
                    children=[build_video_feed(), build_barchart_panel()],
                ),
                html.Div(
                    id="graphs-container",
                    children=[
                        # build_top_panel(stopped_interval),
                        build_chart_panel(),
                    ],
                ),
            ],
        ),
        stopped_interval,
    )

def generate_graph(interval, specs_dict, col, queue_data):
    # Determine the time 30 minutes ago
    time_limit = datetime.datetime.now() - datetime.timedelta(minutes=30)

    # Filter the queues to only include data from the last 30 minutes
    filtered_level_x_array = []
    filtered_level_y_array = []
    filtered_qual_x_array = []
    filtered_qual_y_array = []

    # Iterate through the global queues and add only recent data
    # Assuming all queues have the same length and corresponding indices
    for i in range(len(date_queue)):
        if date_queue[i] >= time_limit:
            filtered_level_x_array.append(date_queue[i])
            filtered_level_y_array.append(level_queue[i])
            filtered_qual_x_array.append(date_queue[i])
            filtered_qual_y_array.append(qual_queue[i])

    fig = {
        "data": [
            {
                "x": filtered_level_x_array,
                "y": filtered_level_y_array,
                "mode": "lines+markers",
                "name": "Level",
                "line": {"color": "#1f77b4"},
            },
            {
                "x": filtered_qual_x_array,
                "y": filtered_qual_y_array,
                "mode": "lines+markers",
                "name": "Confidence",
                "line": {"color": "#ff7f0e"},
            },
        ]
    }

    fig["layout"] = dict(
        margin=dict(t=40),
        hovermode="closest",
        uirevision=col,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"font": {"color": "darkgray"}, "orientation": "h"},
        font={"color": "darkgray"},
        showlegend=True,
        xaxis={
            "title": "Time",
            "tickformat": "%H:%M:%S",
            "titlefont": {"color": "darkgray"},
            "showgrid": False,
            "showline": False,
            "zeroline": False,
            "autorange": True,
            # Set fixed range for X-axis to display only last 30 minutes
            "range": [time_limit.strftime('%Y-%m-%d %H:%M:%S'), datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        },
        yaxis={
            "title": "Value",
            "showgrid": False,
            "showline": False,
            "zeroline": False,
            "titlefont": {"color": "darkgray"},
            "range": [0, 100],
            "ticksuffix": "%"
        },
        xaxis2={
            "title": "Count",
            "titlefont": {"color": "darkgray"},
            "tickformat": "%H:%M:%S",
            "showgrid": False,
            "autorange": "max"
        },
        yaxis2={
            "anchor": "free",
            "overlaying": "y",
            "side": "right",
            "showticklabels": False,
            "titlefont": {"color": "darkgray"},
            "range": [0, 100]
        },
    )

    return fig

#  ======= button to choose/update figure based on click ============
@app.callback(
    output=Output("control-chart-live", "figure"),
    inputs=[
        Input("plot-interval-component", "n_intervals"),
    ],
    state=[State("value-setter-store", "data"), State("control-chart-live", "figure"), State("plot-queues", "data")],
)
def update_control_chart(interval, data, cur_fig, queue_data):
    # Find which one has been triggered
    ctx = dash.callback_context

    if not ctx.triggered:
        return generate_graph(interval, data, params[1], queue_data)

    if ctx.triggered:
        # Get most recently triggered id and prop_type
        splitted = ctx.triggered[0]["prop_id"].split(".")
        prop_id = splitted[0]
        prop_type = splitted[1]

        if prop_type == "n_clicks":
            curr_id = cur_fig["data"][0]["name"]
            prop_id = prop_id[:-7]
            if curr_id == prop_id:
                return generate_graph(interval, data, curr_id, queue_data)
            else:
                return generate_graph(interval, data, prop_id, queue_data)

        if prop_type == "n_intervals" and cur_fig is not None:
            curr_id = cur_fig["data"][0]["name"]
            return generate_graph(interval, data, curr_id, queue_data)


@app.callback(
    Output("point-canvas", "src"),
    Input("img-interval-component", "n_intervals"),
)
def update_image(n_intervals):
    return get_next_img(n_intervals)


@app.callback(
    Output("roi-coords-store", "data"),
    Output("roi-selection-mode", "data", allow_duplicate=True),
    Input("canvas-listener", "event"),
    State("roi-coords-store", "data"),
    State("roi-selection-mode", "data"),
    prevent_initial_call=True,
)
def store_click(event, data, selection_mode):
    if event is None or selection_mode["mode"] == "null":
        return data, {"mode": "null"}
    # Store the new point (offsetX, offsetY)
    if data:
        coords = data
        if len(data) == 1:
            coords.append({})
    else:
        coords = [{}, {}]

    if selection_mode["mode"] == "top_left":
        idx = 0
    else:
        idx = 1

    coords[idx] = {"x": event["offsetX"], "y": event["offsetY"]}
    return coords, {"mode": "null"}


@app.callback(Output("coords-output", "children"), Input("roi-coords-store", "data"))
def display_coords(data):
    if not data:
        return "Click on the canvas to store coordinates."
    return f"Stored points: {data}"


@app.callback(
    Output("roi-selection-mode", "data", allow_duplicate=True),
    Input("roi-top-left", "n_clicks"),
    prevent_initial_call=True,
)
def on_click_top_left(n_clicks):
    return {"mode": "top_left"}


@app.callback(
    Output("roi-selection-mode", "data", allow_duplicate=True),
    Input("roi-bot-right", "n_clicks"),
    prevent_initial_call=True,
)
def on_click_bot_right(n_clicks):
    return {"mode": "bot_right"}


@app.callback(
    Output("model_output", "data"),
    Input("img-interval-component", "n_intervals"),
    State("point-canvas", "src"),
)
def get_model_ouput(n_int, img_url):
    model, transform = load_model_if_not_loaded()
    if img_url:
        _, encoded = img_url.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        img = BytesIO(img_bytes)
        img = Image.open(img)
        img = transform(img).unsqueeze(0)
        img = img.to(device)
        with torch.no_grad():
            outputs = model(img)
        output = output = outputs["out"].squeeze(0)
        out_img = output[0].cpu().numpy()
        return out_img


@app.callback(
    Output("bar-chart-live-graph", "figure"),
    Output("plot-queues", "data"),
    Input("model_output", "data"),
    State("roi-coords-store", "data"),
)
def handle_segmentor_output(out, coords):
    # --- Mitigation Strategy ---
    # 1. Check if 'out' is None or an empty/invalid value from the model output
    #    (model_output.data will be the return value of get_model_ouput)
    if out is None or (isinstance(out, list) and not out):
        # If no valid model output, prevent further updates or return default/empty states
        print("Model output is not yet available or is invalid. Preventing update.")
        # dash.no_update can be used for individual outputs
        # For multiple outputs, returning defaults or raising PreventUpdate is cleaner.
        return dash.no_update, dash.no_update # If you don't want to change anything

        # OR, if you want to display an empty/default graph immediately:
        # return build_bar_figure(values=[0,0]), [] # Return default figure and empty queues

    # At this point, 'out' should be a list (from .tolist() in get_model_ouput)
    # Convert it back to a NumPy array for processing
    out = np.asarray(out)

    # 2. Ensure 'out' is a 2-dimensional array before attempting to slice it.
    if out.ndim != 2:
        print(f"Warning: 'out' is not 2-dimensional (shape: {out.shape}). Preventing update.")
        return dash.no_update, dash.no_update # Or return default figure

    # --- Your existing ROI and level calculation logic starts here ---
    if coords and len(coords) == 2 and coords[0] and coords[1]:
        c0 = coords[0]
        x1, y1 = c0["x"], c0["y"]
        c1 = coords[1]
        x2, y2 = c1["x"], c1["y"]
        x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
        x, y, w, h = min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)
    else:
        # Default ROI if coordinates are not set or partially set
        # Ensure default ROI covers the whole image if no coords provided
        x, y, w, h = 0, 0, out.shape[1], out.shape[0] # out.shape is (H, W)

    # 3. Add checks for valid ROI dimensions after calculation and before slicing
    #    This prevents errors if ROI coordinates result in zero or negative dimensions
    #    Also clip to ensure ROI is within image bounds
    x = np.clip(x, 0, out.shape[1] -1)
    y = np.clip(y, 0, out.shape[0] -1)
    # Ensure w and h are positive and don't extend beyond image boundaries
    w = np.clip(w, 1, out.shape[1] - x) # min width 1
    h = np.clip(h, 1, out.shape[0] - y) # min height 1

    # If, after clipping, w or h become 0 (e.g., if initial x/y were out of bounds)
    if w <= 0 or h <= 0:
        print(f"Invalid ROI dimensions after clipping: x={x}, y={y}, w={w}, h={h}. Preventing update.")
        return dash.no_update, dash.no_update


    # Only proceed if the ROI is valid
    # No need for this `if w > 0 and h > 0` check here if you clipped w,h to min 1 and handled above.
    # The clipping ensures w and h are at least 1, and x, y are within bounds for slicing.
    predicted_level, lvl_top, lvl_bot, y_top, y_bot = get_level(out[y : y + h, x : x + w])

    predicted_level = (h - predicted_level) * 100 / h
    y_top = max(0, y_top)  # Ensure y_top is not negative
    y_bot = min(h, y_bot)  # Ensure y_bot does not exceed height

    y_top = int((h - predicted_level) * 100 / h)
    y_bot = int((h - predicted_level) * 100 / h)
    print(f"Predicted level: {predicted_level}, Top: {lvl_top}, Bottom: {lvl_bot}")

    confidence = get_conf(out[y : y + h, x : x + w])
    lvl_top = np.clip(lvl_top, y, y + h) # Clip within ROI for confidence calculation
    lvl_bot = np.clip(lvl_bot, y, y + h)
    
    # Handle the case where lvl_top >= lvl_bot (empty slice for mean)
    if lvl_top >= lvl_bot:
        print("Warning: Level top is greater than or equal to level bottom for confidence calculation. Setting confidence to 0.")
        confidence = 0.0
    else:
        # Ensure no division by zero for mean if slice is empty (though clipping should prevent this)
        confidence_slice = confidence[lvl_top:lvl_bot]
        if confidence_slice.size == 0:
            print("Warning: Confidence slice is empty. Setting confidence to 0.")
            confidence = 0.0
        else:
            confidence = 1 / (1e-5 + np.mean(confidence_slice))

    confidence = np.clip(confidence, 0, 100) # clip with softmax (as per your comment, though softmax is usually before this)


    level_queue.append(predicted_level)
    qual_queue.append(confidence)
    date_queue.append(datetime.datetime.now())

    return build_bar_figure([predicted_level, confidence]), []

# NOTE: need threaded=True as getting the model output is time intensive
# to keep the app reactive enough, other callbacks should still handle events
# and threads helps with that.
app.run(debug=True, port=8050, threaded=True)