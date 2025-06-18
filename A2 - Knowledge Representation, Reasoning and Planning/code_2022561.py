# Boilerplate for AI Assignment — Knowledge Representation, Reasoning and Planning
# CSE 643

from collections import defaultdict, deque

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Import necessary libraries
import pandas as pd
import plotly.graph_objects as go
from pyDatalog import pyDatalog

## ****IMPORTANT****
## Don't import or use any other libraries other than defined above
## Otherwise your code file will be rejected in the automated testing

# ------------------ Global Variables ------------------
route_to_stops = defaultdict(list)  # Mapping of route IDs to lists of stops
trip_to_route = {}  # Mapping of trip IDs to route IDs
stop_trip_count = defaultdict(int)  # Count of trips for each stop
fare_rules = {}  # Mapping of route IDs to fare information
merged_fare_df = None  # To be initialized in create_kb()

# Load static data from GTFS (General Transit Feed Specification) files
df_stops = pd.read_csv("GTFS/stops.txt")
df_stops["stop_id"] = df_stops["stop_id"].astype(str)
df_stops["zone_id"] = df_stops["zone_id"].astype(str)

df_routes = pd.read_csv("GTFS/routes.txt")
df_routes["route_id"] = df_routes["route_id"].astype(str)

df_stop_times = pd.read_csv("GTFS/stop_times.txt")
df_stop_times["stop_id"] = df_stop_times["stop_id"].astype(str)

df_fare_attributes = pd.read_csv("GTFS/fare_attributes.txt")

df_trips = pd.read_csv("GTFS/trips.txt")
df_trips["route_id"] = df_trips["route_id"].astype(str)
df_trips["service_id"] = df_trips["service_id"].astype(str)

df_fare_rules = pd.read_csv("GTFS/fare_rules.txt")
df_fare_rules["route_id"] = df_fare_rules["route_id"].astype(str)
df_fare_rules["origin_id"] = df_fare_rules["origin_id"].astype(str)
df_fare_rules["destination_id"] = df_fare_rules["destination_id"].astype(str)

df_route_stop = pd.merge(df_stop_times, df_trips, on="trip_id")[
    ["route_id", "stop_id"]
].drop_duplicates()

df_stop_times_inc = df_stop_times.sort_values(["trip_id", "stop_sequence"])

# ------------------ Function Definitions ------------------


# Function to create knowledge base from the loaded data
def create_kb():
    """
    Create knowledge base by populating global variables with information from loaded datasets.
    It establishes the relationships between routes, trips, stops, and fare rules.

    Returns:
        None
    """
    global route_to_stops, trip_to_route, stop_trip_count, fare_rules, merged_fare_df

    # Create trip_id to route_id mapping
    trip_to_route.update(
        df_trips.groupby("trip_id")["route_id"].apply(lambda x: list(set(x))).to_dict()
    )

    # Map route_id to a list of unique stops in order of their sequence
    route_to_stops.update(
        df_route_stop.groupby("route_id")["stop_id"].apply(list).to_dict()
    )

    # Count trips per stop
    stop_trip_count.update(
        df_stop_times_inc.groupby("trip_id")["stop_id"].nunique().to_dict()
    )

    # Create fare rules for routes
    fare_rules.update(
        df_fare_rules.groupby("route_id")
        .apply(lambda x: list(zip(x["fare_id"], x["origin_id"], x["destination_id"])))
        .to_dict()
    )

    # Merge fare rules and attributes into a single DataFrame
    merged_fare_df = pd.merge(
        df_fare_rules, df_fare_attributes, on="fare_id", how="outer"
    )


# Function to find the top 5 busiest routes based on the number of trips
def get_busiest_routes():
    """
    Identify the top 5 busiest routes based on trip counts.

    Returns:
        list: A list of tuples, where each tuple contains:
              - route_id (int): The ID of the route.
              - trip_count (int): The number of trips for that route.
    """
    busiest_routes = list(df_trips["route_id"].value_counts().head(5).to_dict().items())
    print(busiest_routes)
    return busiest_routes


# Function to find the top 5 stops with the most frequent trips
def get_most_frequent_stops():
    """
    Identify the top 5 stops with the highest number of trips.

    Returns:
        list: A list of tuples, where each tuple contains:
              - stop_id (int): The ID of the stop.
              - trip_count (int): The number of trips for that stop.
    """
    frequent_stops = sorted(stop_trip_count.items(), key=lambda x: x[1], reverse=True)[
        :5
    ]
    print(frequent_stops)
    return frequent_stops


# Function to find the top 5 busiest stops based on the number of routes passing through them
def get_top_5_busiest_stops():
    """
    Identify the top 5 stops with the highest number of different routes.

    Returns:
        list: A list of tuples, where each tuple contains:
              - stop_id (int): The ID of the stop.
              - route_count (int): The number of routes passing through that stop.
    """
    busiest_stops = list(
        df_route_stop["stop_id"].value_counts().head(5).to_dict().items()
    )
    print(busiest_stops)
    return busiest_stops


# Function to identify the top 5 pairs of stops with only one direct route between them
def get_stops_with_one_direct_route():
    """
    Identify the top 5 pairs of consecutive stops (start and end) connected by exactly one direct route.
    The pairs are sorted by the combined frequency of trips passing through both stops.

    Returns:
        list: A list of tuples, where each tuple contains:
              - pair (tuple): A tuple with two stop IDs (stop_1, stop_2).
              - route_id (int): The ID of the route connecting the two stops.
    """
    #   df_stop_times_2 = df_stop_times.copy()
    #   df_stop_times_2["route_id"] = df_stop_times_2["trip_id"].map(
    #       lambda x: trip_to_route.get(x, [None])[0]
    #   )

    #   df_stop_times_2 = df_stop_times_2.sort_values(by=["trip_id", "stop_id"])
    #   df_stop_times_2["next_stop"] = df_stop_times_2.groupby("trip_id")["stop_id"].shift(
    #       -1
    #   )

    #   df_consecutive_stops = df_stop_times_2.dropna(subset=["next_stop"])

    #   df_consecutive_stops["combined_freq"] = df_consecutive_stops["stop_id"].map(
    #       stop_trip_count
    #   ) + df_consecutive_stops["next_stop"].map(stop_trip_count)

    #   pair_counts = (
    #       df_consecutive_stops.groupby(["route_id", "stop_id", "next_stop"])[
    #           "combined_freq"
    #       ]
    #       .sum()
    #       .reset_index()
    #   )
    #   top_pair_stops_df = pair_counts.sort_values(
    #       by="combined_freq", ascending=False).head(5)

    #   top_pair_stops = [
    #       ((row["stop_id"], row["next_stop"]), row["route_id"])
    #       for _, row in top_pair_stops_df
    #   ]

    #   top_pair_stops = []
    #   for _, row in top_pair_stops_df.iterrows():
    #       top_pair_stops.append(((row["stop_id"], row["next_stop"]), row["route_id"]))

    #   print(top_pair_stops)
    #   return top_pair_stops
    df_stop_times["route_id"] = (
        df_stop_times["trip_id"]
        .astype(str)
        .map(lambda x: trip_to_route.get(x, [None])[0])
    )

    df_stop_times = df_stop_times.sort_values(by=["trip_id", "stop_id"])
    df_stop_times["next_stop"] = df_stop_times.groupby("trip_id")["stop_id"].shift(-1)

    df_consecutive_stops = df_stop_times.dropna(subset=["next_stop"])

    df_consecutive_stops["combined_freq"] = df_consecutive_stops["stop_id"].map(
        stop_trip_count
    ) + df_consecutive_stops["next_stop"].map(stop_trip_count)

    pair_counts = (
        df_consecutive_stops.groupby(["route_id", "stop_id", "next_stop"])[
            "combined_freq"
        ]
        .sum()
        .reset_index()
    )

    top_stop_pairs_df = pair_counts.sort_values(
        by="combined_freq", ascending=False
    ).head(5)

    top_stop_pairs = []
    for _, row in top_stop_pairs_df.iterrows():
        stop_pair = ((row["stop_id"], row["next_stop"]), row["route_id"])
        top_stop_pairs.append(stop_pair)

    return top_stop_pairs


# Function to get merged fare DataFrame
# No need to change this function
def get_merged_fare_df():
    """
    Retrieve the merged fare DataFrame.

    Returns:
        DataFrame: The merged fare DataFrame containing fare rules and attributes.
    """
    global merged_fare_df
    return merged_fare_df


# Visualize the stop-route graph interactively
def visualize_stop_route_graph_interactive(route_to_stops):
    """
    Visualize the stop-route graph using Plotly for interactive exploration.

    Args:
        route_to_stops (dict): A dictionary mapping route IDs to lists of stops.

    Returns:
        None
    """
    graph = nx.DiGraph()

    for route_id, stops in route_to_stops.items():
        graph.add_node(route_id, label="Route", type="route")

        for i in range(len(stops) - 1):
            s1, s2 = stops[i], stops[i + 1]
            graph.add_edge(s1, s2, label=route_id)
    pos = nx.spring_layout(graph)

    edges_x = []
    edges_y = []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edges_x.append(x0)
        edges_x.append(x1)
        edges_x.append(None)
        edges_y.append(y0)
        edges_y.append(y0)
        edges_y.append(None)

    edge_trace = go.Scatter(
        x=edges_x, y=edges_y, line=dict(width=0.5), hoverinfo="none", mode="lines"
    )
    print("doing1")
    nodes_x = []
    nodes_y = []
    node_text = []
    for node in graph.nodes():
        x, y = pos[node]
        nodes_x.append(x)
        nodes_y.append(y)
        node_text.append(str(node))

    # node_trace = go.Scatter(x=nodes_x,y=nodes_y,mode='markers',hoverinfo='text',marker=dict(thickness=15,title='Node Connections',xanchor='left',titleside='right'),line_width=2,text=node_text)

    node_trace = go.Scatter(
        x=nodes_x,
        y=nodes_y,
        mode="markers",
        hoverinfo="text",
        marker=dict(
            showscale=True,
            colorscale="YlGnBu",
            size=10,
            colorbar=dict(
                thickness=15,
                title="Node Connections",
                xanchor="left",
                titleside="right",
            ),
            line=dict(width=2),
        ),
        text=node_text,
    )
    print("doing2")

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="<br>Route to Stops Interactive Graph",
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=0, l=0, r=0, t=40),
            annotations=[
                dict(
                    text="Visualization of Route to Stop connections",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.005,
                    y=-0.002,
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
        ),
    )
    fig.show()


# Brute-Force Approach for finding direct routes
def direct_route_brute_force(start_stop, end_stop):
    """
    Find all valid routes between two stops using a brute-force method.

    Args:
        start_stop (int): The ID of the starting stop.
        end_stop (int): The ID of the ending stop.

    Returns:
        list: A list of route IDs (int) that connect the two stops directly.
    """
    ans_routes = df_fare_rules[
        (df_fare_rules["origin_id"] == int(start_stop))
        & (df_fare_rules["destination_id"].astype(int) == int(end_stop))
    ]
    ans_routes = np.unique(ans_routes["route_id"].astype(int))
    return ans_routes.tolist()


# Initialize Datalog predicates for reasoning
pyDatalog.create_terms("RouteHasStop, DirectRoute, OptimalRoute, X, Y, Z, R, R1, R2")


def initialize_datalog():
    """
    Initialize Datalog terms and predicates for reasoning about routes and stops.

    Returns:
        None
    """
    pyDatalog.clear()  # Clear previous terms
    print(
        "Terms initialized: DirectRoute, RouteHasStop, OptimalRoute"
    )  # Confirmation print

    # Define Datalog predicates

    create_kb()  # Populate the knowledge base
    add_route_data(route_to_stops)  # Add route data to Datalog


# Adding route data to Datalog
def add_route_data(route_to_stops):
    """
    Add the route data to Datalog for reasoning.

    Args:
        route_to_stops (dict): A dictionary mapping route IDs to lists of stops.

    Returns:
        None
    """
    for i, j in route_to_stops.items():
        for k in j:
            +RouteHasStop(i, k)
    DirectRoute(X, Y, R) <= RouteHasStop(R, X) & RouteHasStop(R, Y)


# Function to query direct routes between two stops
def query_direct_routes(start, end):
    """
    Query for direct routes between two stops.

    Args:
        start (int): The ID of the starting stop.
        end (int): The ID of the ending stop.

    Returns:
        list: A sorted list of route IDs (str) connecting the two stops.
    """
    ans = list(DirectRoute(start, end, R))
    finalans = [x[0] for x in ans]
    return finalans


# Forward chaining for optimal route planning
def forward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform forward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """
    forward = (
        DirectRoute(start_stop_id, stop_id_to_include, R1)
        & DirectRoute(stop_id_to_include, end_stop_id, R2)
        & (R1 != R2)
    )
    forward = [(r1, stop_id_to_include, r2) for r1, r2 in forward]
    return list(forward)


# Backward chaining for optimal route planning
def backward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform backward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """
    backward = (
        DirectRoute(start_stop_id, stop_id_to_include, R1)
        & DirectRoute(stop_id_to_include, end_stop_id, R2)
        & (R1 != R2)
    )
    backward = [(r1, stop_id_to_include, r2) for r1, r2 in backward]
    return list(backward)


# PDDL-style planning for route finding
def pddl_planning(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Implement PDDL-style planning to find routes with optional transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID for a transfer.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """
    pass  # Implementation here


# Function to filter fare data based on an initial fare limit
def prune_data(merged_fare_df, initial_fare):
    """
    Filter fare data based on an initial fare limit.

    Args:
        merged_fare_df (DataFrame): The merged fare DataFrame.
        initial_fare (float): The maximum fare allowed.

    Returns:
        DataFrame: A filtered DataFrame containing only routes within the fare limit.
    """
    pass  # Implementation here


# Pre-computation of Route Summary
def compute_route_summary(pruned_df):
    """
    Generate a summary of routes based on fare information.

    Args:
        pruned_df (DataFrame): The filtered DataFrame containing fare information.

    Returns:
        dict: A summary of routes with the following structure:
              {
                  route_id (int): {
                      'min_price': float,          # The minimum fare for the route
                      'stops': set                # A set of stop IDs for that route
                  }
              }
    """
    pass  # Implementation here


# BFS for optimized route planning
def bfs_route_planner_optimized(
    start_stop_id, end_stop_id, initial_fare, route_summary, max_transfers=3
):
    """
    Use Breadth-First Search (BFS) to find the optimal route while considering fare constraints.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        initial_fare (float): The available fare for the trip.
        route_summary (dict): A summary of routes with fare and stop information.
        max_transfers (int): The maximum number of transfers allowed (default is 3).

    Returns:
        list: A list representing the optimal route with stops and routes taken, structured as:
              [
                  (route_id (int), stop_id (int)),  # Tuple for each stop taken in the route
                  ...
              ]
    """
    pass  # Implementation here
