"""
Dashboard Visualizations

This module provides visualization utilities for the ACIS dashboard.
"""

import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import pandas as pd
import random
from datetime import datetime, timedelta
import json
from typing import List, Dict, Any


def create_mindmap(entity: str, relations: List[str]) -> str:
    """
    Create a mind map visualization using Plotly.
    
    Args:
        entity: The main entity
        relations: List of relation types
        
    Returns:
        HTML string of the visualization
    """
    # Create a network graph
    G = nx.Graph()
    
    # Add the main entity node
    G.add_node(entity, group=0)
    
    # Generate sample nodes and edges for each relation
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    relation_nodes = {}
    
    for i, relation in enumerate(relations):
        # Create nodes for this relation
        relation_nodes[relation] = []
        
        # Generate 3-5 sample nodes for each relation
        num_nodes = random.randint(3, 5)
        for j in range(num_nodes):
            node_name = f"{relation} {j+1}"
            G.add_node(node_name, group=i+1)
            relation_nodes[relation].append(node_name)
            
            # Connect to main entity
            G.add_edge(entity, node_name, weight=1)
        
        # Add some connections between nodes in same relation
        if len(relation_nodes[relation]) > 1:
            for j in range(len(relation_nodes[relation]) - 1):
                if random.random() > 0.3:  # 70% chance of connection
                    G.add_edge(relation_nodes[relation][j], relation_nodes[relation][j+1], weight=0.5)
    
    # Add some cross-relation connections
    if len(relations) > 1:
        for _ in range(min(5, len(relations))):
            rel1, rel2 = random.sample(relations, 2)
            if relation_nodes[rel1] and relation_nodes[rel2]:
                node1 = random.choice(relation_nodes[rel1])
                node2 = random.choice(relation_nodes[rel2])
                if random.random() > 0.7:  # 30% chance of cross-relation connection
                    G.add_edge(node1, node2, weight=0.3)
    
    # Get node positions using a layout algorithm
    pos = nx.spring_layout(G, seed=42)
    
    # Create edge trace
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    # Create node trace
    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_text = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Set color based on group
        group = G.nodes[node].get('group', 0)
        node_colors.append(colors[group % len(colors)])
        
        # Set size - larger for main entity
        if node == entity:
            node_sizes.append(25)
        else:
            node_sizes.append(15)
        
        node_text.append(node)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        marker=dict(
            color=node_colors,
            size=node_sizes,
            line=dict(width=1, color='#000')
        ))
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=dict(
                            text=f"Mind Map for {entity}",
                            font=dict(size=16)
                        ),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    
    # Return HTML representation
    return fig.to_html(include_plotlyjs=True, full_html=False)


def create_trend_chart(competitors: List[str], time_range: str, metrics: List[str]) -> go.Figure:
    """
    Create a trend chart visualization.
    
    Args:
        competitors: List of competitors
        time_range: Time range to display
        metrics: List of metrics to include
        
    Returns:
        Plotly figure
    """
    # Generate date range based on time_range
    end_date = datetime.now()
    if time_range == "1 month":
        start_date = end_date - timedelta(days=30)
        freq = "3D"
    elif time_range == "3 months":
        start_date = end_date - timedelta(days=90)
        freq = "1W"
    elif time_range == "6 months":
        start_date = end_date - timedelta(days=180)
        freq = "2W"
    elif time_range == "1 year":
        start_date = end_date - timedelta(days=365)
        freq = "1M"
    else:  # 2 years
        start_date = end_date - timedelta(days=730)
        freq = "2M"
    
    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # Create combined figure
    fig = go.Figure()
    
    # Generate data for each competitor and metric
    for metric in metrics:
        for competitor in competitors:
            # Generate random trend data
            base_value = random.uniform(50, 200)
            trend_factor = random.uniform(0.8, 1.2)
            noise_factor = random.uniform(0.05, 0.2)
            
            values = []
            for i in range(len(date_range)):
                # Create a trend with some noise
                value = base_value * (trend_factor ** (i / len(date_range))) * (1 + random.uniform(-noise_factor, noise_factor))
                values.append(value)
            
            # Add to figure
            fig.add_trace(go.Scatter(
                x=date_range,
                y=values,
                mode='lines+markers',
                name=f"{competitor} - {metric}",
                line=dict(width=2),
                marker=dict(size=6)
            ))
    
    # Update layout
    fig.update_layout(
        title=f"Competitor Trends: {', '.join(metrics)}",
        xaxis_title="Date",
        yaxis_title="Value",
        legend_title="Competitor & Metric",
        template="plotly_white"
    )
    
    return fig


def create_competitor_comparison_chart(competitors: List[str], metrics: List[str]) -> go.Figure:
    """
    Create a comparison chart for multiple competitors.
    
    Args:
        competitors: List of competitors
        metrics: List of metrics
        
    Returns:
        Plotly figure
    """
    # Generate data for each competitor and metric
    data = []
    
    for competitor in competitors:
        competitor_data = {"Competitor": competitor}
        
        for metric in metrics:
            # Generate random value
            competitor_data[metric] = random.uniform(0, 100)
        
        data.append(competitor_data)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create figure
    fig = px.bar(
        df, 
        x="Competitor", 
        y=metrics,
        barmode="group",
        title="Competitor Comparison",
        template="plotly_white"
    )
    
    return fig


def create_sentiment_timeline(competitor: str, time_range: str) -> go.Figure:
    """
    Create a sentiment timeline visualization.
    
    Args:
        competitor: Competitor name
        time_range: Time range to display
        
    Returns:
        Plotly figure
    """
    # Generate date range based on time_range
    end_date = datetime.now()
    if time_range == "1 month":
        start_date = end_date - timedelta(days=30)
        freq = "1D"
    elif time_range == "3 months":
        start_date = end_date - timedelta(days=90)
        freq = "3D"
    elif time_range == "6 months":
        start_date = end_date - timedelta(days=180)
        freq = "1W"
    else:  # 1 year
        start_date = end_date - timedelta(days=365)
        freq = "2W"
    
    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # Generate sentiment data
    sentiment_values = []
    for _ in range(len(date_range)):
        sentiment_values.append(random.uniform(-1, 1))
    
    # Create DataFrame
    df = pd.DataFrame({
        "Date": date_range,
        "Sentiment": sentiment_values
    })
    
    # Create figure
    fig = px.line(
        df, 
        x="Date", 
        y="Sentiment",
        title=f"Sentiment Timeline for {competitor}",
        template="plotly_white"
    )
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    # Add color gradient based on sentiment
    fig.update_traces(
        line=dict(
            color='green',
            width=3
        ),
        line_gradient=dict(
            color=df["Sentiment"].apply(lambda x: "green" if x > 0 else "red")
        )
    )
    
    return fig 