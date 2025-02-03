import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from typing import List, Dict

class MVFPlotter:
    def __init__(self, zoom=18):  # Default to higher zoom for building-level view
        self.fig = go.Figure()
        self.fig.update_layout(
            mapbox_style="open-street-map",
            autosize=True,
            mapbox_zoom=zoom,
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            showlegend=True,
        )

    def plot_spaces(self, spaces: Dict, color='rgb(200,200,200)', name='Spaces'):
        """Plot rooms and hallways."""
        for space_id, space in spaces.items():
            coords = space['geometry']['coordinates'][0]  # Assuming polygon
            self.fig.add_trace(
                go.Scattermapbox(
                    lat=[coord[1] for coord in coords],
                    lon=[coord[0] for coord in coords],
                    mode='lines',
                    fill='toself',
                    fillcolor=color,
                    name=f"{name}: {space['properties'].get('name', space_id)}",
                    opacity=0.5,
                )
            )

    def plot_nodes(self, nodes: Dict, color='rgb(255,0,0)', size=5):
        """Plot navigation nodes."""
        for node in nodes['features']:
            coord = node['geometry']['coordinates']
            self.fig.add_trace(
                go.Scattermapbox(
                    lat=[coord[1]],
                    lon=[coord[0]],
                    mode='markers',
                    marker_size=size,
                    marker_color=color,
                    name=f"Node: {node['properties'].get('id', 'unknown')}"
                )
            )

    def plot_connections(self, connections: Dict, color='rgb(0,255,0)'):
        """Plot vertical connections (stairs/elevators)."""
        for conn_id, conn in connections.items():
            # Assuming connection has start and end coordinates
            start = conn['properties']['startPoint']
            end = conn['properties']['endPoint']
            self.fig.add_trace(
                go.Scattermapbox(
                    lat=[start[1], end[1]],
                    lon=[start[0], end[0]],
                    mode='lines',
                    line=dict(color=color, width=2),
                    name=f"Connection: {conn['properties'].get('type', 'unknown')}"
                )
            )

    def plot_entrances(self, entrances: Dict, color='rgb(0,0,255)', size=8):
        """Plot entrances between spaces."""
        for entrance_id, entrance in entrances.items():
            coords = entrance['geometry']['coordinates']
            self.fig.add_trace(
                go.Scattermapbox(
                    lat=[coords[1]],
                    lon=[coords[0]],
                    mode='markers',
                    marker_symbol='square',
                    marker_size=size,
                    marker_color=color,
                    name=f"Entrance: {entrance_id}"
                )
            )

    def plot_windows(self, windows: Dict, color='rgb(0,255,255)'):
        """Plot windows as LineStrings."""
        for window_id, window in windows.items():
            coords = window['geometry']['coordinates']
            self.fig.add_trace(
                go.Scattermapbox(
                    lat=[coord[1] for coord in coords],
                    lon=[coord[0] for coord in coords],
                    mode='lines',
                    line=dict(color=color, width=2),
                    name=f"Window: {window_id}"
                )
            )

    def plot_obstructions(self, obstructions: Dict, color='rgb(169,169,169)'):
        """Plot non-traversable areas."""
        for obs_id, obstruction in obstructions.items():
            coords = obstruction['geometry']['coordinates'][0]  # Assuming polygon
            self.fig.add_trace(
                go.Scattermapbox(
                    lat=[coord[1] for coord in coords],
                    lon=[coord[0] for coord in coords],
                    mode='lines',
                    fill='toself',
                    fillcolor=color,
                    opacity=0.7,
                    name=f"Obstruction: {obstruction['properties'].get('type', obs_id)}"
                )
            )

    def set_center(self, manifest: Dict):
        """Set the center of the map based on manifest center point."""
        center_point = manifest['features'][0]['geometry']['coordinates']
        self.fig.update_layout(
            mapbox_center=dict(
                lat=center_point[1],
                lon=center_point[0]
            )
        )

# Color scheme for different MVF elements
mvf_colors = {
    'space': 'rgb(200,200,200)',      # Gray for rooms
    'node': 'rgb(255,0,0)',           # Red for nodes
    'connection': 'rgb(0,255,0)',     # Green for connections
    'entrance': 'rgb(0,0,255)',       # Blue for entrances
    'window': 'rgb(0,255,255)',       # Cyan for windows
    'obstruction': 'rgb(169,169,169)' # Dark gray for obstructions
}

def plot_floor(mvf_data: Dict, floor_id: str):
    """Create a visualization for a specific floor."""
    plotter = MVFPlotter()
    
    # Set center from manifest
    plotter.set_center(mvf_data['manifest'])
    
    # Plot each element type for the specified floor
    for space in mvf_data['spaces'].values():
        if space['properties']['floorId'] == floor_id:
            plotter.plot_spaces({'space': space}, color=mvf_colors['space'])
    
    # Filter and plot other elements by floor_id
    # ... similar filtering for other element types ...
    
    return plotter.fig
