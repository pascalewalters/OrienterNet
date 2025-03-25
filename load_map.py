import os
import json
import shapely
import numpy as np
import matplotlib.pyplot as plt
from shapely.ops import unary_union
import yaml
import pandas as pd

MVF_PATH = "/home/pascale/Documents/VPS/Mappedin_VPS_Data/YYC_VPS/YYC_MVF/"


class Floor:
    def __init__(self, floor_id, external_id, elevation, name, short_name):
        self.floor_id = floor_id
        self.external_id = external_id
        self.elevation = elevation
        self.name = name
        self.short_name = short_name

    def __str__(self):
        return f"Floor {self.floor_id}: {self.name}"


class FloorStack:
    def __init__(self, floorstack_id, external_id, name, maps, default_floor=None):
        self.floorstack_id = floorstack_id
        self.external_id = external_id
        self.map_ids = maps
        self.floors = []
        self.name = name
        self.default_floor = default_floor

    def __str__(self):
        return (
            f"FloorStack {self.floorstack_id}: {self.name} with {len(self.floors)} floors\n"
            + f"{[str(floor) for floor in self.floors]}"
        )


class Point:
    def __init__(self, x, y):
        self.point = shapely.geometry.Point(x, y)
        self.x = x
        self.y = y

    def __str__(self):
        return f"({self.x}, {self.y})"


class LineString:
    def __init__(self, line_id, points=[]):
        self.line_id = line_id
        self.points = points
        self.line = shapely.geometry.LineString(points)

    def __str__(self):
        return f"LineString with {len(self.points)} points"


class Polygon:
    def __init__(self, polygon_id, points=None):
        self.polygon_id = polygon_id
        # This can either be a list of points or a list of lists of points
        if points is None:
            self.points = []
            self.polygon = None
        elif len(points) == 1 and len(points[0]) == 2:
            # The list of points is [[x1, y1], [x2, y2], ...]
            self.polygon = shapely.geometry.Polygon(points)
            self.points = [points, []]
        elif len(points) == 1 and len(points[0]) > 2:
            # The list of points is [[[x1, y1], [x2, y2], ...], []]
            self.polygon = shapely.geometry.Polygon(points[0])
            self.points = points
        elif len(points) > 1:
            # The list of points is [[[x1, y1], [x2, y2], ...], [[x3, y3], [x4, y4], ...], ...]
            # (outer and inner rings)
            outer = points[0]
            inners = points[1:]
            self.polygon = shapely.geometry.Polygon(outer, inners)
            self.points = [outer, inners]
        else:
            raise ValueError("Invalid polygon points", points)

    def __str__(self):
        if len(self.points) == 0:
            return "Polygon with no points"
        else:
            return f"Polygon with {len(self.points[0])} points"


class Node:
    def __init__(self, node_id, external_id, map_id, point, neighbors, space):
        self.node_id = node_id
        self.external_id = external_id
        self.point = point
        self.map_id = map_id
        self.neighbors = neighbors
        self.space = space

    def __str__(self):
        return f"Node {self.node_id} at {self.point} on floor {self.map_id}"


class Connection:
    def __init__(self, connection_id, type, nodes, external_id, accessible=False):
        self.connection_id = connection_id
        self.type = type
        self.nodes = nodes
        self.external_id = external_id
        self.accessible = accessible

    def __str__(self):
        return f"Connection {self.connection_id} with {len(self.nodes)} nodes"


class Style:
    def __init__(self, style_id, altitude, color, height, opacity, shapes=[]):
        self.style_id = style_id
        self.altitude = altitude
        self.color = color
        self.height = height
        self.opacity = opacity
        self.shapes = shapes

    def __str__(self):
        return f"Style {self.style_id}: {self.color} at altitude {self.altitude}"


class Space:
    def __init__(
        self,
        space_id,
        external_id,
        floor_id,
        shape,
        kind=None,
        details=None,
        destination_nodes=[],
        center=None,
    ):
        self.space_id = space_id
        self.external_id = external_id
        self.floor_id = floor_id
        self.kind = kind
        self.details = details
        self.shape = shape
        self.destination_nodes = destination_nodes
        self.center = center

        self.node_id = []

        self.location = []
        self.category = []
        self.layer = ""

    def __str__(self):
        return f"Space {self.space_id}: {self.center} with {self.shape}"


class Category:
    def __init__(
        self, category_id, name, external_id, sort_order, locations, children, color
    ):
        self.category_id = category_id
        self.name = name
        self.external_id = external_id
        self.sort_order = sort_order
        self.locations = locations
        self.children = children
        self.color = color

    def __str__(self):
        return f"Category {self.category_id}: {self.name}"


class Location:
    def __init__(
        self,
        location_id,
        name,
        location_type,
        external_id,
        sort_order,
        polygons,
        spaces,
        nodes,
        description,
    ):
        self.location_id = location_id
        self.name = name
        self.location_type = location_type
        self.external_id = external_id
        self.sort_order = sort_order
        self.polygons = polygons
        self.spaces = spaces
        self.nodes = nodes
        self.description = description

    def __str__(self):
        return f"Location {self.location_id}: {self.name}"


class Layer:
    def __init__(self, layer_id, name, spaces):
        self.layer_id = layer_id
        self.name = name
        self.spaces = spaces

    def __str__(self):
        return f"Layer {self.layer_id}: {self.name} with {len(self.spaces)} spaces"


class MVF:
    def __init__(self, mvf_path):
        self.floorstacks = []
        self.floor_id_to_floorstack = {}

        # Read in manifest file
        with open(os.path.join(mvf_path, "manifest.geojson"), "r") as f:
            manifest_data = json.load(f)

        self.center_point = Point(
            manifest_data["features"][0]["geometry"]["coordinates"][0],
            manifest_data["features"][0]["geometry"]["coordinates"][1],
        )

        self.name = manifest_data["features"][0]["properties"]["name"]
        self.map_name = manifest_data["features"][0]["properties"]["map"]
        self.version = manifest_data["features"][0]["properties"]["version"]
        self.natural_bearing = manifest_data["features"][0]["properties"][
            "naturalBearing"
        ]
        self.folder_structure = manifest_data["features"][0]["properties"][
            "folder_struct"
        ]
        self.file_list = []

        for entry in self.folder_structure:
            if entry["type"] == "file":
                self.file_list.append(entry["name"])
            elif entry["type"] == "folder":
                for file_entry in entry["children"]:
                    self.file_list.append(
                        os.path.join(entry["name"], file_entry["name"])
                    )

        with open(os.path.join(mvf_path, "floorstack.json"), "r") as f:
            floorstacks_data = json.load(f)

        for floorstack_data in floorstacks_data:
            floorstack = FloorStack(
                floorstack_data["id"],
                floorstack_data["externalId"],
                floorstack_data["name"],
                floorstack_data["maps"],
                floorstack_data.get("defaultFloor", None),
            )
            self.floorstacks.append(floorstack)

            for floor_id in floorstack_data["maps"]:
                self.floor_id_to_floorstack[floor_id] = floorstack

        with open(os.path.join(mvf_path, "floor.geojson"), "r") as f:
            floors_data = json.load(f)

        self.floor_id_dict = {}

        for floor_data in floors_data["features"]:
            floor = Floor(
                floor_data["properties"]["id"],
                floor_data["properties"]["externalId"],
                floor_data["properties"]["elevation"],
                floor_data["properties"]["name"],
                floor_data["properties"]["shortName"],
            )
            self.floor_id_to_floorstack[floor.floor_id].floors.append(floor)
            self.floor_id_dict[floor.floor_id] = floor

        with open(os.path.join(mvf_path, "node.geojson"), "r") as f:
            node_data = json.load(f)

        self.nodes = []
        self.nodes_dict = {}
        for node in node_data["features"]:
            node_point = Point(
                node["geometry"]["coordinates"][0], node["geometry"]["coordinates"][1]
            )
            node_obj = Node(
                node["properties"]["id"],
                node["properties"]["externalId"],
                node["properties"]["map"],
                node_point,
                node["properties"]["neighbors"],
                node["properties"]["space"],
            )
            self.nodes.append(node_obj)
            if node["properties"]["id"] in self.nodes_dict:
                raise ValueError(f"Duplicate node ID {node['properties']['id']}")
            self.nodes_dict[node["properties"]["id"]] = node_obj

        with open(os.path.join(mvf_path, "connection.json"), "r") as f:
            connection_data = json.load(f)

        self.connections = []
        for connection in connection_data:
            connection_obj = Connection(
                connection["id"],
                connection["type"],
                connection["nodes"],
                connection["externalId"],
                connection.get("accessible", False),
            )
            self.connections.append(connection_obj)

        with open(os.path.join(mvf_path, "styles.json"), "r") as f:
            style_data = json.load(f)

        self.styles = []
        for style_id, style_entry in style_data.items():
            if "polygons" in style_entry.keys():
                style_obj = Style(
                    style_id,
                    style_entry["altitude"],
                    style_entry["color"],
                    style_entry["height"],
                    style_entry["opacity"],
                    [Polygon(polygon_id) for polygon_id in style_entry["polygons"]],
                )
            else:
                raise NotImplementedError("Only polygon styles are supported")
            self.styles.append(style_obj)

        self.spaces_list = []
        self.spaces_dict = {}
        self.spaces_dict_by_floor = {}
        # There is one space file per floor
        for floor_id, floorstack in self.floor_id_to_floorstack.items():
            with open(os.path.join(mvf_path, "space", f"{floor_id}.geojson"), "r") as f:
                space_data = json.load(f)

            spaces_by_floor = []

            for space in space_data["features"]:
                if space["geometry"]["type"] == "Polygon":
                    geometry_obj = Polygon(
                        space["properties"]["id"],
                        space["geometry"]["coordinates"],
                    )
                elif space["geometry"]["type"] == "LineString":
                    geometry_obj = LineString(
                        space["properties"]["id"],
                        space["geometry"]["coordinates"],
                    )
                elif space["geometry"]["type"] == "Point":
                    geometry_obj = Point(
                        space["geometry"]["coordinates"][0],
                        space["geometry"]["coordinates"][1],
                    )
                else:
                    raise NotImplementedError(
                        "Only polygons, linestrings, and points are supported"
                    )

                if "center" in space["properties"].keys():
                    center = Point(
                        space["properties"]["center"][0],
                        space["properties"]["center"][1],
                    )
                else:
                    center = None

                space_obj = Space(
                    space["properties"]["id"],
                    space["properties"]["externalId"],
                    floor_id,
                    geometry_obj,
                    kind=space["properties"].get("kind", None),
                    details=space["properties"].get("details", None),
                    destination_nodes=space["properties"].get("destinationNodes", None),
                    center=center,
                )
                self.spaces_list.append(space_obj)
                spaces_by_floor.append(space_obj)
                if space["properties"]["id"] in self.spaces_dict:
                    raise ValueError(f"Duplicate space ID {space['properties']['id']}")
                self.spaces_dict[space["properties"]["id"]] = space_obj
            self.spaces_dict_by_floor[floor_id] = spaces_by_floor

        # Read in enterprise data
        with open(os.path.join(mvf_path, "enterprise", "venue.json"), "r") as f:
            enterprise_venue_data = json.load(f)
        # print(enterprise_venue_data)
        # TODO: do something with the venue data?

        with open(os.path.join(mvf_path, "enterprise", "categories.json"), "r") as f:
            enterprise_categories_data = json.load(f)

        self.categories = []
        for enterprise_category in enterprise_categories_data:
            self.categories.append(
                Category(
                    enterprise_category["id"],
                    enterprise_category["name"],
                    enterprise_category["externalId"],
                    enterprise_category["sortOrder"],
                    enterprise_category["locations"],
                    enterprise_category.get("children", []),
                    enterprise_category.get("color", ""),
                )
            )

        with open(os.path.join(mvf_path, "enterprise", "locations.json"), "r") as f:
            enterprise_location_data = json.load(f)

        self.locations = []
        for enterprise_location in enterprise_location_data:
            self.locations.append(
                Location(
                    enterprise_location["id"],
                    enterprise_location["name"],
                    enterprise_location["type"],
                    enterprise_location["externalId"],
                    enterprise_location["sortOrder"],
                    enterprise_location["polygons"],
                    enterprise_location.get("spaces", []),
                    enterprise_location["nodes"],
                    enterprise_location.get("description", ""),
                )
            )

        with open(os.path.join(mvf_path, "enterprise", "layers.json"), "r") as f:
            layers_data = json.load(f)

        self.layers = []
        for layer_data in layers_data:
            self.layers.append(
                Layer(layer_data["id"], layer_data["name"], layer_data["spaces"])
            )

            for space_dict in layer_data["spaces"]:
                space = self.spaces_dict[space_dict["spaceId"]]
                space.layer = layer_data["name"]

        self.floors_of_interest = []
        self.navigation_lines = []

    def __str__(self):
        return f"MVF with {len(self.floorstacks)} floorstacks"

    def load_geometry_for_floor(self, potential_floor_id):
        """
        For a potential floor ID, load all the geometry on that floor.
        Search all floors across the venue at the same elevation. For example,
        International Terminal Level and Domestic Terminal Level 1
        """
        potential_floor_elevation = self.floor_id_dict[potential_floor_id].elevation

        same_elevation_floor_ids = []
        for floor_id, floor in self.floor_id_dict.items():
            if floor.elevation == potential_floor_elevation:
                same_elevation_floor_ids.append(floor_id)
        print(f"Considering floors at elevation {potential_floor_elevation}")

        self.floors_of_interest = same_elevation_floor_ids

        nodes_on_this_floor = []
        spaces_on_this_floor = []

        for floor_id in same_elevation_floor_ids:
            print(
                f"Floor {floor_id}: {self.floor_id_to_floorstack[floor_id].name} {self.floor_id_dict[floor_id].name}"
            )

            nodes = []
            for node in self.nodes:
                if node.map_id == floor_id:
                    nodes.append(node)

            spaces = self.spaces_dict_by_floor[floor_id]
            for space in spaces:
                for node in nodes:
                    if space.space_id in node.space:
                        space.node_id.append(node.node_id)

                for location in self.locations:
                    location_polygon_ids = [
                        polygon["id"] for polygon in location.polygons
                    ]
                    if space.space_id in location_polygon_ids:
                        space.location.append(location)

                        for category in self.categories:
                            if location.location_id in category.locations:
                                space.category.append(category)

                    location_node_ids = [node["id"] for node in location.nodes]

                    node_id_intersection = list(
                        set(space.node_id) & set(location_node_ids)
                    )
                    if len(node_id_intersection) > 0:
                        space.location.append(location)

                        for category in self.categories:
                            if location.location_id in category.locations:
                                space.category.append(category)

                    location_space_ids = [space["id"] for space in location.spaces]
                    if space.space_id in location_space_ids:
                        space.location.append(location)

                        for category in self.categories:
                            if location.location_id in category.locations:
                                space.category.append(category)

                if space.layer == "":
                    space.layer = "None"

            spaces_on_this_floor.extend(spaces)
            nodes_on_this_floor.extend(nodes)

        return spaces_on_this_floor, nodes_on_this_floor

    def linestring_to_polygon(self, linestring):
        """
        Convert a Shapely LineString to a Polygon

        Args:
            linestring: Shapely LineString object
        Returns:
            Shapely Polygon object
        """
        coords = list(linestring.coords)

        # Check if the linestring is closed (first and last points match)
        if coords[0] != coords[-1]:
            coords.append(coords[0])  # Close the ring by adding first point at end

        return shapely.Polygon(coords)

    def get_bounding_polygon_for_spaces(self, spaces):
        """
        Return a shapely Polygon that bounds all the spaces
        """
        # Assuming you have a list of Space objects with polygon shapes
        polygons = [
            space.shape.polygon
            for space in spaces
            if isinstance(space.shape, Polygon) and space.shape.polygon is not None
        ]

        # Combine all polygons into one
        union = unary_union(polygons)
        # Get the boundary of the combined polygons
        boundary = union.boundary
        if boundary.geom_type == "LineString":
            return [self.linestring_to_polygon(boundary)]
        elif boundary.geom_type == "MultiLineString":
            # Convert each linestring to a polygon
            boundary_polygons = []
            for line in boundary.geoms:
                try:
                    poly = self.linestring_to_polygon(line)
                    if poly.is_valid:
                        boundary_polygons.append(poly)
                except Exception:
                    continue

            # Sort polygons by area (largest first)
            boundary_polygons.sort(key=lambda p: p.area, reverse=True)

            # Keep only non-overlapping polygons
            non_overlapping = []
            for poly in boundary_polygons:
                # Check if this polygon overlaps with any we've already kept
                if not any(poly.intersects(p) for p in non_overlapping):
                    non_overlapping.append(poly)

            # Return the largest non-overlapping polygon
            return non_overlapping

        return union

    def plot_all_shapes(self, spaces, nodes, navigation_lines):
        """
        Display all the shapes in the given list of spaces, nodes,
        and navigation lines
        """
        for space in spaces:
            if isinstance(space.shape, Polygon):
                coords = np.array(space.shape.polygon.exterior.coords)
                plt.plot(coords[:, 0], coords[:, 1], "k-")
                if space.layer == "Non Public":
                    plt.fill(coords[:, 0], coords[:, 1], "gray", alpha=0.5)
                for interior in space.shape.polygon.interiors:
                    coords = np.array(interior.coords)
                    plt.plot(coords[:, 0], coords[:, 1], "k-")
            elif isinstance(space.shape, LineString):
                coords = np.array(space.shape.line.coords)
                plt.plot(coords[:, 0], coords[:, 1], "k-")
            elif isinstance(space.shape, Point):
                plt.plot(space.shape.x, space.shape.y, "ko")
            else:
                raise ValueError(f"Unsupported shape type {type(space.shape)}")

        for node in nodes:
            plt.plot(node.point.x, node.point.y, "ro")

        for line in navigation_lines:
            coords = np.array(line.line.coords)
            plt.plot(coords[:, 0], coords[:, 1], "r")

        plt.axis("equal")
        plt.show()

    def set_floors(self, floor_ids):
        """
        We don't need to consider all of the floors of the MVF. We'll also need
        to combine floors of the same elevation (e.g., International Terminal Level 1
        and Domestic Terminal Level 1) into one floor.
        """
        # Get the floorstacks that contain the floor_ids
        relevant_floorstacks = [
            floorstack
            for floorstack in self.floorstacks
            if any(f"f_{floor_id}" in floorstack.map_ids for floor_id in floor_ids)
        ]
        self.floorstacks = relevant_floorstacks

        # Get the floors that are in the relevant floorstacks
        relevant_floors_dict = {}
        for floorstack in self.floorstacks:
            for floor_id in floorstack.map_ids:
                relevant_floors_dict[floor_id] = self.floor_id_dict[floor_id]
        self.floor_id_dict = relevant_floors_dict

    def get_categories(self, spaces):
        """
        Get all the categories of the spaces
        """
        categories = set()
        for space in spaces:
            if len(space.category) > 0:
                categories.update([category.name for category in space.category])
            else:
                categories.add("None")
        return list(categories)

    def set_categories(self, categories_path, spaces):
        """
        Set the categories for the MVF
        """
        with open(categories_path, "r") as f:
            categories_data = yaml.load(f, Loader=yaml.FullLoader)

        new_spaces = []
        for space in spaces:
            if len(space.category) > 0:
                for category in space.category:
                    if (
                        categories_data["categories"][category.name]
                        and categories_data["layers"][space.layer]
                    ):
                        new_spaces.append(space)
                        # Only append the space once
                        break

            else:
                if (
                    categories_data["categories"]["None"]
                    and categories_data["layers"][space.layer]
                ):
                    new_spaces.append(space)

        return new_spaces

    def get_layers(self, spaces):
        """
        Get all the layers of the spaces
        """
        layers = set()
        for space in spaces:
            layers.add(space.layer)
        return list(layers)

    def get_navigable_spaces(self, spaces, navigation_lines):
        """
        Navigable spaces are spaces that contain nodes
        """
        navigable_spaces = []
        for space in spaces:
            if len(space.node_id) > 0:
                navigable_spaces.append(space)

        for navigation_line in navigation_lines:
            for space in spaces:
                if isinstance(space.shape, Polygon) and space.shape.polygon is not None:
                    if space.shape.polygon.intersects(navigation_line.line):
                        navigable_spaces.append(space)

        return navigable_spaces

    def get_spaces_in_bounding_polygon(self, bounds, spaces):
        """
        Return all of the spaces within the bounding polygon

        Args:
            bounds: Shapely Polygon that defines the boundary
            spaces: List of Space objects to check

        Returns:
            List of Space objects that intersect with the boundary
        """
        spaces_within = []
        for space in spaces:
            if isinstance(space.shape, Polygon) and space.shape.polygon is not None:
                if bounds.intersects(space.shape.polygon):
                    spaces_within.append(space)
        return spaces_within

    def get_navigation_graph(self, nodes):
        """
        Represent the navigation graph as a list of linestrings
        """
        self.navigation_lines = []
        for node in nodes:
            for neighbor in node.neighbors:
                if self.nodes_dict[neighbor["id"]].map_id in self.floors_of_interest:
                    # Only consider nodes that are on this floor
                    # (don't consider connections between floors)
                    line = LineString(
                        f"{node.node_id}_{neighbor['id']}",
                        [node.point.point, self.nodes_dict[neighbor["id"]].point.point],
                    )
                    self.navigation_lines.append(line)

        return self.navigation_lines


if __name__ == "__main__":
    mvf = MVF(MVF_PATH)
