import os
import json
import shapely
import numpy as np
import matplotlib.pyplot as plt
from shapely.ops import unary_union
import cv2
from pyproj import Proj, Transformer
import dill


MVF_PATH = "/home/pascale/Documents/VPS/WayIL/Naver/hyundai/Hyundai Department Store Pangyo-mvf"
PICKLE_PATH = "hyundai_4F_mvf_data.pkl"
# AREA_MAPPING_FILE = "hyundai_area_index_mapping.json"
# LINE_MAPPING_FILE = "hyundai_line_index_mapping.json"
FLOOR_ID = "m_b68fd90d5826aecf"


def create_local_mercator_transform(ref_lat, ref_lon):
    """Create a transformer for local Mercator projection"""
    # WGS84 projection
    wgs84 = Proj("epsg:4326")

    # Custom local Mercator centered at reference point
    local_mercator = Proj(
        f"+proj=tmerc +lat_0={ref_lat} +lon_0={ref_lon} "
        "+k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
    )
    # Create transformer with always_xy=True to ensure consistent coordinate order
    return Transformer.from_proj(wgs84, local_mercator, always_xy=True)


class Floor:
    def __init__(self, floor_id, external_id, elevation, name):
        self.floor_id = floor_id
        self.external_id = external_id
        self.elevation = elevation
        self.name = name

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

        if not self.polygon.is_valid:
            raise ValueError("Invalid polygon", self.polygon_id)

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

    def __str__(self):
        return f"Space {self.space_id}: {self.center} with {self.shape}"


class Object:
    def __init__(
        self,
        object_id,
        external_id,
        floor_id,
        shape,
        kind=None,
        details=None,
        center=None,
    ):
        self.object_id = object_id
        self.external_id = external_id
        self.floor_id = floor_id
        self.kind = kind
        self.details = details
        self.shape = shape
        self.center = center

    def __str__(self):
        return f"Object {self.object_id}: {self.center} with {self.shape}"


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


class Entrance:
    def __init__(self, entrance_id, node_id, shape):
        self.entrance_id = entrance_id
        self.node_id = node_id
        self.shape = shape

    def __str__(self):
        return f"Entrance {self.entrance_id}: {self.node_id} with {self.shape}"


class MVF:
    """
    The class to represent an MVF from Self-serve
    (not to be confused with an enterprise MVF)
    """

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

        self.transformer = create_local_mercator_transform(
            ref_lat=self.center_point.y, ref_lon=self.center_point.x
        )

        # Store origin in UTM coordinates for local reference frame
        self.origin_east, self.origin_north = self.transformer.transform(
            self.center_point.x, self.center_point.y
        )

        self.map_name = manifest_data["features"][0]["properties"]["name"]
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

        self.spaces_list = []
        self.spaces_dict = {}
        self.spaces_dict_by_floor = {}

        self.objects_list = []
        self.objects_dict = {}
        self.objects_dict_by_floor = {}

        self.entrances_list = []
        self.entrances_dict = {}
        self.entrances_dict_by_floor = {}
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

            objects_by_floor = []
            with open(
                os.path.join(mvf_path, "obstruction", f"{floor_id}.geojson"), "r"
            ) as f:
                obstruction_data = json.load(f)
            for obstruction in obstruction_data["features"]:
                if obstruction["properties"]["kind"] != "object":
                    continue
                if obstruction["geometry"]["type"] == "Polygon":
                    geometry_obj = Polygon(
                        obstruction["properties"]["id"],
                        obstruction["geometry"]["coordinates"],
                    )
                elif obstruction["geometry"]["type"] == "LineString":
                    geometry_obj = LineString(
                        obstruction["properties"]["id"],
                        obstruction["geometry"]["coordinates"],
                    )
                elif obstruction["geometry"]["type"] == "Point":
                    geometry_obj = Point(
                        obstruction["geometry"]["coordinates"][0],
                        obstruction["geometry"]["coordinates"][1],
                    )
                else:
                    raise NotImplementedError(
                        "Only polygons, linestrings, and points are supported"
                    )
                if "center" in obstruction["properties"].keys():
                    center = Point(
                        obstruction["properties"]["center"][0],
                        obstruction["properties"]["center"][1],
                    )
                else:
                    center = None

                obj = Object(
                    obstruction["properties"]["id"],
                    obstruction["properties"]["externalId"],
                    floor_id,
                    geometry_obj,
                    kind=obstruction["properties"].get("kind", None),
                    details=obstruction["properties"].get("details", None),
                    center=center,
                )
                self.objects_list.append(obj)
                objects_by_floor.append(obj)
                self.objects_dict[obstruction["properties"]["id"]] = obj
            self.objects_dict_by_floor[floor_id] = objects_by_floor

            with open(
                os.path.join(mvf_path, "entrance", f"{floor_id}.geojson"), "r"
            ) as f:
                entrance_data = json.load(f)
            entrances_by_floor = []
            for entrance in entrance_data["features"]:
                geometry_obj = LineString(
                    entrance["properties"]["id"],
                    entrance["geometry"]["coordinates"],
                )
                entrance_obj = Entrance(
                    entrance["properties"]["id"],
                    entrance["properties"]["node"],
                    geometry_obj,
                )
                self.entrances_list.append(entrance_obj)
                entrances_by_floor.append(entrance_obj)
                self.entrances_dict[entrance["properties"]["id"]] = entrance_obj
            self.entrances_dict_by_floor[floor_id] = entrances_by_floor

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
        objects_on_this_floor = []
        entrances_on_this_floor = []

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

            spaces_on_this_floor.extend(spaces)
            nodes_on_this_floor.extend(nodes)
            objects_on_this_floor.extend(self.objects_dict_by_floor[floor_id])
            entrances_on_this_floor.extend(self.entrances_dict_by_floor[floor_id])

        return (
            spaces_on_this_floor,
            nodes_on_this_floor,
            objects_on_this_floor,
            entrances_on_this_floor,
        )

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

    def plot_all_shapes(self, spaces, nodes, objects, entrances, navigation_lines):
        """
        Display all the shapes in the given list of spaces, nodes,
        and navigation lines
        """
        for space in spaces + objects:
            if isinstance(space.shape, Polygon):
                coords = np.array(space.shape.polygon.exterior.coords)
                plt.plot(coords[:, 0], coords[:, 1], "k-")
                if isinstance(space, Object):
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

        for entrance in entrances:
            coords = np.array(entrance.shape.line.coords)
            plt.plot(coords[:, 0], coords[:, 1], "b-")

        for node in nodes:
            plt.plot(node.point.x, node.point.y, "ro")

        for line in navigation_lines:
            coords = np.array(line.line.coords)
            plt.plot(coords[:, 0], coords[:, 1], "r")

        plt.axis("equal")
        plt.show()

    def get_navigable_spaces(self, spaces, navigation_lines):
        """
        Navigable spaces are spaces that contain nodes
        """
        navigable_spaces = []
        navigable_spaces_ids = []

        for navigation_line in navigation_lines:
            for space in spaces:
                if space.space_id in navigable_spaces_ids:
                    continue
                if isinstance(space.shape, Polygon) and space.shape.polygon is not None:
                    if navigation_line.line.intersects(space.shape.polygon.boundary):
                        navigable_spaces_ids.append(space.space_id)
                        navigable_spaces.append(space)

        return navigable_spaces

    def get_navigable_nodes(self, navigable_spaces):
        """
        Return a list of all nodes that can be navigated to
        """

        navigable_node_ids = []
        for space in navigable_spaces:
            navigable_node_ids.extend(space.node_id)
        navigable_node_ids = list(set(navigable_node_ids))

        navigable_nodes = []
        for node_id in navigable_node_ids:
            node = self.nodes_dict[node_id]
            navigable_nodes.append(node)

        return navigable_nodes

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

    def rasterize_polygons(
        self,
        spaces,
        navigable_space_ids,
        navigation_lines,
        objects,
        entrances,
        nodes,
        pixels_per_meter=2,
    ):
        """
        Convert spaces to a binary raster image
        Args:
            spaces: List of Space objects
            pixels_per_meter: Number of pixels per meter
        Returns:
            numpy array of shape (H,W) with binary values
        """
        # Convert all polygon coordinates to meters
        polygons = []
        polygon_space_ids = []
        for space in spaces:
            if isinstance(space.shape, Polygon) and space.shape.polygon is not None:
                # Convert each point in the polygon
                exterior_coords = []
                for lon, lat in space.shape.polygon.exterior.coords:
                    x, y = self.transformer.transform(lon, lat)
                    exterior_coords.append((x, y))

                interior_coords = []
                for interior in space.shape.polygon.interiors:
                    interior_ring = []
                    for lon, lat in interior.coords:
                        x, y = self.transformer.transform(lon, lat)
                        interior_ring.append((x, y))
                    interior_coords.append(interior_ring)

                # Create new polygon in meters
                if interior_coords:
                    poly = shapely.geometry.Polygon(exterior_coords, interior_coords)
                else:
                    poly = shapely.geometry.Polygon(exterior_coords)
                polygons.append(poly)
                polygon_space_ids.append(space.space_id)
        union = unary_union(polygons)
        bounds = union.bounds  # (minx, miny, maxx, maxy)

        # Calculate raster dimensions
        width_meters = bounds[2] - bounds[0]
        height_meters = bounds[3] - bounds[1]
        width_pixels = int(np.ceil(width_meters * pixels_per_meter))
        height_pixels = int(np.ceil(height_meters * pixels_per_meter))

        # Create empty raster
        raster_polygons = np.zeros((height_pixels, width_pixels), dtype=np.uint8)
        raster_lines = np.zeros((height_pixels, width_pixels), dtype=np.uint8)
        raster_points = np.zeros((height_pixels, width_pixels), dtype=np.uint8)

        # Define class indices
        polygon_class_mapping = {
            "non_navigable": 1,
            "navigable": 2,
            "objects": 3,
        }
        line_class_mapping = {
            "walls": 1,
            "entrances": 2,
            "navigation_lines": 3,
            "objects": 4,
        }

        # Draw each space into raster
        for polygon_space_id, polygon in zip(polygon_space_ids, polygons):
            if polygon_space_id in navigable_space_ids:
                colour = polygon_class_mapping["navigable"]
            else:
                colour = polygon_class_mapping["non_navigable"]

            # Convert polygon coordinates to pixel coordinates
            coords = np.array(polygon.exterior.coords)
            x = (coords[:, 0] - bounds[0]) * pixels_per_meter
            y = (coords[:, 1] - bounds[1]) * pixels_per_meter
            x = x.astype(np.int32)
            y = y.astype(np.int32)

            # Fill polygon
            pts = np.stack([x, y], axis=1)
            cv2.fillPoly(raster_polygons, [pts], colour)
            # Draw outline
            cv2.polylines(
                raster_lines,
                [pts],
                isClosed=True,
                color=line_class_mapping["walls"],
                thickness=2,
            )

        for obj in objects:
            if isinstance(obj.shape, Polygon) and obj.shape.polygon is not None:
                # Convert each point in the polygon
                exterior_coords = []
                for lon, lat in obj.shape.polygon.exterior.coords:
                    x, y = self.transformer.transform(lon, lat)
                    exterior_coords.append((x, y))

                poly = shapely.geometry.Polygon(exterior_coords)

                # Convert polygon coordinates to pixel coordinates
                coords = np.array(poly.exterior.coords)
                x = (coords[:, 0] - bounds[0]) * pixels_per_meter
                y = (coords[:, 1] - bounds[1]) * pixels_per_meter
                x = x.astype(np.int32)
                y = y.astype(np.int32)

                # Fill polygon
                pts = np.stack([x, y], axis=1)
                cv2.fillPoly(raster_polygons, [pts], polygon_class_mapping["objects"])
                cv2.polylines(
                    raster_lines,
                    [pts],
                    isClosed=True,
                    color=line_class_mapping["objects"],
                    thickness=2,
                )

        for node in nodes:
            x, y = self.transformer.transform(node.point.x, node.point.y)
            # Convert node coordinates to pixel coordinates
            x = int((x - bounds[0]) * pixels_per_meter)
            y = int((y - bounds[1]) * pixels_per_meter)

            cv2.circle(raster_points, (x, y), 5, 1, -1)

        for entrance in entrances:
            meter_coords = []
            for lon, lat in entrance.shape.line.coords:
                x, y = self.transformer.transform(lon, lat)
                meter_coords.append((x, y))
            entrance_meters = shapely.geometry.LineString(meter_coords)

            # Convert line coordinates to pixel coordinates
            coords = np.array(entrance_meters.coords)
            x = (coords[:, 0] - bounds[0]) * pixels_per_meter
            y = (coords[:, 1] - bounds[1]) * pixels_per_meter
            x = x.astype(np.int32)
            y = y.astype(np.int32)

            pts = np.stack([x, y], axis=1)
            cv2.polylines(
                raster_lines,
                [pts],
                isClosed=True,
                color=line_class_mapping["entrances"],
                thickness=2,
            )

        for navigation_line in navigation_lines:
            meter_coords = []
            for lon, lat in navigation_line.line.coords:
                x, y = self.transformer.transform(lon, lat)
                meter_coords.append((x, y))
            navigation_line_meters = shapely.geometry.LineString(meter_coords)

            # Convert line coordinates to pixel coordinates
            coords = np.array(navigation_line_meters.coords)
            x = (coords[:, 0] - bounds[0]) * pixels_per_meter
            y = (coords[:, 1] - bounds[1]) * pixels_per_meter
            x = x.astype(np.int32)
            y = y.astype(np.int32)

            pts = np.stack([x, y], axis=1)
            cv2.polylines(
                raster_lines,
                [pts],
                isClosed=True,
                color=line_class_mapping["navigation_lines"],
                thickness=2,
            )

        return raster_polygons, raster_lines, raster_points, bounds


if __name__ == "__main__":
    mvf = MVF(MVF_PATH)
    spaces, nodes, objects, entrances = mvf.load_geometry_for_floor(FLOOR_ID)
    # Get all lines that connect nodes
    temp_navigation_lines = mvf.get_navigation_graph(nodes)
    # Navigable spaces have a navigation line that crosses its boundary
    navigable_spaces = mvf.get_navigable_spaces(spaces, temp_navigation_lines)
    # Navigable nodes are the ones that pass through the navigable spaces
    navigable_nodes = mvf.get_navigable_nodes(navigable_spaces)
    navigation_lines = mvf.get_navigation_graph(navigable_nodes)

    navigable_space_ids = [space.space_id for space in navigable_spaces]

    # mvf.plot_all_shapes(spaces, navigable_nodes, objects, entrances, navigation_lines)

    polygon_raster, line_raster, node_raster, bounds = mvf.rasterize_polygons(
        spaces,
        navigable_space_ids,
        navigation_lines,
        objects,
        entrances,
        navigable_nodes,
        pixels_per_meter=2,
    )

    # print(len(spaces))
    # print(len(navigable_spaces))

    # Visualize result
    # plt.imshow(line_raster, origin="lower")
    # plt.title("Rasterized Spaces (2 pixels/meter)")
    # plt.show()

    output_mvf_dict = {
        FLOOR_ID: {
            "spaces": spaces,
            "nodes": navigable_nodes,
            "objects": objects,
            "entrances": entrances,
            "navigation_lines": navigation_lines,
            "bounds": bounds,
        },
        "center_point_lon": mvf.center_point.x,
        "center_point_lat": mvf.center_point.y,
    }

    # Save MVF object to pickle file
    with open(f"{FLOOR_ID}_mvf_data.pkl", "wb") as f:
        dill.dump(output_mvf_dict, f)

    raster_map_dir = "raster_maps"
    if not os.path.exists(raster_map_dir):
        os.makedirs(raster_map_dir, exist_ok=True)
    output_raster_file = os.path.join(raster_map_dir, f"raster_map_{FLOOR_ID}.npy")

    raster_image = np.stack([polygon_raster, line_raster, node_raster], axis=0)
    np.save(output_raster_file, raster_image)

    # with open(AREA_MAPPING_FILE, "w") as f:
    #     json.dump(
    #         {
    #             "non_navigable": 1,
    #             "navigable": 2,
    #             "objects": 3,
    #         },
    #         f,
    #         indent=2,
    #     )
    # with open(LINE_MAPPING_FILE, "w") as f:
    #     json.dump(
    #         {
    #             "walls": 1,
    #             "entrances": 2,
    #             "navigation_lines": 3,
    #             "objects": 4,
    #         },
    #         f,
    #         indent=2,
    #     )
