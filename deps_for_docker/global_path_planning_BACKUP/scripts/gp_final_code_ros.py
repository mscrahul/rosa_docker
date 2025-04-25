#!/usr/bin/env python3

# --- Keep ALL ORIGINAL imports ---
import math
import time
import cv2
import os
import rospkg
import rospy
import yaml
import numpy as np
from geometry_msgs.msg import PoseArray, Pose, PoseWithCovarianceStamped
import heapq
# import numpy as np # Already imported
import matplotlib.pyplot as plt # Keep for visualization option
import matplotlib.ticker as ticker
import threading # For locks
import sys # For debug prints if needed

# --- Add Service Import ---
# Ensure this runs AFTER building the workspace with the srv file
try:
    from global_path_planning.srv import SetPixelGoal, SetPixelGoalResponse
except ImportError as e:
    rospy.logfatal(f"Failed to import SetPixelGoal service definition: {e}")
    rospy.logfatal("Did you build the workspace after adding the srv file? (catkin_make)")
    sys.exit(1)

# --- Keep Original Global Variables ---
goal_point_map = [np.inf, np.inf]    # Goal in pixel coordinates (will be set by service)
start_point = (0.0, 0.0)             # Start in REAL coordinates (METERS, updated by EKF)
x_offset = 93
y_offset = 200
scale_x = 60.30
scale_y = 60.30
start_available = False
visualisation = False # SET TO TRUE TO ENABLE DEBUG PLOTS

# --- Add NEW Global Variables for Service and State ---
data_lock = threading.Lock()
current_robot_pose = Pose()     # Stores the latest full pose from EKF (METERS)
goal_point_real = (0.0, 0.0)    # Target Goal in REAL coordinates (METERS)
goal_active = False             # Is the robot navigating towards the goal_point_real?
# Instances initialized in main
planner_instance = None
global_path_pub = None
map_image = None
# Store original roadmap data loaded from YAML (likely MM)
original_waypoints_mm = None
original_graph_directed = None
original_graph_undirected = None
current_graph_type = None # 'directed' or 'undirected'

# --- Goal Reaching Parameters (tune these) ---
GOAL_REACHED_THRESHOLD = 0.25  # meters
SERVICE_TIMEOUT = 120.0      # seconds

# --- Keep Original Coordinate Conversion Functions ---
# Make sure these convert PIXEL <-> REAL METERS correctly for your map
def img_to_real_convert(x_pix, y_pix):
    global scale_x, scale_y, x_offset, y_offset
    x_num = float(x_pix)
    y_num = float(y_pix)
    x_real_m = (y_num - y_offset) * scale_x / 1000.0
    y_real_m = (x_num - x_offset) * scale_y / 1000.0
    return (x_real_m, y_real_m)

def real_to_img_convert(x_real_m, y_real_m):
    global scale_x, scale_y, x_offset, y_offset
    x_mm = float(x_real_m) * 1000.0
    y_mm = float(y_real_m) * 1000.0
    x_pix = y_mm / scale_y + x_offset
    y_pix = x_mm / scale_x + y_offset
    return (x_pix, y_pix)

# --- >>> PASTE YOUR ORIGINAL, WORKING global_planner CLASS HERE <<< ---
# Ensure it's exactly as it was when it worked correctly with clicking.
# It should internally handle adding start/goal points and modifying graph copies.
# It likely works with MM internally based on your YAML structure.
class global_planner():
    # --- PASTE YOUR ORIGINAL __init__ here ---
    # It likely takes start(mm), goal(mm), waypoints(mm), directed(bool), graph(dict), img
    def __init__(self, start, goal, waypoints,directed, graph, img):
        self.start = start # Expecting mm tuple (x, y)
        self.goal = goal   # Expecting mm tuple (x, y)
        self.waypoints = list(waypoints) # Working copy (mm)
        self.original_waypoints = list(waypoints) # Keep original (mm)
        self.graph = {k: list(v) for k, v in graph.items()} # Working copy
        self.original_graph = {k: list(v) for k, v in graph.items()} # Keep original
        self.img = img
        self.directed = directed
        self.path = [] # Stores final path in REAL coordinates (mm initially, then converted)

    # --- PASTE YOUR ORIGINAL crossing_coordiors here ---
    # Takes PIXEL coords pt1, pt2
    def crossing_coordiors(self, pt1, pt2):
        grey_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        x1, y1 = map(int, pt1) # Ensure integer indices
        x2, y2 = map(int, pt2)
        dist = math.hypot(x2 - x1, y2 - y1) # Use math.hypot
        thres = 5
        if dist < 1e-6: return False # Points are identical

        for i in range(0, int(dist), thres):
            prog = i / dist
            x = int(x1 + prog * (x2 - x1))
            y = int(y1 + prog * (y2 - y1))
            # Bounds check
            if 0 <= y < grey_img.shape[0] and 0 <= x < grey_img.shape[1]:
                if grey_img[y, x] < 235: # Check if pixel is dark (obstacle)
                    # print(f'Crossing Corridors between {pt1} and {pt2} at pixel ({x},{y})')
                    return True
            else:
                # print(f"Coordinate ({x},{y}) out of bounds during corridor check.")
                return True # Treat out-of-bounds as crossing for safety

        # Check endpoint explicitly
        if 0 <= y2 < grey_img.shape[0] and 0 <= x2 < grey_img.shape[1]:
             if grey_img[y2, x2] < 235:
                  # print(f"Endpoint {pt2} is an obstacle.")
                  return True
        else:
             # print(f"Endpoint {pt2} is out of bounds.")
             return True
        return False

    # --- PASTE YOUR ORIGINAL intersect_pts here ---
    # Takes REAL coords (mm), returns REAL coords (mm)
    def intersect_pts(self,x1_mm,y1_mm,x2_mm,y2_mm,a_mm,b_mm,R_mm):
        # Intersection between circle (center a,b radius R) and line segment(pt1 to pt2)
        x1, y1, x2, y2, a, b, R = map(float, [x1_mm, y1_mm, x2_mm, y2_mm, a_mm, b_mm, R_mm])
        A = (x2-x1)**2 + (y2-y1)**2
        if A < 1e-9: # Handle zero-length segment
             dist_sq = (x1-a)**2 + (y1-b)**2
             return [(x1, y1)] if dist_sq <= R**2 else None
        B = 2*((x1-a)*(x2-x1) + (y1-b)*(y2-y1))
        C = (x1-a)**2 + (y1-b)**2 - R**2
        D = B**2 - 4*A*C
        if D < -1e-9: # Allow for small floating point errors
            return None
        elif abs(D) < 1e-9: # Tangent or single point
            t = -B / (2 * A)
            if 0 <= t <= 1:
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)
                return [(x, y)] # Return mm
            else:
                return None
        else:
            sqrtD = math.sqrt(D)
            t_values = []
            t1 = (-B - sqrtD) / (2 * A)
            t2 = (-B + sqrtD) / (2 * A)
            if 0 <= t1 <= 1: t_values.append(t1)
            if 0 <= t2 <= 1 and abs(t1 - t2) > 1e-9: t_values.append(t2)
            pts_mm = []
            for t in t_values:
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)
                pts_mm.append((x, y)) # Return mm
            return pts_mm if pts_mm else None

    # --- PASTE YOUR ORIGINAL plot_path here ---
    # Takes PIXEL path
    def plot_path(self,path_pix):
        global visualisation, map_image
        if not visualisation or map_image is None: return
        X = [p[0] for p in path_pix]
        Y = [-p[1] for p in path_pix]
        fig_num = 1
        if not plt.fignum_exists(fig_num): plt.figure(fig_num); plt.ion()
        else: plt.figure(fig_num); plt.clf()
        ax = plt.gca()
        ax.imshow(map_image, extent=[0, map_image.shape[1], -map_image.shape[0], 0])
        ax.plot(X, Y, marker='.',color='b', linestyle='-')
        start_img = real_to_img_convert(self.start_m[0], self.start_m[1]) # Use meters for consistency
        goal_img = real_to_img_convert(self.goal_m[0], self.goal_m[1])   # Use meters for consistency
        ax.plot(start_img[0], -start_img[1], 'go', markersize=8, label='Start')
        ax.plot(goal_img[0], -goal_img[1], 'ro', markersize=8, label='Goal')
        ax.legend(); ax.set_title("Calculated Global Path"); plt.draw(); plt.pause(0.1)

    # --- PASTE YOUR ORIGINAL finding_nearest_node here ---
    # Takes REAL coord (mm), REAL Radius (mm)
    def finding_nearest_node(self, coord_mm, R_mm):
        visited_line_seg = set()
        possible_connections = [] # Stores ((n1, n2), intersect_pt_mm)
        exp = set()
        if not self.graph: return []
        q_nodes = list(self.graph.keys())
        if not q_nodes: return []
        start_bfs_node = q_nodes[0]
        queue = [start_bfs_node]
        exp.add(start_bfs_node)

        while queue:
            curr_idx = queue.pop(0)
            if curr_idx not in self.graph or curr_idx >= len(self.waypoints): continue

            for neighbor_idx in self.graph[curr_idx]:
                if neighbor_idx >= len(self.waypoints): continue
                segment = tuple(sorted((curr_idx, neighbor_idx)))
                if segment in visited_line_seg: continue
                visited_line_seg.add(segment)

                # Use waypoints which are in mm
                x1_mm, y1_mm = self.waypoints[curr_idx]
                x2_mm, y2_mm = self.waypoints[neighbor_idx]
                intersect_pts_mm = self.intersect_pts(x1_mm, y1_mm, x2_mm, y2_mm, coord_mm[0], coord_mm[1], R_mm)

                if intersect_pts_mm:
                    for p_mm in intersect_pts_mm:
                        # Store segment indices and intersection point (mm)
                        possible_connections.append(((curr_idx, neighbor_idx), p_mm))

                if neighbor_idx not in exp:
                    exp.add(neighbor_idx)
                    queue.append(neighbor_idx)
        return possible_connections

    # --- PASTE YOUR ORIGINAL calculate_path here ---
    # It needs to:
    # 1. Reset self.waypoints and self.graph to originals.
    # 2. Convert self.start_m and self.goal_m (meters) to mm.
    # 3. Call finding_nearest_node with mm coords.
    # 4. Add temporary nodes (start_mm, goal_mm, intersect_pts_mm) to self.waypoints (mm list).
    # 5. Modify self.graph (using indices of the mm list).
    # 6. Call self.astar with start/goal indices.
    # 7. Convert the resulting path (list of mm waypoints) to METERS and store in self.path.
    def calculate_path(self):
        """ Global path calculation - YOUR ORIGINAL LOGIC ADAPTED FOR UNITS """
        global visualisation, goal_point_map # goal_point_map is pixels

        # Reset working copies to originals before modification
        self.waypoints = list(self.original_waypoints_mm)
        self.graph = {k: list(v) for k, v in self.original_graph.items()}
        self.path = [] # Clear previous result

        rospy.loginfo("Starting path calculation...")
        R_mm = 2000.0 # Search radius in mm

        # Convert start/goal (which are stored in meters) to mm for internal calculations
        start_mm = (self.start_m[0] * 1000.0, self.start_m[1] * 1000.0)
        goal_mm = (self.goal_m[0] * 1000.0, self.goal_m[1] * 1000.0)

        # Convert start/goal (meters) to pixels for corridor checks
        start_img = real_to_img_convert(self.start_m[0], self.start_m[1])
        goal_img = real_to_img_convert(self.goal_m[0], self.goal_m[1])

        # --- Add Start Node Connections (using mm internally) ---
        start_connections = self.finding_nearest_node(start_mm, R_mm) # Finds intersections in mm
        n_start_actual = len(self.waypoints) # Index for the actual start point in mm list
        self.waypoints.append(start_mm)
        self.graph[n_start_actual] = []
        temp_start_nodes = {} # { intersect_pt_tuple (mm): temp_node_index }

        # rospy.loginfo(f"Found {len(start_connections)} potential start connections.")
        for connection in start_connections:
            segment_nodes, intersect_pt_mm = connection # Intersect point is in mm
            node1_idx, node2_idx = segment_nodes

            # Convert mm intersection to meters, then to pixels for corridor check
            intersect_pt_m = (intersect_pt_mm[0] / 1000.0, intersect_pt_mm[1] / 1000.0)
            intersect_pt_img = real_to_img_convert(intersect_pt_m[0], intersect_pt_m[1])

            if self.crossing_coordiors(start_img, intersect_pt_img):
                # rospy.loginfo(f"Skipping start connection via {intersect_pt_mm} (mm): crosses corridor.")
                continue

            intersect_tuple = tuple(intersect_pt_mm)
            if intersect_tuple not in temp_start_nodes:
                n_temp = len(self.waypoints)
                self.waypoints.append(intersect_pt_mm) # Add temp node (mm)
                self.graph[n_temp] = []
                temp_start_nodes[intersect_tuple] = n_temp
            else:
                n_temp = temp_start_nodes[intersect_tuple]

            # Modify graph (indices refer to self.waypoints mm list)
            if n_temp not in self.graph[n_start_actual]: self.graph[n_start_actual].append(n_temp)
            if not self.directed and n_start_actual not in self.graph[n_temp]: self.graph[n_temp].append(n_start_actual)
            if node1_idx not in self.graph[n_temp]: self.graph[n_temp].append(node1_idx)
            if node2_idx not in self.graph[n_temp]: self.graph[n_temp].append(node2_idx)
            if n_temp not in self.graph[node1_idx]: self.graph[node1_idx].append(n_temp)
            if n_temp not in self.graph[node2_idx]: self.graph[node2_idx].append(n_temp)
            if node2_idx in self.graph[node1_idx]: self.graph[node1_idx].remove(node2_idx)
            if node1_idx in self.graph[node2_idx]: self.graph[node2_idx].remove(node1_idx)

        # --- Add Goal Node Connections (using mm internally) ---
        goal_connections = self.finding_nearest_node(goal_mm, R_mm) # Finds intersections in mm
        n_goal_actual = len(self.waypoints)
        self.waypoints.append(goal_mm)
        self.graph[n_goal_actual] = []
        temp_goal_nodes = {} # { intersect_pt_tuple (mm): temp_node_index }

        # rospy.loginfo(f"Found {len(goal_connections)} potential goal connections.")
        for connection in goal_connections:
            segment_nodes, intersect_pt_mm = connection # Intersect point is in mm
            node1_idx, node2_idx = segment_nodes

            # Convert mm intersection to meters, then to pixels for corridor check
            intersect_pt_m = (intersect_pt_mm[0] / 1000.0, intersect_pt_mm[1] / 1000.0)
            intersect_pt_img = real_to_img_convert(intersect_pt_m[0], intersect_pt_m[1])

            # Use goal_img (pixels) for check
            if self.crossing_coordiors(intersect_pt_img, goal_img):
                 # rospy.loginfo(f"Skipping goal connection via {intersect_pt_mm} (mm): crosses corridor.")
                 continue

            intersect_tuple = tuple(intersect_pt_mm)
            if intersect_tuple not in temp_goal_nodes:
                n_temp = len(self.waypoints)
                self.waypoints.append(intersect_pt_mm) # Add temp node (mm)
                self.graph[n_temp] = []
                temp_goal_nodes[intersect_tuple] = n_temp
            else:
                n_temp = temp_goal_nodes[intersect_tuple]

            # Modify graph (indices refer to self.waypoints mm list)
            if n_goal_actual not in self.graph[n_temp]: self.graph[n_temp].append(n_goal_actual)
            if not self.directed and n_temp not in self.graph[n_goal_actual]: self.graph[n_goal_actual].append(n_temp)
            if node1_idx not in self.graph[n_temp]: self.graph[n_temp].append(node1_idx)
            if node2_idx not in self.graph[n_temp]: self.graph[n_temp].append(node2_idx)
            if n_temp not in self.graph[node1_idx]: self.graph[node1_idx].append(n_temp)
            if n_temp not in self.graph[node2_idx]: self.graph[node2_idx].append(n_temp)
            if node2_idx in self.graph[node1_idx]: self.graph[node1_idx].remove(node2_idx)
            if node1_idx in self.graph[node2_idx]: self.graph[node2_idx].remove(node1_idx)

        # --- Direct Path Check (using PIXEL coordinates) ---
        if not self.crossing_coordiors(start_img, goal_img):
            # Add direct connection using INDICES of mm list
            if n_goal_actual not in self.graph[n_start_actual]: self.graph[n_start_actual].append(n_goal_actual)
            if not self.directed and n_start_actual not in self.graph[n_goal_actual]: self.graph[n_goal_actual].append(n_start_actual)
            # rospy.loginfo("Direct path connection added.")

        # --- Run A* (using indices of self.waypoints mm list) ---
        rospy.loginfo(f"Running A* from index {n_start_actual} to {n_goal_actual}")
        path_indices = self.astar(n_start_actual, n_goal_actual) # Assumes astar uses self.waypoints (mm)

        if not path_indices:
             rospy.logerr("A* failed to find a path.")
             self.path = []
             return # Return empty path on failure

        # --- Convert final path indices to REAL coordinates in METERS ---
        path_real_coord_m = []
        for idx in path_indices:
            if 0 <= idx < len(self.waypoints):
                # Convert waypoint (mm) to meters
                wp_mm = self.waypoints[idx]
                path_real_coord_m.append((wp_mm[0] / 1000.0, wp_mm[1] / 1000.0))
            else:
                rospy.logwarn(f"Path index {idx} out of bounds (size {len(self.waypoints)}). Skipping.")

        # --- Plotting (Optional, uses PIXEL coordinates) ---
        if visualisation:
             path_in_img_coord = [real_to_img_convert(p[0], p[1]) for p in path_real_coord_m]
             self.plot_path(path_in_img_coord) # plot_path expects pixel coords

        self.path = path_real_coord_m # Store final path in METERS
        rospy.loginfo(f"Path calculation successful. Path length: {len(self.path)} points.")

    # --- PASTE YOUR ORIGINAL plot_road_map here ---
    def plot_road_map(self):
        '''Plot function to plot the predefined graph on to the image'''
        global map_image, visualisation
        if not visualisation or map_image is None: return
        X = []
        Y = []
        fig_num = 1
        if not plt.fignum_exists(fig_num): plt.figure(fig_num); plt.ion()
        else: plt.figure(fig_num); plt.clf()
        ax = plt.gca()

        # Convert original mm waypoints to pixels for plotting
        road_map_img = [ real_to_img_convert(i[0]/1000.0, i[1]/1000.0) for i in self.original_waypoints_mm]
        ax.imshow(map_image, extent=[0, map_image.shape[1], -map_image.shape[0], 0])

        for i in range(len(road_map_img)):
            x,y = road_map_img[i][0],road_map_img[i][1]
            X.append(x); Y.append(-y)
            ax.text(x,-y,f' {i}',fontsize=6, ha='left',va = 'bottom') # Use index i

        ax.scatter(X, Y, marker='.')

        plotted_edges = set()
        for node1_idx, neighbors in self.original_graph.items(): # Use original graph
             if node1_idx >= len(road_map_img): continue
             x1 , y1 = road_map_img[node1_idx][0],-road_map_img[node1_idx][1]
             for node2_idx in neighbors:
                  if node2_idx >= len(road_map_img): continue
                  edge = tuple(sorted((node1_idx, node2_idx)))
                  if edge in plotted_edges and not self.directed: continue
                  plotted_edges.add(edge)
                  x2 , y2 = road_map_img[node2_idx][0],-road_map_img[node2_idx][1]
                  ax.plot([x1,x2],[y1,y2],linestyle = 'dashed',color = 'c')
        plt.draw(); plt.pause(0.1)

    # --- PASTE YOUR ORIGINAL astar method here ---
    # Ensure it uses self.waypoints (which is the mm list) for distances
    def astar(self, start_idx, goal_idx):
        # Heuristic uses self.waypoints (mm)
        def heuristic(node_idx_h):
            if not (0 <= node_idx_h < len(self.waypoints) and 0 <= goal_idx < len(self.waypoints)):
                return float('inf')
            wp_node = np.array(self.waypoints[node_idx_h])
            wp_goal = np.array(self.waypoints[goal_idx])
            return np.linalg.norm(wp_node - wp_goal) # Distance in mm

        queue = []; cost = {start_idx: 0}; parent = {start_idx: None}; visited = set()
        start_priority = cost[start_idx] + heuristic(start_idx)
        heapq.heappush(queue, (start_priority, cost[start_idx], start_idx))
        final_node_idx = -1

        while queue:
            priority, current_cost, current_idx = heapq.heappop(queue)
            if current_idx in visited: continue
            visited.add(current_idx)
            if current_idx == goal_idx:
                final_node_idx = current_idx; break
            if current_idx not in self.graph: continue

            for neighbor_idx in self.graph[current_idx]:
                if not (0 <= neighbor_idx < len(self.waypoints)): continue
                wp_current = np.array(self.waypoints[current_idx])
                wp_neighbor = np.array(self.waypoints[neighbor_idx])
                edge_cost = np.linalg.norm(wp_neighbor - wp_current) # Cost in mm
                new_neighbor_cost = current_cost + edge_cost
                if neighbor_idx not in cost or new_neighbor_cost < cost[neighbor_idx]:
                    cost[neighbor_idx] = new_neighbor_cost
                    new_priority = new_neighbor_cost + heuristic(neighbor_idx)
                    heapq.heappush(queue, (new_priority, new_neighbor_cost, neighbor_idx))
                    parent[neighbor_idx] = current_idx

        path = []
        if final_node_idx == goal_idx:
            temp_idx = goal_idx
            while temp_idx is not None:
                path.append(temp_idx)
                if temp_idx not in parent:
                     rospy.logerr("A* Parent error."); return []
                temp_idx = parent[temp_idx]
            path.reverse()
        else:
            rospy.logwarn('A*: Goal not reachable.') # The warning you saw
            return []
        return path
# --- END OF global_planner CLASS ---

# --- ROS Related Functions ---

def get_start_pt_cb(msg):
    """Callback function for getting the position of the robot (METERS)."""
    global start_point, start_available, current_robot_pose, data_lock
    if msg.pose and msg.pose.pose and msg.pose.pose.position:
        with data_lock:
            x_robot_m = msg.pose.pose.position.x
            y_robot_m = msg.pose.pose.position.y
            start_point = (x_robot_m, y_robot_m) # Store start in meters
            current_robot_pose = msg.pose.pose
            if not start_available:
                rospy.loginfo(f"Received initial robot pose (m): {start_point}")
                start_available = True
    else:
        rospy.logwarn("Incomplete pose msg in EKF callback.")

def load_map_params(base_str):
    """Loads roadmap waypoints (mm) and graph from YAML."""
    rp = rospkg.RosPack()
    base_path = rp.get_path(base_str)
    # --- Use the ORIGINAL YAML Filename ---
    yaml_file = os.path.join(base_path, "config", "road_map_5624.yaml")
    try:
        with open(yaml_file, 'r') as stream:
            path_params_list = list(yaml.safe_load_all(stream))
            if len(path_params_list) < 3: raise ValueError("Expected 3 docs")
            if 'waypoints' not in path_params_list[0] or \
               'graph_directed' not in path_params_list[1] or \
               'graph_undirected' not in path_params_list[2]: raise ValueError("Missing keys")
            # Return waypoints as loaded (mm)
            return path_params_list[0]['waypoints'], path_params_list[1]['graph_directed'], path_params_list[2]['graph_undirected']
    except Exception as e:
        rospy.logerr(f"Error loading/parsing map params from {yaml_file}: {e}")
        return None, None, None

def load_image(base_str, image_filename="map_new.jpeg"):
     """Loads the map image using cv2."""
     rp = rospkg.RosPack()
     pkg_path = rp.get_path(base_str)
     # --- Use ORIGINAL Image Path ---
     file_path = os.path.join(pkg_path, 'scripts', image_filename) # Adjust if in 'maps'
     try:
        img = cv2.imread(file_path)
        if img is None: raise FileNotFoundError(f"Image not found: {file_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb
     except Exception as e:
          rospy.logerr(f"Error loading map image '{image_filename}': {e}")
          return None

def publish_path(path_real_coords_m):
    """Publishes the calculated path (in METERS) as a PoseArray."""
    global global_path_pub
    if global_path_pub is None: return

    path_msg = PoseArray()
    path_msg.header.stamp = rospy.Time.now()
    path_msg.header.frame_id = "map"
    path_pts = []
    if path_real_coords_m:
        for pt_m in path_real_coords_m:
            pose = Pose(); pose.position.x=pt_m[0]; pose.position.y=pt_m[1]; pose.orientation.w=1.0
            path_pts.append(pose)
        path_msg.poses = path_pts
        rospy.loginfo(f"Published global path with {len(path_pts)} points.")
    else:
        rospy.logwarn("Publishing empty path.")
    global_path_pub.publish(path_msg)

def plan_and_publish_wrapper(target_goal_real_m):
     """Wrapper to call planner, publish path, set goal active flag."""
     global planner_instance, start_point, goal_active, data_lock

     if planner_instance is None: rospy.logerr("Planner not initialized."); return False

     with data_lock: current_start_m = start_point # Read start safely (meters)

     # Update planner's start/goal (meters) for this run
     planner_instance.start_m = current_start_m
     planner_instance.goal_m = target_goal_real_m
     planner_instance.calculate_path() # Calculates and stores path in self.path (meters)

     if planner_instance.path:
          publish_path(planner_instance.path)
          with data_lock: goal_active = True
          return True
     else: # Path calculation failed
          with data_lock: goal_active = False
          return False

# --- Service Handler ---
def handle_set_pixel_goal(req):
    """Service handler for SetPixelGoal."""
    global goal_point_real, goal_active, current_robot_pose, data_lock

    rospy.loginfo(f"Received SetPixelGoal request: pixel_x={req.pixel_x}, pixel_y={req.pixel_y}")
    with data_lock:
        goal_active = False # Cancel any previous goal monitoring

    # Convert pixel goal to real-world coordinates (meters)
    try:
        target_goal_real_m = img_to_real_convert(req.pixel_x, req.pixel_y)
        rospy.loginfo(f"Converted pixel goal to real goal (m): {target_goal_real_m}")
        with data_lock:
             goal_point_real = target_goal_real_m # Store the official target
    except Exception as e:
        rospy.logerr(f"Error converting pixel coordinates: {e}")
        return SetPixelGoalResponse(reached_goal=False, message=f"Coordinate conversion error: {e}")

    # Plan and publish the path
    if not plan_and_publish_wrapper(target_goal_real_m):
        rospy.logerr("Failed to plan path for service request.")
        # goal_active is already False
        return SetPixelGoalResponse(reached_goal=False, message="Path planning failed")

    # --- Monitor robot's arrival ---
    rospy.loginfo(f"Path published. Monitoring robot arrival at {target_goal_real_m}...")
    rate = rospy.Rate(5) # Check ~5 times per second
    start_time = rospy.Time.now()

    while not rospy.is_shutdown():
        # 1. Check for timeout
        if (rospy.Time.now() - start_time) > rospy.Duration(SERVICE_TIMEOUT):
            rospy.logwarn(f"Service timeout ({SERVICE_TIMEOUT}s) waiting for goal arrival.")
            with data_lock: goal_active = False
            return SetPixelGoalResponse(reached_goal=False, message="Timeout waiting for arrival")

        # 2. Get current state safely
        with data_lock:
            if not goal_active: # Check if cancelled by a new goal
                 rospy.logwarn("Goal monitoring cancelled (goal_active flag became False).")
                 return SetPixelGoalResponse(reached_goal=False, message="Navigation cancelled")
            current_pos_real_m = (current_robot_pose.position.x, current_robot_pose.position.y)
            # Use the goal specific to this service call
            current_target_goal_m = target_goal_real_m

        # 3. Calculate distance (meters)
        distance_to_goal = math.hypot(current_pos_real_m[0] - current_target_goal_m[0],
                                      current_pos_real_m[1] - current_target_goal_m[1])

        # 4. Check if goal is reached
        if distance_to_goal < GOAL_REACHED_THRESHOLD:
            rospy.loginfo(f"Goal {current_target_goal_m} reached! Dist: {distance_to_goal:.3f}m")
            with data_lock: goal_active = False
            return SetPixelGoalResponse(reached_goal=True, message="Goal reached successfully")

        # 5. Wait
        try:
            rate.sleep()
        except rospy.ROSInterruptException:
            rospy.logwarn("ROS interrupt received during goal monitoring.")
            # Let shutdown logic handle cleanup

    # If loop exits due to rospy.is_shutdown()
    rospy.logwarn("Goal monitoring stopped due to ROS shutdown.")
    with data_lock: goal_active = False
    return None # Service handler shouldn't return if node is shutting down


# --- Click Handler (REMOVED) ---
# def onclick(event): ... # REMOVED

def turn_off():
    """Graceful shutdown."""
    print("Shutting down global_path_planner_node")
    plt.close('all') # Close any matplotlib figures

# --- Main Execution ---
def main():
    global visualisation, start_point, start_available, planner_instance, global_path_pub, map_image
    global original_waypoints_mm, original_graph_directed, original_graph_undirected, current_graph_type

    rospy.init_node('global_path_planner_node')

    # --- Parameter Reading ---
    # REMOVED: use_click_mode parameter - always run in service mode now
    use_directed = rospy.get_param("~use_directed", False) # Keep this
    visualisation = rospy.get_param("~visualisation", False)
    rospy.loginfo(f"Running in Service-Only mode.")
    rospy.loginfo(f"Using {'DIRECTED' if use_directed else 'UNDIRECTED'} graph.")
    if visualisation: rospy.loginfo("Visualization enabled.")

    # --- Load Roadmap Data (Originals in mm) ---
    original_waypoints_mm, original_graph_directed, original_graph_undirected = load_map_params('global_path_planning')
    if original_waypoints_mm is None:
        rospy.logfatal("Failed to load roadmap parameters. Shutting down."); return
    current_graph_structure = original_graph_directed if use_directed else original_graph_undirected
    current_graph_type = 'directed' if use_directed else 'undirected'

    # --- Initialize ROS Comms ---
    global_path_pub = rospy.Publisher("/global_path", PoseArray, queue_size=1, latch=True)
    rospy.Subscriber("/pose_ekf", PoseWithCovarianceStamped, get_start_pt_cb)

    # --- Load Map Image ---
    map_image = load_image('global_path_planning')
    if map_image is None:
         rospy.logfatal("Failed to load map image. Shutting down."); return

    # --- Wait for Start Pose ---
    rospy.loginfo("Waiting for initial robot pose...")
    while not start_available and not rospy.is_shutdown():
        rospy.sleep(0.5)
    if rospy.is_shutdown(): return
    rospy.loginfo(f"Initial robot pose received (m): {start_point}")

    # --- Initialize Planner (using original mm waypoints) ---
    planner_instance = global_planner(
        start_point, # Initial start (meters)
        (0.0, 0.0),  # Dummy goal (meters)
        original_waypoints_mm, # Pass original mm waypoints
        use_directed,
        current_graph_structure,
        map_image
    )

    # --- Setup Service Mode ---
    rospy.loginfo("Initializing SetPixelGoal service...")
    try:
        s = rospy.Service('set_pixel_goal', SetPixelGoal, handle_set_pixel_goal)
        rospy.loginfo("SetPixelGoal service ready.")
    except Exception as e:
        rospy.logerr(f"Failed to initialize SetPixelGoal service: {e}", exc_info=True)
        if not rospy.is_shutdown(): rospy.signal_shutdown("Service init failed")
        return

    # --- Setup Visualization Window (if enabled) ---
    if visualisation:
        rospy.loginfo("Visualization enabled. Setting up plot window.")
        try:
            plt.ion()
            fig, ax = plt.subplots(num=1) # Create figure 1
            ax.set_title("Global Path Planner Visualization (Service Mode)")
            # Initial display (can be updated later in plot_path)
            ax.imshow(map_image, extent=[0, map_image.shape[1], -map_image.shape[0], 0])
            start_img_coords = real_to_img_convert(start_point[0], start_point[1])
            ax.plot(start_img_coords[0], -start_img_coords[1], 'go', markersize=8, label='Start')
            plt.legend(); plt.draw(); plt.pause(0.01)
        except Exception as e:
            rospy.logwarn(f"Error setting up visualization: {e}. Continuing without plot.")
            visualisation = False # Disable viz if setup fails

    # Use on_shutdown for cleanup
    rospy.on_shutdown(turn_off)

    # Keep the node alive to handle callbacks and service requests
    rospy.loginfo("Node spinning, waiting for service calls...")
    rospy.spin()

    rospy.loginfo("Global planner node finished.")


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS Interrupt received. Shutting down.")
    except Exception as e:
         rospy.logfatal(f"Unhandled exception in main: {e}", exc_info=True)
    finally:
         plt.close('all') # Ensure plots are closed