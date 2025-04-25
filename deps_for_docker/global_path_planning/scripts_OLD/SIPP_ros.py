#!/usr/bin/env python3

import os
import copy
import math

import numpy as np
import heapq
import cv2
import time
import pygame
from math import cos,sin,radians,sqrt,degrees
import matplotlib.pyplot as plt
import tf
import rospy

from geometry_msgs.msg import PoseArray,PoseWithCovarianceStamped,Pose
from tracking.msg import PoseIDArray

from global_path_planning.msg import PoseFloat, PoseFloatArray

#Global variables

global_path = []
obstacles = []
obstacle_available = False

v = 10
vel_real = 900 #calculated by moving robot in a speed i think we should move it
dtheta = 30
ddtheta = 10
primitive_reduced = 2
primitive_reset = 0

x_offset = 93 # x offset to adjust with the map, home coordinates
y_offset = 200 # y offset to adjust with the map, home coordinates
scale_x = 60.30 # scale for x values to adjust with the map
scale_y = 60.30
collision_radius = 13
tas = []



XDIM = 795
YDIM = 661
WINSIZE = [XDIM, YDIM]
EPSILON = 7.0
NUMNODES = 5000


waypoints = [[22733.1, 6150.6],  # dummy would not be accessed in any case
             [22733.1, 6150.599999999999], [22190.399999999998, 1085.3999999999999], [1085.3999999999999, 1628.1],
             [1085.3999999999999, 20381.399999999998], [1085.3999999999999, 23094.899999999998],
             [1085.3999999999999, 33828.299999999996], [-7959.599999999999, 33828.299999999996],
             [21828.6, 6150.599999999999], [21828.6, 2231.1], [2291.4, 2170.7999999999997], [1869.3, 22009.5],
             [13627.8, 22009.5], [20079.899999999998, 22009.5], [20079.899999999998, 23094.899999999998],
             [20079.899999999998, 28220.399999999998], [13627.8, 23094.899999999998], [13386.599999999999, 29969.1],
             [24903.899999999998, 29969.1], [25386.3, 33828.299999999996], [25386.3, 39617.1],
             [13748.4, 33828.299999999996], [12421.8, 33285.6], [12723.3, 23456.699999999997], [2291.4, 23034.6],
             [1869.3, 33104.7]]

# This
# graph = {
#             1: [2],  2: [3],  3: [4],  4: [5],  5: [6, 24],
#             6: [7, 25],  7: [6],  8: [],  9: [8],  10: [9],
#             11: [5, 10],  12: [11, 23],  13: [12],  14: [13, 15],  15: [14],
#             16: [12, 14],  17: [16, 22],  18: [17],  19: [18, 20],
#             20: [19],  21: [17, 19],  22: [21, 25], 23: [16, 22],
#             24: [11, 23],  25: [6, 24]
#         }


# This below is bidirectional
graph = {
    1: [2], 2: [1, 3], 3: [2, 4], 4: [3, 5], 5: [4, 6, 24, 11],
    6: [5, 7, 21, 25], 7: [6], 8: [9], 9: [8, 10], 10: [9, 11],
    11: [10, 5, 24, 12], 12: [11, 23, 13, 16], 13: [12, 14], 14: [13, 15, 16],
    15: [14], 16: [12, 14, 17, 23], 17: [16, 22, 18, 21], 18: [17, 19],
    19: [18, 20, 21], 20: [19], 21: [17, 19, 22, 6], 22: [21, 25, 17, 23],
    23: [16, 22, 12, 24], 24: [11, 23, 5, 25], 25: [6, 24, 22]
}



#Classes

class state(object):
    def __init__(self, cfg = (-1, -1, 3.14), sf_idx = 0 , time = 0):
        self.cfg = cfg
        self.sf_idx = sf_idx
        self.time = time
    # When cost same in the priority queue compares the state object
    #So just return the first object used in comparision
    def __lt__(self, other):
        return True
    

#Functions

def img_to_real_convert(x,y):
    global scale_y,scale_x
    X = 0 + (y - y_offset)*scale_x
    Y = 0 + (x - x_offset)*scale_y
    return [X,Y]

def real_to_img_convert(x,y):
    global scale_y,scale_x
    X = y/scale_y + x_offset
    Y = x/scale_x + y_offset
    return (X,Y)


def get_start_pt_cb(msg):
    '''Callback function for getting the position of the robot'''
    global start_point, start_available
    x_robot = msg.pose.pose.position.x  # / 1000
    y_robot = msg.pose.pose.position.y  # / 1000
    # Need to get orientation of robot also and convert it to degrees
    orientation_list = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z,
                        msg.pose.pose.orientation.w]
    euler = tf.transformations.euler_from_quaternion(orientation_list)
    theta = degrees(euler[2])

    start_point = (x_robot, y_robot, theta - 90)
    start_available = True

    # rospy.logerr(f'In callback, {start_point}')

def collision(safe_interval , state):             #Generates a safe interval for state ( cfg[:2])
    """
    Collision check only done along the route of the human.
    Basically the check of collision radius gets us points that can cause collision along the path of obstalce with state
    Create safe intervals for these points alone, Then going over this should give us collision free path ?
    :param safe_interval:
    :param state:
    :return:
    """
    default_safe_int = [[0, float('inf')]]
    intersection_pts = []
    for i in safe_interval:
        if math.sqrt( (i[0] - state[0])**2 + (i[1] - state[1])**2 ) < collision_radius:
            intersection_pts.append(safe_interval[i])
    if not intersection_pts:
        return default_safe_int
    else:
        c = []
        a = {} #Dummy state interval
        for cfg_interval in intersection_pts:
            if cfg_interval[0][0] != 0:
                c.append([0,cfg_interval[0][0]])
            for i in range(len(cfg_interval) - 1):
                c.append([cfg_interval[i][1] + 1, cfg_interval[i + 1][0] - 1])
        for i in c:
            for j in range(i[0],i[1]+1):
                safe_interval_create(j, a, state)
        return a[state]



def pDistance(x, y, x1, y1, x2, y2) :
  A = x - x1
  B = y - y1
  C = x2 - x1
  D = y2 - y1

  dot = A * C + B * D
  len_sq = C * C + D * D
  param = -1
  if  not len_sq == 0: #in case of 0 length line
      param = dot / len_sq

  if param < 0:
    xx = x1
    yy = y1
  elif param > 1:
    xx = x2
    yy = y2
  else:
    xx = x1 + param * C
    yy = y1 + param * D

  dx = x - xx
  dy = y - yy
  return sqrt(dx * dx + dy * dy)

def xte(pt :list): #[x ,y, theta]
    global global_path
    x, y, theta = pt[0], pt[1], pt[2]
    dist = float('inf')
    
    for i in range(len(global_path)-1):
        x1,y1,x2,y2 = global_path[i][0], global_path[i][1] , global_path[i+1][0] , global_path[i+1][1]
        dist = min(pDistance(x,y,x1,y1,x2,y2), dist)
    return dist

def angle_manage(theta):
    if theta < 0:
        theta = theta + 360
    elif theta >= 360:
        theta = theta - 360
    return theta

def SIPP_neighbor(s,safe_interval_obstacle , safe_interval): #s dict and would contain (x,y,theta), i -> safe interval, t
    #print(safe_interval)
    x , y ,theta = s.cfg
    #s_grid = copy.deepcopy(s)
    #s_grid.cfg = (s.cfg[0]//collision_radius*collision_radius , s.cfg[1]//collision_radius*collision_radius ,s.cfg[2])
    successors = []
    successors_grid = []

    neighbors_grid = neighbors_mp_grid(x, -y, theta)
    neighbors = neighbors_mp(x, -y, theta)

    default_safe_int = [[0, float('inf')]]
    # Maybe call neighbors here
    for k ,cfg in enumerate(neighbors):
        # cfg_grid = copy.deepcopy(cfg)
        cfg_grid = (cfg[0] // collision_radius * collision_radius, cfg[1] // collision_radius * collision_radius, cfg[2])
        #print(cfg_grid)
        # m_time = np.linalg.norm(np.array(s.cfg[:2]) - np.array(cfg[:2]) ) #/v  or cfg[-1] if cost included in primitive
        m_time = cfg[3]
        start_t = s.time + m_time
        #print('current_time ' , s.time)
        # end_t = safe_interval.get(s_grid.cfg[:2], default_safe_int)[s.sf_idx][1] + m_time
        end_t = safe_interval.get(s.cfg[:2])[s.sf_idx][1] + m_time
        a = collision(safe_interval_obstacle, cfg[:2])
        for idx, i in enumerate(a):
            # if cfg_grid[:2] == (300,225):
            #     #print(f'Interval is {i} , time {start_t}, end time {end_t}')
            if i[0] > end_t or i[1] < start_t:
               # print('continuing')
                continue

            t = max(start_t, i[0])
            # if t is None:
            #     continue
            #print('Time:', start_t , i[0])
            s_grid = state(neighbors_grid[k][:3], idx, t)
            successors_grid.append(s_grid)
            #successors_grid.append(neighbors_grid[k][:3])
            s_ = state(cfg[:3], idx, t) #exclude cost
            successors.append((s_,cfg[3]))
            safe_interval[s_.cfg[:2]] = a
    return successors , successors_grid

def safe_interval_create(t, safe_interval , cfg): #si_cfg : safe intervals of the configuration i.e state
    si_cfg = safe_interval.get(cfg)
    if si_cfg is None:
        si_cfg = [[0, t-1],[t+1, float('inf')]]
        safe_interval[cfg] = si_cfg
    for i, sf in enumerate(si_cfg):
        if sf[0] < t < sf[1]:
            si_cfg.pop(i)
            #idx = i
            #if sf[0] != t -1:
            si_cfg.insert(i , [sf[0], t-1])
            si_cfg.insert(i+1, [t + 1, sf[1]])
             #   idx += 1
            break
        #if sf[1] != t + 1:
        elif sf[0]== t:
            if sf[1] >= t + 1:
                sf[0] = t+1
            else:
                si_cfg.pop(i)
            break
        elif sf[1] == t:
            if sf[0] <= t - 1:
                sf[1] = t-1
            else:
                si_cfg.pop(i)
            break
    #print(safe_interval)

def neighbors_mp(x,y,theta):
    theta = angle_manage(theta)
    multiplier = v/(vel_real/scale_x)  #While changing v accordingly change the cost, hence using multiplier is used here
    forward = (x + v * cos(radians(theta)), -(y + v * sin(radians(theta))), theta, 1*multiplier)
    left_t =  (x + v * cos(radians(theta+dtheta/2)), -(y + v * sin(radians(theta+dtheta/2))), theta + dtheta, 1*multiplier )
    right_t = (x + v * cos(radians(theta-dtheta/2)), -(y + v * sin(radians(theta-dtheta/2))), theta - dtheta, 1*multiplier )
    neighbors = [forward, left_t, right_t]
    return neighbors #should return neighbour in the form (x,y,theta)

def neighbors_mp_grid(x,y,theta):
    thres = 2
    theta = angle_manage(theta)
    forward = (round((x + v * cos(radians(theta)))/thres)*thres, round((-(y + v * sin(radians(theta))))/thres)*thres, theta)
    left_t = (round((x + v * cos(radians(theta+dtheta/2)))/thres)*thres, round((-(y + v * sin(radians(theta+dtheta/2))))/thres)*thres, theta + dtheta )
    right_t = (round((x + v * cos(radians(theta-dtheta/2)))/thres)*thres, round((-(y + v * sin(radians(theta-dtheta/2))))/thres)*thres, theta - dtheta)
    neighbors = [forward,left_t,right_t]
    return neighbors #should return neighbour in the form (x,y,theta)

def astar(map_array, start, goal, obstacles):
    #Initialise the Obstacle trajectory thingy
    #safe_interval = {state:[[],[]]}
    global v, primitive_reset, scale_x

    animation = False
    #global obstacle
    #Pygame for graph visualization:
    if animation:
        pygame.init()
        screen = pygame.display.set_mode(WINSIZE)
        img = pygame.image.load('map_new.jpeg')
        img.convert()

        pygame.display.set_caption('RRT      S. LaValle    May 2011')
        white = 255, 240, 200
        black = 20, 20, 40
        RED = (255, 0, 0)
        screen.fill(black)
        rect = img.get_rect()
        rect.center = XDIM // 2, YDIM // 2
        screen.blit(img, rect)
        pygame.draw.rect(screen, RED, rect, 1)
        pygame.display.update()
        #print("from A*", obstacle)
        #A*
    start_t = time.time_ns()
    height, width = map_array.shape


    def heuristic(node):
        return np.linalg.norm(np.array(node) - np.array(goal[:2]))



    # Initializing
    queue = []
    cost = {start.cfg: 0}
    gm = {start: start}  # maps points grid ones to crct/actual primitive
    heapq.heappush(queue, (start.time, start))
    parent = {start: None}
    safe_interval_obstacle = {} #form (x,y,theta): [[],[],[]]
    safe_interval = {start.cfg[:2]: [[0,float('inf')]]}
    #Curr posn (x,y,theta)
    
    for obstacle in obstacles:
        #traj = bill(obstacle)# For simulation
        #traj = human(obstacle)
        for i, pos in enumerate(obstacle):
            safe_interval_create((i+1),safe_interval_obstacle,pos)
        #print(safe_interval)
    while queue:
        _, current = heapq.heappop(queue)
        current_gm = gm[current]
        if parent[current] == None:
            pass
        else:
            if animation:
                pygame.draw.line(screen, black, parent[current].cfg[:2], current.cfg[:2])
                pygame.display.update()

        if current.cfg == goal or cost[current_gm.cfg] > 200:
            goal = current
            #print('goal')
            break

        if heuristic(current_gm.cfg[:2]) < v/2 and abs(current_gm.cfg[2] - goal[2]) <= dtheta :
            goal = current
            #print('thresholded')
            break

        neighbors = []

        x, y, theta = current_gm.cfg

        neighbors, neighbors_grid = SIPP_neighbor(current_gm , safe_interval_obstacle, safe_interval)

        if primitive_reset == 5:
            v = 10
        else:
            primitive_reset += 1

        for i,neighbor in enumerate(neighbors):

            # if map_array[neighbor[0]][neighbor[1]] == 0:
            #     continue
            # Calculate the cost of reaching the neighbor
            if 0 <= neighbor[0].cfg[0] < width and 0 <= neighbor[0].cfg[1] < height:
                #Check if obstacle
                if map_array[int(neighbor[0].cfg[1])][int(neighbor[0].cfg[0])] < 200 :
                    continue

                # elif np.sqrt((neighbor.cfg[0]-obstacle[0])**2 + (neighbor.cfg[1]-obstacle[1])**2) < 13.0:
                #     # print(neighbor,obstacle,'Collision with Bill')
                #     continue

            cross_track_error = xte(current_gm.cfg)
            #print(neighbor)
            #include time and also why is the other nodes not getting expanded (curr.time - neighbor.time)
            if neighbor[0].time - current_gm.time < 0:
                raise Exception('The Time machine worked')
            # np.linalg.norm(np.array(neighbor.cfg[:2]) - np.array(current_gm.cfg[:2])) + \
            #print(goal)
            neighbor_cost =  cost[current_gm.cfg] +  (neighbor[0].time - current_gm.time) + \
                              + abs(neighbor[0].cfg[2] - goal[2])/90 + 5*cross_track_error#+ (255 - map_array[int(neighbor[0].cfg[1])][int(neighbor[0].cfg[0])]) * (1 / 255)  + 15*cross_track_error #Increased cost for going near the wall
                # neighbor[1] -> Can be though as the cost to move there
            #(neighbor[0].time - current_gm.time) -> cost and also time taken to move next is also penalised
            if neighbors_grid[i].cfg not in cost or neighbor_cost < cost[neighbors_grid[i].cfg]:
                # g
                cost[neighbor[0].cfg] = neighbor_cost

                # f = g+h
                priority = neighbor_cost + heuristic(neighbor[0].cfg[:2])*(v/(vel_real/scale_x))# + 0.5*((abs(neighbor[2] - goal[2])%180)//dtheta)
                # adding new nodes to the q
                # heapq.heappush(queue, (priority, neighbors_grid[i][:3]))
                heapq.heappush(queue, (priority, neighbors_grid[i]))
                # connect node to its parent
                # parent[neighbors_grid[i][:3]] = current
                parent[neighbors_grid[i]] = current
                # gm[neighbors_grid[i][:3]] = neighbor[:3]
                gm[neighbors_grid[i]] = neighbor[0]
                #print('queue',queue)


    # Retrieve the path from the goal to the start
    path = []
    current = goal
    global primitive_reduced
    while current:
        
        try:
            path.append(gm[current])
            current = parent.get(current)
        except:
            v = 10
            if v - primitive_reduced >= 2 :
                v = v - primitive_reduced
                primitive_reset = 0
                print('Trying out a smaller motion primitive', v)
                primitive_reduced += 2
                return astar(map_array, start , goal, obstacles)
            else:
                print('No path')
                primitive_reduced = 0
                return []
    print(v,'Hello',primitive_reduced)
    primitive_reduced = 0
    path.reverse()
    return path


def get_obstalce_pt_cb(msgs):
    
    global obstacles, obstacle_available

    obstacles = []

    for msg in msgs.poses:
        obstacle_available = True
        x_obstacle = msg.pose.position.x  # / 1000
        y_obstacle = msg.pose.position.y  # / 1000
        v = msg.pose.position.z

        #print('v obstacle!!!!!!!!!!!!!!!:',v)

        # Need to get orientation of robot also and convert it to degrees
        # orientation_list = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z,
        #                     msg.pose.orientation.w]
        # euler = tf.transformations.euler_from_quaternion(orientation_list)
        theta = np.pi -  msg.pose.orientation.z

        # print('Obstacle ori in real frame :', theta)
        obstacle = (x_obstacle, y_obstacle, theta , v)


        obstacles.append(obstacle)
    

def intersect_pts(start1, angle1, start2 , end2, length = 100000):
    """
    Find the intersection point of two line segments AB and CD.
    Parameters:
    A, B : tuple
        Coordinates of the first line segment's endpoints (x, y).
    C, D : tuple
        Coordinates of the second line segment's endpoints (x, y).
    Returns:
    tuple
        Intersection point (x, y) if exists, else None if the segments don't intersect.
    """
    # angle_radians = math.radians(angle1)
    dx = length * math.cos(angle1)
    dy = length * math.sin(angle1)
    x1, y1 = start1
    x2, y2 = (x1 + dx, y1 + dy)
    x3, y3 = start2
    x4, y4 = end2
    # Calculate the denominators
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        # Lines are parallel or coincident
        return None
    # Calculate the numerators
    t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    u_num = (x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)
    # Calculate the parameters t and u
    t = t_num / denom
    u = -u_num / denom
    # Check if the intersection point is within the bounds of both segments
    if 0 <= t <= 1 and 0 <= u <= 1:
        # Calculate the intersection point
        intersection_x = x1 + t * (x2 - x1)
        intersection_y = y1 + t * (y2 - y1)
        # print('intersection found')
        return [(intersection_x, intersection_y)]
    # No intersection within the bounds of the segments
    # print('No Intersection Found &&&&&&&')
    return []

def base_line(wp1, wp2 , i_pt, robot_pos):
    global waypoints
    #Here all pts in real coordinates
    wp1_c = waypoints[wp1]
    wp2_c = waypoints[wp2]
    x1, y1 = robot_pos[0], robot_pos[1]
    v1 = [wp1_c[0] - i_pt[0] , wp1_c[1] - i_pt[1]]
    v2 = [wp2_c[0] - i_pt[0], wp2_c[1] - i_pt[1]]
    v3 = [x1 - i_pt[0] , y1 - i_pt[1]]
    if v1[0]*v3[0] + v1[1]*v3[1] > 0 :
        return wp1 , wp2
    else:
        return wp2 , wp1

def finding_nearest_node(robot_pos, pose, R):  # Needs current pose and Radius of search
    visited_line_seg = set()
    possible_start = []
    exp = set()
    # iterate through waypoints and make line segments and check
    Q = [1]
    # print('GRAPH', graph)
    count = 0
    while len(Q) != 0:
        curr = Q.pop(0)
        if curr not in exp:
            for i in graph[curr]:
                if (curr, i) in visited_line_seg or (i, curr) in visited_line_seg:
                    continue
                x1, y1 = waypoints[curr]
                x2, y2 = waypoints[i]
                visited_line_seg.add((curr, i))
                points = intersect_pts((robot_pos[0], robot_pos[1]), pose ,(x1, y1), (x2, y2) )
                count += 1
                # print('Counting' , count)
                if points:
                    # print('Points Points Poinst:', points)
                    for p in points:

                        # if angle(robot_pos, pose, p):
                        # print('p', p)
                        next_wp, prev_wp = base_line(curr , i ,p , robot_pos)
                        if len(points) == 2:
                            possible_start.append((prev_wp, p, next_wp, True))
                        else:
                            possible_start.append((prev_wp, p, next_wp))

            Q = Q + graph[curr]
            exp.add(curr)
    min_d = float('inf')
    min_d_point = None
    for idx, start in enumerate(possible_start):
        d = np.linalg.norm(np.array(start[1]) - np.array(robot_pos))
        if min_d > d:
            min_d_point = start
            min_d = d
    if min_d_point:
        return [min_d_point]
    else:
        return []

def intersect_pts_c(x1, y1, x2, y2, a, b, R):
    # Intersection between circle and line segment(parametric form) , we get quadratic in t
    A = (x2 - x1)**2 + (y2 - y1)**2
    B = 2 * (x1 - a)*(x2 - x1) + 2*(y1 - b) * (y2 - y1)
    C = (x1 - a)**2 + (y1 - b)**2 - R**2
    D = B**2 - 4 * A * C

    if D < 0:
        return None
    elif D == 0:
        x = x1 + (-B / (2 * A)) * (x2 - x1)
        y = y1 + (-B / (2 * A)) * (y2 - y1)
        return [(x, y)]
    else:
        t1 = (-B - math.sqrt(D)) / (2 * A)
        t2 = (-B + math.sqrt(D)) / (2 * A)
        t = [t1, t2]
        t = [i for i in t if 0 <= i <= 1]
        # i = (t[0] + t[1]) / 2
        pts = []
        for i in t:
            x = x1 + i * (x2 - x1)
            y = y1 + i * (y2 - y1)
            pts.append(([x, y]))  # ,(x2,y2),(x1,y1))
        return pts


def forward_propagate(start_pt, graph , waypoints, distance_thres=5000):
    # print('Inside FWD start_pt:', start_pt)
    visited = set()
    fringe = [(start_pt, 0)]
    parent = {}
    last_nodes = []
    while fringe:
        curr, dist = fringe.pop(0)
        if curr in visited:
            continue
        visited.add(curr)

        for i in graph[curr]:

            d_check = dist + math.sqrt(
                (waypoints[curr][0] - waypoints[i][0])** 2 + (waypoints[curr][1] - waypoints[i][1])** 2)
            if d_check == dist:
                fringe.append((i, d_check))
                print('Point on top of global map line segment itself')
                continue
            elif d_check < distance_thres and i not in visited:
                fringe.append((i, d_check))
                parent[tuple(waypoints[i]),d_check] = tuple(waypoints[curr]), dist
            elif i not in visited:
                # print('PPP')
                x1, y1 = waypoints[curr]
                x2, y2 = waypoints[i]
                final_pt = intersect_pts_c(x1, y1, x2, y2, x1, y1, distance_thres - dist)
                parent[tuple(final_pt[0]),distance_thres] = tuple(waypoints[curr]), dist
                last_nodes.append((tuple(final_pt[0]),distance_thres))
    # print(last_nodes)
    # print(parent)
    paths = [[] for i in last_nodes]
    
    for i in range(len(last_nodes)):
        print(i)
        temp = last_nodes[i]   
        while temp:
            # print(last_nodes[i][-1],'\n')
            paths[i].append(temp)
            temp = parent.get(temp)      
    return paths # Would contain all predicted path

def calculate_intermediary_points(points, v):
    intermediary_points = [points[0]]
    residual = 0  # This will hold any leftover length from the previous segment

    # Loop through each consecutive pair of points to form a line segment
    for i in range(len(points) - 1):
        start = np.array(points[i])
        end = np.array(points[i + 1])

        # Calculate the distance between the two points
        segment_length = np.linalg.norm(end - start)

        # Include any residual length from the previous segment
        total_length = segment_length + residual

        # Calculate the direction vector for the line segment
        direction = (end - start) / segment_length

        # Generate points starting from the current position including residual
        current_position = start + direction * residual
        distance_covered = residual

        # Calculate the number of intermediary points needed based on length v
        while distance_covered + v <= total_length:
            current_position = current_position + direction * v
            intermediary_points.append(current_position.tolist())
            distance_covered += v

        # Update residual length for the next segment
        residual = total_length - distance_covered
        # print('IIII')

    return intermediary_points


def calculate_path( obstacles :list):
    '''
    Calculates the possible paths for the obstacle (human) using the map information
    :param obstacles: list of obstacles pose (x,y,angle)
    :return: list of path for each obstacle (the path for single obstacle can again be a list of paths)
    '''
    # v should be in real coordinates
    obstacles_path = []
    # print('Detected ',len(obstacles), 'persons')
    for obstacle in obstacles:
        person_pos, ori,v = obstacle[:2], obstacle[2],obstacle[3]*1000
        print('Obstacle Detection :',person_pos, ori, v)
        if v == 0:
            return [[person_pos for _ in range(20)]]
        plt.ion()
        graph_temp = copy.deepcopy(graph)
        waypoints_temp = copy.deepcopy(waypoints)

        R = 2500

        """Check how pose is gotten and what needs to be done here""" #Check here
        # ori = ori + (np.pi / 2)  # This is correct and is in ori is in real coordinates


        #person pos needs to be a list
        a = finding_nearest_node(person_pos, ori, R)
        # print('a', person_pos, a)


        n_start = len(waypoints_temp)
        waypoints_temp.append(person_pos)
        itr = iter(range(len(a)))
        for i in itr:
            n = len(waypoints_temp)
            waypoints_temp.append(a[i][1])

            graph_temp[n] = [a[i][0]]
            graph_temp[a[i][2]].remove(a[i][0])
            graph_temp[a[i][0]].remove(a[i][2])


        start_nodes = []
        for i in range(n_start + 1, len(waypoints_temp)):
            start_nodes.append(i)
        graph_temp[n_start] = start_nodes

        paths = forward_propagate(n_start, graph_temp, waypoints_temp)

        # If NO paths use constant velocity model (when no path obstacle intial position is returned by forward propagate)
        if len(paths) == 0 or (len(paths) == 1 and len(paths[0]) == 1):
            v_x = v*cos(ori)
            v_y = v*sin(ori)
            traj = []
            for t in range(0,20): # Predict for 20 seconds
                traj.append((person_pos[0]+v_x*t , person_pos[1] + v_y*t))
                traj_img = [tuple(real_to_img_convert(i[0],i[1])) for i in traj]
            return [traj_img] #Needs to return list of list
            
        #Converting each individual pts to tuple . Y tho ??
        for i in range(len(paths)):
            path = paths[i]
            path = [tuple(j[0]) for j in path]
            paths[i] = path

        x_start, y_start = waypoints_temp[n_start]
        # print('PATHS ###' ,paths)
        final_path = []
        for idx,path in enumerate(paths):

            if math.sqrt( (x_start - path[0][0])**2 + (y_start - path[0][1])**2 ) < 3000:
                continue
            if path:

                path.reverse()
                print(path ,v)
                path = calculate_intermediary_points(path, v)
                
                # plot_path(b, img)
            final_path.append([tuple(real_to_img_convert(i[0], i[1])) for i in path]) 
        obstacles_path.extend(final_path)
        print('Obstacle Path returned')
        # print('Paths final iteration: ',final_path)
        # print('All possible obstacles',obstacles_path)
    return obstacles_path

def main():
    
    global obstacles, global_path
    obstacles_traj = [[]]
    rospy.init_node('local_planner')
    plt.ion()
    global_path_pose = rospy.wait_for_message('/global_path', PoseArray, timeout=5)
    # obstacle_paths_pub = rospy.Publish('/obstacle_path', PoseArrayArray, queue_size = 10)
    local_path_pub = rospy.Publisher('/local_path', PoseFloatArray, queue_size=10)
    rospy.Subscriber("/pose_ekf", PoseWithCovarianceStamped, get_start_pt_cb)
    # Have to convert to image coordinates
    rospy.Subscriber("/globalkalmanposeArray", PoseIDArray, get_obstalce_pt_cb,queue_size=1)
    waypoints = global_path_pose.poses
    obstacles_predicted_path_pub = rospy.Publisher('/obstacles_predicted_path', PoseArray, queue_size=10)

    rospy.sleep(0.2)  # So that start_point updated
    if waypoints:
        cwd_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(cwd_path, 'map_new.jpeg')  # finding full file path
        image = cv2.imread(file_path)
        # Use the cvtColor() function to grayscale the image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # cv2_imshow( gray_image)
        # Apply binary thresholding to create a binary image
        threshold_value = 210
        max_value = 255
        ret, binary_image = cv2.threshold(gray_image, threshold_value, max_value, cv2.THRESH_BINARY)
        im = np.asarray(binary_image)
        # box_blur_ker = (1/25)*np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]])
        box_blur_ker = (1 / 81) * np.ones((9, 9))
        Box_blur = cv2.filter2D(src=im, ddepth=-1, kernel=box_blur_ker)
        # cv2_imshow(Box_blur)
        imblur = np.asarray(Box_blur)
        map_array = imblur

        # wp - waypoints
        wp_img_coord = [list(real_to_img_convert(i.position.x, i.position.y))+[0] for i in waypoints]
        global_path = wp_img_coord
        #global_path = orientation_clc(wp_img_coord)
        # Start point orientation according to image
        global_path[0][2]=(angle_manage(start_point[2] - 90)) #Check if angle correction
        distance_threshold = rospy.get_param("distance_threshold")
        n = len(global_path)
        x, y = real_to_img_convert(start_point[0], start_point[1])
        start_point_img = x, y, start_point[2]  # //dtheta *dtheta
        goal_point = global_path[-1]

        print('Local Planner Running')
        
        
        # b = time.time()
        #print(f'Time taken to execute bill traj function is : {b - a:.3f}')
        # if bill1_traj == []:
        #     bill1_traj = billf(bill1, v=5)  # constant velocity model

        while not np.linalg.norm(np.array(start_point[:2]) - np.array([waypoints[-1].position.x, waypoints[
            -1].position.y])) < distance_threshold:  # In real world coordinates
            # Start point has to be checked for orientation and stuff
            
            if obstacles:
                # a = time.time()
                obstacles_traj = calculate_path(obstacles)
                # print('Time taken for forward prop is ', time.time() - a )
            # a = time.time()
            path = astar(map_array, state(start_point_img, 0, 0), goal_point, obstacles_traj)
            # print('Time taken for astar is ', time.time() - a)
            #astar(map_array, SIPP.state(start_state, 0, 0), goal_point, [bill1, bill2])
            
            lp_real_coord = [img_to_real_convert(i.cfg[0], i.cfg[1]) + [i.cfg[2]] for i in path]

                
            # Convert orientaions and then publish
            if len(path) > 1:
                # Storing the position as a pose array to be used by other nodes
                path_msg = PoseFloatArray()
                path_pts = []
                for i, pt in enumerate(lp_real_coord):
                    l_path_pts = PoseFloat()
                    l_path_pts.pose.position.x = pt[0]
                    l_path_pts.pose.position.y = pt[1]
                    # Is the correct orientation correct or enough:
                    l_path_pts.pose.orientation.z = pt[2]
                    l_path_pts.time = path[i].time
                    path_pts.append(l_path_pts)
                    path_msg.poses = path_pts
                    local_path_pub.publish(path_msg)
            
            if len(obstacles_traj) > 0:
                path_msg = PoseArray()
                obstacle_path_pts = []
                for obstacle_path in obstacles_traj:
                    for i,pt in enumerate(obstacle_path):
                        o_path_pts = Pose()
                        o_path_pts.position.x = pt[0]
                        o_path_pts.position.y = pt[1]
                        # Is the correct orientation correct or enough:
                        obstacle_path_pts.append(o_path_pts)
                path_msg.poses = obstacle_path_pts
                obstacles_predicted_path_pub.publish(path_msg)
            # Write code here for obstacle path publishing.
            # if obstacles_traj:
            #     path = Pose




            # x, y = real_to_img_convert(start_point[0], start_point[1])
            # ddtheta = 45
            # start_point_img = x, y, round(start_point[2] / ddtheta) * ddtheta

            rate = rospy.Rate(10)
            rate.sleep()

            if rospy.is_shutdown():
                print('Shutting Down local planner')
                break

        else:
            rospy.logerr('Global Plan not available')


if __name__ == '__main__':
    main()