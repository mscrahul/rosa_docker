#!/usr/bin/env python3
import copy
import numpy as np
import heapq
import cv2
import time
import rospy
import pygame
from math import cos,sin,radians,degrees, sqrt
import matplotlib.pyplot as plt
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseWithCovarianceStamped
import matplotlib.pyplot as plt
import tf
import os
from global_path_planning.msg import PoseFloat, PoseFloatArray
from tracking.msg import PoseIDArray
global_path = []
obstacles = [(0,0)]
human_img_coord = []
humans_real_coord = []
start_point = (0,0,0)
start_available = False
primitive_reduced = 2
primitive_reset = 0
collision_radius = 7
tas = []
#We are giving 0.25 in controller 
v = 10
vel_sim = 1
dtheta = 30


obstacles = [[122, 295] , [300,215]]
#obstacle = 290, 210

#image to real coordinates
x_offset = 93 # x offset to adjust with the map, home coordinates
y_offset = 200 # y offset to adjust with the map, home coordinates
scale_x = 60.30 # scale for x values to adjust with the map
scale_y = 60.30

#Pygame
XDIM = 795
YDIM = 661
WINSIZE = [XDIM, YDIM]
EPSILON = 7.0
NUMNODES = 5000

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


# def get_obstalce_pt_cb(msg):
#     global obstacle, collision_radius
#     x_obstalce = msg.pose.pose.position.x
#     y_obstacle = msg.pose.pose.position.y
#     collision_radius = 1
#     obstacle = (x_robot, y_robot )

class state(object):
    def __init__(self, cfg = (-1, -1, 3.14), sf_idx = 0 , time = 0):
        self.cfg = cfg
        self.sf_idx = 0
        self.time = time
    # When cost same in the priority queue compares the state object
    #So just return the first object used in comparision
    def __lt__(self, other):
        return True
    
    def __repr__(self) -> str:
        return f'Cfg: {self.cfg}, time: {self.time}'

def update_human():
    v_x = 0
    v_y = -1
    rate = 4
    global obstacles
    for obstacle in obstacles:
        obstacle[0] += v_x*(1/rate) 
        obstacle[1] += v_y*(1/rate) 

#Used for simulation
def human(curr_posn):
    '''Calculation of future traj is done here. Update models and stuff here'''
    v_x = 0
    v_y = -1
    trajectory = []
    for t in range(1,10):
        trajectory.append(( curr_posn[0] + v_x*t, curr_posn[1] + v_y*t))
        #trajectory.append(((curr_posn[0] + v_x*t)//collision_radius*collision_radius, (curr_posn[1] + v_y*t )//collision_radius*collision_radius))
        #
    return tuple(trajectory)

#Used for simulation
def bill(curr_posn, v = 1 ):
    x, y, theta = curr_posn
    v_x = v*cos(theta)
    v_y = -v*sin(theta)

    trajectory = []
    for t in range(1,10):
        trajectory.append(((( x + v_x*t)//collision_radius)*collision_radius, ((y + v_y*t )//collision_radius)*collision_radius))
        # trajectory.append(( x + v_x*t, y + v_y*t))
    return tuple(trajectory)
# plot_path(human(obstacle))


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
        if sqrt( (i[0] - state[0])**2 + (i[1] - state[1])**2 ) < collision_radius:
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

def orientation_clc(path):
    final_path = [[path[0][0], path[0][1]]]
    for i in range( 1, len(path)-1):   #We have start orientation hence from the next waypoint onwards we need the 
        x = path[i+1][0]- path[i][0]
        y = path[i+1][1] - path[i][1]
        theta = (np.arctan2(y,x)/ (np.pi))*180
        final_path.append([path[i][0], path[i][1] , theta]) # -ve because of the way the image coordinates is in
    final_path.append([path[i+1][0], path[i+1][1] , final_path[-1][2]]) # orientation of goal and the point before it same
    #     path[i].append(-theta) # - because of the way the image coordinates is in
    # path[len(path)-1].append(path[len(path)-2][2])
    return final_path

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
            wait_time = max(i[0] - start_t, 0)
            successors.append((s_,wait_time))
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
    multiplier = v/vel_sim  #While changing v accordingly change the cost, hence using multiplier is used here
    #vel_sim can be used to account for actual speed of robot in real world as well
    forward = (x + v * cos(radians(theta)), -(y + v * sin(radians(theta))), theta, 1*multiplier)
    left_t =  (x + v * cos(radians(theta+dtheta/2)), -(y + v * sin(radians(theta+dtheta/2))), theta + dtheta, 1.25*multiplier )
    right_t = (x + v * cos(radians(theta-dtheta/2)), -(y + v * sin(radians(theta-dtheta/2))), theta - dtheta, 1.25*multiplier )
    neighbors = [forward, left_t, right_t]
    return neighbors #should return neighbour in the form (x,y,theta)

def neighbors_mp_grid(x,y,theta):
    thres = 4
    theta = angle_manage(theta)
    forward = (round((x + v * cos(radians(theta)))/thres)*thres, round((-(y + v * sin(radians(theta))))/thres)*thres, theta)
    left_t = (round((x + v * cos(radians(theta+dtheta/2)))/thres)*thres, round((-(y + v * sin(radians(theta+dtheta/2))))/thres)*thres, theta + dtheta )
    right_t = (round((x + v * cos(radians(theta-dtheta/2)))/thres)*thres, round((-(y + v * sin(radians(theta-dtheta/2))))/thres)*thres, theta - dtheta)
    neighbors = [forward,left_t,right_t]
    return neighbors #should return neighbour in the form (x,y,theta)

'''Doesn't return cost'''
# def neighbors_mp(x,y,theta):
#     theta = angle_manage(theta)
#     forward = (x + v * cos(radians(theta)), -(y + v * sin(radians(theta))), theta)
#     left_t_1 = (x + v * cos(radians(theta + ddtheta / 2)), -(y + v * sin(radians(theta + ddtheta / 2))), theta + ddtheta )
#     right_t_1 = (x + v * cos(radians(theta - ddtheta / 2)), -(y + v * sin(radians(theta - ddtheta / 2))), theta - ddtheta )
#     left_t_2 =  (x + v * cos(radians(theta+dtheta/2)), -(y + v * sin(radians(theta+dtheta/2))), theta + dtheta )
#     right_t_2 = (x + v * cos(radians(theta-dtheta/2)), -(y + v * sin(radians(theta-dtheta/2))), theta - dtheta )
#     neighbors = [forward, left_t_1,right_t_1 ]
#
# #    neighbors = [forward, left_t_1, left_t_2, right_t_1, right_t_2 ]
#     return neighbors #should return neighbour in the form (x,y,theta)
#
# def neighbors_mp_grid(x,y,theta):
#     thres = 4
#     theta = angle_manage(theta)
#     forward = ( round((x + v * cos(radians(theta))) / thres) * thres, round((-(y + v * sin(radians(theta)))) / thres) * thres,theta)
#     left_t_1 = (round((x + v * cos(radians(theta + ddtheta / 2))) / thres) * thres,
#               round((-(y + v * sin(radians(theta + ddtheta / 2)))) / thres) * thres, theta + ddtheta)
#     right_t_1 = (round((x + v * cos(radians(theta - ddtheta / 2))) / thres) * thres,
#                round((-(y + v * sin(radians(theta - ddtheta / 2)))) / thres) * thres, theta - ddtheta)
#     left_t_2 = (round((x + v * cos(radians(theta + dtheta / 2))) / thres) * thres,round((-(y + v * sin(radians(theta + dtheta / 2)))) / thres) * thres, theta + dtheta)
#     right_t_2 = (round((x + v * cos(radians(theta - dtheta / 2))) / thres) * thres,round((-(y + v * sin(radians(theta - dtheta / 2)))) / thres) * thres, theta - dtheta)
#     neighbors = [forward, left_t_1, right_t_1]
# #   neighbors = [forward,left_t,right_t, left_t_2 ,right_t_2]
#     return neighbors #should return neighbour in the form (x,y,theta)




def astar(map_array, start, goal, obstacles = obstacles):
    #Initialise the Obstacle trajectory thingy
    #safe_interval = {state:[[],[]]}
    global v, primitive_reset
    animation = False
    #global obstacle
    #Pygame for graph visualization:
    if animation:
        pygame.init()
        cwd_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(cwd_path, 'map_new.jpeg') 
        screen = pygame.display.set_mode(WINSIZE)
        img = pygame.image.load(file_path)
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
    wait_time = {start:0}
    gm = {start: start}  # maps points grid ones to crct/actual primitive
    heapq.heappush(queue, (start.time, start))
    parent = {start: None}
    safe_interval_obstacle = {} #form (x,y,theta): [[],[],[]]
    safe_interval = {start.cfg[:2]: [[0,float('inf')]]}
    #Curr posn (x,y,theta)
    human_posn = ()
    for obstacle in obstacles:
        #traj = bill(obstacle)# For simulation
        traj = bill(obstacle)
        for i, pos in enumerate(traj):
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

        if current.cfg == goal or cost[current_gm.cfg] > 600:
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
	
	#Allows only 5 small nodes to be expanded and then starts using bigger primitive
        if primitive_reset == 15:
            v = 10
        else:
            primitive_reset += 1
        
        #If robot is unable to find a path even after 
        #if v <= 2:
         #   inplace_rotation = current[:2]
          #  neighbors_grid.apppend(current_gm


        for i,neighbor in enumerate(neighbors):

            # if map_array[neighbor[0]][neighbor[1]] == 0:
            #     continue
            # Calculate the cost of reaching the neighbor
            if 0 <= neighbor[0].cfg[0] < width and 0 <= neighbor[0].cfg[1] < height:
                #Check if obstacle
                if map_array[int(neighbor[0].cfg[1])][int(neighbor[0].cfg[0])] < 230 :
                    continue

                # elif np.sqrt((neighbor.cfg[0]-obstacle[0])*2 + (neighbor.cfg[1]-obstacle[1])*2) < 13.0:
                #     # print(neighbor,obstacle,'Collision with Bill')
                #     continue

            cross_track_error = xte(current_gm.cfg)
            #print(neighbor)
            #include time and also why is the other nodes not getting expanded (curr.time - neighbor.time)
            if neighbor[0].time - current_gm.time < 0:
                raise Exception('The Time machine worked')
            # np.linalg.norm(np.array(neighbor.cfg[:2]) - np.array(current_gm.cfg[:2])) + \
            neighbor_cost =  cost[current_gm.cfg] +  (neighbor[0].time - current_gm.time) + \
                              + abs(neighbor[0].cfg[2] - goal[2])/90 + 5*cross_track_error#+ (255 - map_array[int(neighbor[0].cfg[1])][int(neighbor[0].cfg[0])]) * (1 / 255)  + 15*cross_track_error #Increased cost for going near the wall careful when adjusting it since the algo might start behaving differently if the and the priority thing is not of the same scale. Have to match heuristic too
                # neighbor[1] -> Can be though as the cost to move there
            #(neighbor[0].time - current_gm.time) -> cost and also time taken to move next is also penalised
            if neighbors_grid[i].cfg not in cost or neighbor_cost < cost[neighbors_grid[i].cfg]:
                # g
                cost[neighbor[0].cfg] = neighbor_cost

                # f = g+h
                priority = neighbor_cost + heuristic(neighbor[0].cfg[:2])*(v/vel_sim) # + 0.5((abs(neighbor[2] - goal[2])%180)//dtheta)
                # adding new nodes to the q
                # heapq.heappush(queue, (priority, neighbors_grid[i][:3]))
                heapq.heappush(queue, (priority, neighbors_grid[i]))
                # connect node to its parent
                # parent[neighbors_grid[i][:3]] = current
                parent[neighbors_grid[i]] = current
                # gm[neighbors_grid[i][:3]] = neighbor[:3]
                gm[neighbors_grid[i]] = neighbor[0]
                #print('queue',queue)
                wait_time[neighbors_grid[i] ] = neighbor[1]

    # Retrieve the path from the goal to the start
    path = []
    # if len(queue) == 0:
    #     print('Appears that there are no possible path from here')
    #     return
    current = goal
    flag = 1
    global primitive_reduced
    while current:
        #print(v,'Hello',primitive_reduced)
        try:
            path.append((gm[current], wait_time[current]))
            current = parent.get(current)
        except:
            
            if v - primitive_reduced >= 2 :
                v = v - primitive_reduced
                primitive_reset = 0
                print('Trying out a smaller motion primitive', v)
                primitive_reduced += 2
                return astar(map_array, start , goal)
            else:
                print('No path')
                primitive_reduced = 0
                return []
    primitive_reduced = 0
    path.reverse()
    #print(path)
    return path

def get_obstacle(msg):
    poses = msg.poses
    global humans_real_coord, human_img_coord
    humans_real_coord  = [(i.pose.position.x, i.pose.position.y, i.pose.orientation.z) for i in poses]
    human_img_coord = [list(real_to_img_convert(i[0], i[1])).append(i[2]) for i in poses ] 
    
def main():
    global global_path
    rospy.init_node('local_planner')
    plt.ion()
    global_path_pose = rospy.wait_for_message('/global_path', PoseArray, timeout=5)
    obstacle = rospy.Subscriber('/PredictedPoses',PoseIDArray,  get_obstacle)
    obstacle_pub = rospy.Publisher('/obstacles', PoseArray, queue_size= 10)
    obstacle_paths_pub = rospy.Publisher('/obstacles_path', PoseArray, queue_size= 10)
    local_path_pub = rospy.Publisher('/local_path', PoseFloatArray, queue_size=10)
    rospy.Subscriber("/pose_ekf", PoseWithCovarianceStamped, get_start_pt_cb)
    # Have to convert to image coordinates
    #rospy.Subscriber("/pose", PoseWithCovarianceStamped, get_obstalce_pt_cb)
    waypoints = global_path_pose.poses

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
        im_gray = np.asarray(gray_image)
        # box_blur_ker = (1/25)*np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]])
        box_blur_ker = (1 / 81) * np.ones((9, 9))
        Box_blur = cv2.filter2D(src=im, ddepth=-1, kernel=box_blur_ker)
        # cv2_imshow(Box_blur)
        imblur = np.asarray(Box_blur)
        map_array = imblur

        img = plt.imread(file_path)

        # wp - waypoints
        wp_img_coord = [real_to_img_convert(i.position.x, i.position.y) for i in waypoints]
        global_path = orientation_clc(wp_img_coord)
        # Start point orientation according to image
        global_path[0].append(angle_manage(start_point[2] - 90))
        distance_threshold = rospy.get_param("distance_threshold")
        n = len(global_path)
        x, y = real_to_img_convert(start_point[0], start_point[1])
        start_point_img = x, y, start_point[2]  # //dtheta *dtheta
        goal_point = global_path[-1]

        print('Local Planner Running')
        global obstacles
        while not np.linalg.norm(np.array(start_point[:2]) - np.array([waypoints[-1].position.x, waypoints[-1].position.y])) < distance_threshold:  # In real world coordinates
            # Start point has to be checked for orientation and stuff
            path = astar(map_array, state(start_point_img, 0, 0), goal_point, obstacles)
            #astar(map_array, SIPP.state(start_state, 0, 0), goal_point, [bill1, bill2])
            lp_real_coord = [img_to_real_convert(i[0].cfg[0], i[0].cfg[1]) + [i[0].cfg[2]] for i in path]
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
                    l_path_pts.time = path[i][1]
                    path_pts.append(l_path_pts)

                path_msg.poses = path_pts
                local_path_pub.publish(path_msg)
            
            """Obstacle temporary publish"""
            obstacle_msg = PoseArray()
            obstacle_pts = []
            for obstacle in obstacles:
                ob_pts = Pose()
                ob_pts.position.x = obstacle[0]
                ob_pts.position.y = obstacle[1]
                # Is the correct orientation correct or enough:
                # l_path_pts.orientation.z = pt[2]
                #Try to get radius here
                obstacle_pts.append(ob_pts)

            obstacle_msg.poses = obstacle_pts
            obstacle_pub.publish(obstacle_msg)
            
            #Obstacle path all put into single list and then scatter plot plotted
            obstacle_msg = PoseArray()
            obstacle_pts = []
            for obstacle in obstacles:
                temp = human(obstacle)
                for i in temp:
                    ob_pts = Pose()
                    ob_pts.position.x = i[0]
                    ob_pts.position.y = i[1]
                # Is the correct orientation correct or enough:
                # l_path_pts.orientation.z = pt[2]
                #Try to get radius here
                    obstacle_pts.append(ob_pts)

            obstacle_msg.poses = obstacle_pts
            obstacle_paths_pub.publish(obstacle_msg)
            
            update_human()

            x, y = real_to_img_convert(start_point[0], start_point[1])
            ddtheta = 45
            start_point_img = x, y, round(start_point[2] / ddtheta) * ddtheta

            rate = rospy.Rate(4)
            rate.sleep()
            if rospy.is_shutdown():
                print('Shutting Down local planner')
                break

        else:
            rospy.logerr('Global Plan not available')


if __name__ == '__main__':
    main()
