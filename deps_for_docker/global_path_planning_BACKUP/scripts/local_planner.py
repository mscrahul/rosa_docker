#!/usr/bin/env python3
import numpy as np
import heapq
import cv2
import time
import rospy
#import pygame
from math import cos,sin,radians,degrees, sqrt
import matplotlib.pyplot as plt
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseWithCovarianceStamped
import tf
import os

start_point = (0,0,0)
start_available = False
primitive_reduced = 2
primitive_reset = 0

XDIM = 795
YDIM = 661
WINSIZE = [XDIM, YDIM]
EPSILON = 7.0
NUMNODES = 5000

global_path = []

x_offset = 93 # x offset to adjust with the map, home coordinates
y_offset = 200 # y offset to adjust with the map, home coordinates
scale_x = 60.30 # scale for x values to adjust with the map
scale_y = 60.30

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

#220,224
# obstacles = [(122,262), (105,295)]
obstacles = [(172, 211)]
#Min step size of primitives  
v = 10
dtheta = 30

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

def neighbors_mp(x,y,theta):
    theta = angle_manage(theta)
    forward = (x + v * cos(radians(theta)), -(y + v * sin(radians(theta))), theta)
    left_t =  (x + v * cos(radians(theta+dtheta/2)), -(y + v * sin(radians(theta+dtheta/2))), theta + dtheta )
    right_t = (x + v * cos(radians(theta-dtheta/2)), -(y + v * sin(radians(theta-dtheta/2))), theta - dtheta )
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


def neighbors_mp(x,y,theta):#t- time
    theta = angle_manage(theta)
    forward = (x + v * cos(radians(theta)), -(y + v * sin(radians(theta))), theta)
    left_t_2 = (x + (v/2) * cos(radians(theta + dtheta / 4)), -(y + (v/2) * sin(radians(theta + dtheta / 4))), theta + dtheta/2 )
    right_t_2 = (x + (v/2) * cos(radians(theta - dtheta / 4)), -(y + (v/2) * sin(radians(theta - dtheta / 4))), theta - dtheta/2)
    left_t_1 =  (x + v * cos(radians(theta+dtheta/2)), -(y + v * sin(radians(theta+dtheta/2))), theta + dtheta)
    right_t_1 = (x + v * cos(radians(theta-dtheta/2)), -(y + v * sin(radians(theta-dtheta/2))), theta - dtheta )
    neighbors = [forward, left_t_1, left_t_2, right_t_1, right_t_2 ]
    return neighbors #should return neighbour in the form (x,y,theta)

def neighbors_mp_grid(x,y,theta):
    thres = 4
    theta = angle_manage(theta)
    forward = ( round((x + v * cos(radians(theta))) / thres) * thres, round((-(y + v * sin(radians(theta)))) / thres) * thres,theta)
    left_t_2 = (round((x + (v/2) * cos(radians(theta + dtheta / 4))) / thres) * thres,
              round((-(y + (v/2)* sin(radians(theta + dtheta / 4)))) / thres) * thres, theta + dtheta/2)
    right_t_2 = (round((x + (v/2) * cos(radians(theta - dtheta / 4))) / thres) * thres,
               round((-(y + (v/2) * sin(radians(theta - dtheta / 4)))) / thres) * thres, theta - dtheta/2 )
    left_t = (round((x + v * cos(radians(theta + dtheta / 2))) / thres) * thres,round((-(y + v * sin(radians(theta + dtheta / 2)))) / thres) * thres, theta + dtheta)
    right_t = (round((x + v * cos(radians(theta - dtheta / 2))) / thres) * thres,round((-(y + v * sin(radians(theta - dtheta / 2)))) / thres) * thres, theta - dtheta)
    neighbors = [forward,left_t,right_t, left_t_2 ,right_t_2]
    return neighbors #should return neighbour in the form (x,y,theta)

def get_start_pt_cb(msg):
    '''Callback function for getting the position of the robot'''
    global start_point, start_available
    x_robot = msg.pose.pose.position.x #/ 1000
    y_robot = msg.pose.pose.position.y #/ 1000
    #Need to get orientation of robot also and convert it to degrees
    orientation_list = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z,
                         msg.pose.pose.orientation.w]
    euler = tf.transformations.euler_from_quaternion(orientation_list)
    theta = degrees(euler[2]) 

    start_point = (x_robot, y_robot, theta-90)
    start_available = True
    
    #rospy.logerr(f'In callback, {start_point}')

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


def astar(map_array, start, goal):
    #Pygame for graph visualization:
    # pygame.init()
    # screen = pygame.display.set_mode(WINSIZE)
    # cwd_path = os.path.dirname(os.path.abspath(__file__))
    # file_path = os.path.join(cwd_path, 'map_new.jpeg') 
    # img = pygame.image.load(file_path)
    # img.convert()

    # pygame.display.set_caption('RRT      S. LaValle    May 2011')
    # white = 255, 240, 200
    # black = 20, 20, 40
    # RED = (255, 0, 0)
    # screen.fill(black)
    # rect = img.get_rect()
    # rect.center = XDIM // 2, YDIM // 2
    # screen.blit(img, rect)
    # pygame.draw.rect(screen, RED, rect, 1)
    # pygame.display.update()
    
    # print("from A*", obstacle)
    # #A*
    #start_t = time.time_ns()
    global v, primitive_reset

    height, width = map_array.shape


    def heuristic(node):
        return np.linalg.norm(np.array(node) - np.array(goal[:2]))



    # Initializing
    queue = []
    cost = {start: 0}
    gm = {start:start}  #maps points grid ones to crct/actual primitive
    heapq.heappush(queue, (0, start))
    parent = {start: None}

    while queue:
        _, current = heapq.heappop(queue)
        current_gm = gm[current]
        #First if condition for pygame animation
        # if parent[current] == None:
        #     pass
        # else:
        #     pygame.draw.line(screen, black, gm[parent[current]][:2], current_gm[:2])
        #     pygame.display.update()

        if current_gm == goal or cost[current] > 200:
            goal = current
            #print('goal')
            break

        #If within set threshold then goal is changed to current to trace back the path
        if heuristic(current_gm[:2]) < v/2 and abs(current_gm[2] - goal[2])< dtheta:
            print('thresholded', goal , current)
            goal = current 
            break

        neighbors = []

        x, y, theta = current_gm

        neighbors = neighbors_mp(x, -y, theta)
        neighbors_grid = neighbors_mp_grid(x, -y, theta)
        
        if primitive_reset == 5:
            v = 10
        else:
            primitive_reset += 1

        for i,neighbor in enumerate(neighbors):

            # if map_array[neighbor[0]][neighbor[1]] == 0:
            #     continue
            # Calculate the cost of reaching the neighbor
            if 0 <= neighbor[0] < width and 0 <= neighbor[1] < height:
                #Check if obstacle
                if map_array[int(neighbor[1])][int(neighbor[0])] < 210 :
                    continue

                #Obstacle collision check
                for obstacle in obstacles:
                    in_collision = False
                    if np.sqrt((neighbor[0]-obstacle[0])**2 + (neighbor[1]-obstacle[1])**2) < 8.0:
                        # print(neighbor,obstacle,'Collision with Bill')
                        in_collision = True
                        break
                if in_collision:
                    continue
            cross_track_error = xte(current_gm)
            #print(neighbor)
            neighbor_cost = cost[current] + np.linalg.norm(np.array(neighbor[:2]) - np.array(current_gm[:2])) + \
                            2*((abs(np.array(neighbor[2]) - np.array(current[2])) % 180)/dtheta) +\
                            (255 - map_array[int(neighbor[1])][int(neighbor[0])]) * 10 / 255 + cross_track_error

            if neighbors_grid[i] not in cost or neighbor_cost < cost[neighbors_grid[i]]:
                # g
                cost[neighbors_grid[i]] = neighbor_cost
                # f = g+h
                priority = neighbor_cost + heuristic(neighbor[:2]) # + 0.5*((abs(neighbor[2] - goal[2])%180)//dtheta)
                # adding new nodes to the q
                heapq.heappush(queue, (priority, neighbors_grid[i]))
                # connect node to its parent
                parent[neighbors_grid[i]] = current
                gm[neighbors_grid[i]] = neighbor
                #print('queue',queue)
    
    # Retrieve the path from the goal to the start
    path = []
    current = goal
    flag = 1
    global primitive_reduced
    while current:
        print(v,'Hello',primitive_reduced)
        try:
            path.append(gm[current])
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
    return path


if __name__ == '__main__':
    rospy.init_node('local_planner')
    plt.ion()
    global_path_pose = rospy.wait_for_message('/global_path',PoseArray, timeout=5)
    
    local_path_pub = rospy.Publisher('/local_path',PoseArray, queue_size = 10)
    rospy.Subscriber("/pose_ekf", PoseWithCovarianceStamped, get_start_pt_cb) 
    #Have to convert to image coordinates
    waypoints = global_path_pose.poses
    
    rospy.sleep(0.2) #So that start_point updated
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
        
        
        #wp - waypoints
        wp_img_coord = [ real_to_img_convert(i.position.x,i.position.y) for i in waypoints ] 
        global_path = orientation_clc(wp_img_coord)
        #Start point orientation according to image 
        global_path[0].append(angle_manage(start_point[2]- 90) )
        distance_threshold = rospy.get_param("distance_threshold")
        n = len(global_path) 
        x, y = real_to_img_convert(start_point[0], start_point[1])
        start_point_img = x, y, start_point[2]#//dtheta *dtheta
        goal_point = global_path[-1]

        print('Local Planner Running')

        while not np.linalg.norm( np.array(start_point[:2]) - np.array([waypoints[-1].position.x , waypoints[-1].position.y])) < distance_threshold:   #In real world coordinates
        # Start point has to be checked for orientation and stuff
            path = astar(map_array, start_point_img, goal_point)
            
            lp_real_coord = [ img_to_real_convert(i[0],i[1]) + [i[2]] for i in path]
            #Convert orientaions and then publish
            if len(path) > 1:
                # Storing the position as a pose array to be used by other nodes
                path_msg = PoseArray()
                path_pts = []
                for pt in lp_real_coord:
                    l_path_pts = Pose()
                    l_path_pts.position.x = pt[0]
                    l_path_pts.position.y = pt[1]
                    # Is the correct orientation correct or enough:
                    l_path_pts.orientation.z = pt[2]
                    path_pts.append(l_path_pts)
        
                path_msg.poses = path_pts
                local_path_pub.publish(path_msg)
            
            x, y = real_to_img_convert(start_point[0], start_point[1])
            ddtheta = 45
            start_point_img = x, y, round(start_point[2]/ddtheta)*ddtheta

            rate = rospy.Rate(4)
            rate.sleep()
            if rospy.is_shutdown():
                print('Shutting Down local planner')
                break
            
        else:
            rospy.logerr('Global Plan not available')

'''Below works by working  with waypoints and running lp between waypoints'''

# if __name__ == '__main__':
#     rospy.init_node('local_planner')
#     global_path = rospy.wait_for_message('/global_path',PoseArray, timeout=5)
    
#     local_path_pub = rospy.publisher('/local_path',PoseArray, queue_size = 10)
#     rospy.Subscriber("/pose_ekf", PoseWithCovarianceStamped, get_start_pt_cb) 
#     #Have to convert to image coordinates
#     waypoints = global_path.poses

#     if waypoints:
#         image = cv2.imread('map_new.jpeg')
#         # Use the cvtColor() function to grayscale the image
#         gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         # cv2_imshow( gray_image)
#         # Apply binary thresholding to create a binary image
#         threshold_value = 210
#         max_value = 255
#         ret, binary_image = cv2.threshold(gray_image, threshold_value, max_value, cv2.THRESH_BINARY)
#         im = np.asarray(binary_image)
#         im_gray = np.asarray(gray_image)
#         # box_blur_ker = (1/25)*np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]])
#         box_blur_ker = (1 / 225) * np.ones((15, 15))
#         Box_blur = cv2.filter2D(src=im, ddepth=-1, kernel=box_blur_ker)
#         # cv2_imshow(Box_blur)
#         imblur = np.asarray(Box_blur)
#         map_array = imblur
        
        
#         #wp - waypoints
#         wp_img_coord = [ real_to_img_convert(i.position.x,i.position.y) for i in waypoints ] 
#         global_path= orientation_clc(waypoints)
#         #Start point orientation according to image 
#         wp_img_with_ori[0].append(angle_manage(start_point[2]) - 90)
#         distance_threshold = rospy.get_param("distance_threshold")
#         n = len(wp_img_with_ori) 
#         for i in range(1,n):        
#             x, y = real_to_img_convert(start_point[0], start_point[1])
#             start_point_img = x, y, start_point[2]
#             while np.linalg.norm( np.array(start_point[:2]) - np.array([waypoints[i].position.x , waypoints[i].position.y])) < distance_threshold:
#                 if i != n - 1:
#                     # Start point has to be checked for orientation and stuff
#                     path = astar(map_array, start_point_img, wp_img_with_ori[i]) + astar(map_array, wp_img_with_ori[i], wp_img_with_ori[i+1]) 
#                 else:
#                     path = astar(map_array, start_point, wp_img_with_ori[i])  # Final waypoint to goal
#                 #Update start point
#                 lp_real_coord = [ img_to_real_convert(i[0],i[1]) + [i[2]] for i in path]
#                 #Convert orientaions and then publish
#                 if len(path) > 1:
#                     # Storing the position as a pose array to be used by other nodes
#                     path_msg = PoseArray()
#                     path_pts = []
#                     for pt in lp_real_coord:
#                         l_path_pts = Pose()
#                         l_path_pts.position.x = pt[0]
#                         l_path_pts.position.y = pt[1]
#                         # Is the correct orientation correct or enough:
#                         l_path_pts.orientation.z = pt[2]
#                         path_pts.append(l_path_pts)
            
#         #starting point not required for gtg
#                     path_msg.poses = path_pts
#                     local_path_pub.publish(path_msg)
                
#                 x, y = real_to_img_convert(start_point[0], start_point[1])
#                 start_point_img = x, y, start_point[2]
#             # img = plt.imread("map_new.jpeg")
#             # fig, ax = plt.subplots()
#             # x = range(300)
#             # ax.imshow(img, extent=[0, 795,-661,0])
#             # Drawing_colored_circle = plt.Circle((obstacle[0],-obstacle[1]), 6.5, color='g')
#             # ax.add_patch(Drawing_colored_circle)
#             # X_final_path = []
#             # Y_final_path = []
#             # for i,j,k in path:
#             #     X_final_path.append(i)
#             #     Y_final_path.append(-j)
#             # plt.scatter( X_final_path, Y_final_path,marker='.')
#             # plt.show()
#     else:
#         rospy.logerr('Global Plan not available')

