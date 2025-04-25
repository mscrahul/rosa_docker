#!/usr/bin/env python3
import math
import time
import cv2
import os
import rospkg
import rospy
import yaml
import numpy as np
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseWithCovarianceStamped
import heapq
import numpy as np
import matplotlib.pyplot as plt

goal_point_map = [np.inf,np.inf]
start_point = (0,0)
start_available = False

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




# 1-25 WAYPOINTS
waypoints= [ [22733.1,6150.6],#dummy would not be accessed in any case
            [22733.1, 6150.599999999999], [22190.399999999998, 1085.3999999999999], [1085.3999999999999, 1628.1],
            [1085.3999999999999, 20381.399999999998], [1085.3999999999999, 23094.899999999998],
            [1085.3999999999999, 33828.299999999996], [-7959.599999999999, 33828.299999999996],
            [21828.6, 6150.599999999999], [21828.6, 2231.1], [2291.4, 2170.7999999999997], [1869.3, 22009.5],
            [13627.8, 22009.5], [20079.899999999998, 22009.5], [20079.899999999998, 23094.899999999998],
            [20079.899999999998, 28220.399999999998], [13627.8, 23094.899999999998], [13386.599999999999, 29969.1],
            [24903.899999999998, 29969.1], [25386.3, 33828.299999999996], [25386.3, 39617.1],
            [13748.4, 33828.299999999996], [12421.8, 33285.6], [12723.3, 23456.699999999997], [2291.4, 23034.6],
            [1869.3, 33104.7]]

#This
graph = {
            1: [2],  2: [3],  3: [4],  4: [5],  5: [6, 24],
            6: [7, 25],  7: [6],  8: [],  9: [8],  10: [9],
            11: [5, 10],  12: [11, 23],  13: [12],  14: [13, 15],  15: [14],
            16: [12, 14],  17: [16, 22],  18: [17],  19: [18, 20],
            20: [19],  21: [17, 19],  22: [21, 25], 23: [16, 22],
            24: [11, 23],  25: [6, 24]
        }


#This below is bidirectional
# graph  = {
#             1: [2],  2: [1, 3],  3: [2, 4],  4: [3, 5],  5: [4, 6, 24, 11],
#             6: [5, 7, 21, 25],  7: [6],  8: [9],  9: [8, 10],  10: [9, 11],
#             11: [10, 5, 24, 12],  12: [11, 23, 13, 16],  13: [12, 14],  14: [13, 15, 16],
#             15: [14], 16: [12, 14, 17, 23],  17: [16, 22, 18, 21],  18: [17, 19],
#             19: [18, 20, 21], 20: [19],  21: [17, 19, 22, 6],  22: [21, 25, 17, 23],
#             23: [16, 22, 12, 24], 24: [11, 23, 5, 25],  25: [6, 24, 22]
#          }

def intersect_pts(x1,y1,x2,y2,a,b,R):
    #Intersection between circle and line segment(parametric form) , we get quadratic in t
    A = (x2-x1)**2 + (y2-y1)**2
    B = 2*(x1-a)*(x2-x1) + 2*(y1-b)*(y2-y1)
    C = (x1-a)**2 + (y1-b)**2 - R**2
    D = B**2 - 4*A*C

    if D < 0:
        return None
    elif D == 0:
        x = x1 + (-B/(2*A))*(x2-x1)
        y = y1 + (-B/(2*A))*(y2 - y1)
        return [(x,y)]
    else:
        t1 = (-B - math.sqrt(D)) / (2 * A)  # pt1 --- t1 --- t2 --- pt2
        t2 = (-B + math.sqrt(D)) / (2 * A)
        t = [t1,t2]
        t = [i for i in t if 0 <= i <= 1]
        # i = (t[0] + t[1]) / 2
        pts = []
        for i in t:
            x = x1 + i * (x2 - x1)
            y = y1 + i * (y2 - y1)
            pts.append(([x, y]))#,(x2,y2),(x1,y1))
        return pts

def plot_path(path):
    X = []
    Y = []
    cwd_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(cwd_path, 'map_new.jpeg')  # finding full file path
    img = cv2.imread(file_path)
    
    fig, ax = plt.subplots()
    plt.cla()
    ax = plt.gca()
    ax.imshow(img, extent=[0, 795, -661, 0])
    for i in path:
        x, y = i[0], i[1]
        X.append(x)
        Y.append(-y)
    plt.plot(X, Y, marker='.')
    plt.show()

def finding_nearest_node(robot_pos, R): # Needs current pose and Radius of search
    visited_line_seg = set()
    possible_start = []
    exp = set()
    #iterate through waypoints and make line segments and check
    Q = [1]
    while len(Q) != 0:
        curr = Q.pop(0)
        if curr not in exp:
            for i in graph[curr]:
                if (curr,i) in visited_line_seg or (i,curr) in visited_line_seg:
                    continue
                x1, y1 = waypoints[curr]
                x2, y2 = waypoints[i]
                visited_line_seg.add((curr,i))
                points = intersect_pts(x1,y1,x2,y2,robot_pos[0], robot_pos[1], R)
                if points:
                    for p in points:
                        if len(points) == 2:
                            possible_start.append((curr,p,i,True))
                        else:
                            possible_start.append((curr, p, i))
            Q = Q + graph[curr]
            exp.add(curr)
    return possible_start

def calculate_path():
    global start_point, start_available
    R = 2000
    rospy.sleep(1)
    while not start_available:
        #cv2.waitKey(0)
        print('start not updated')
        rospy.sleep(0.5)
    start = start_point
    rospy.logerr(f'In start, {start_point}')
	
    # Getting Goal Point
    cwd_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(cwd_path, 'map_new.jpeg')  # finding full file path
    image = cv2.imread(file_path)
    gray_map = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # To select the start and end point
    cv2.imshow('Occupancy Map:Choose a goal point', gray_map)
    cv2.setMouseCallback('Occupancy Map:Choose a goal point', click_event)
    cv2.waitKey(0)

    # Wait until a goal point is selected
    while goal_point_map == [np.inf, np.inf]:  # and not rospy.is_shutdown():
        rospy.logerr("Goal Point not chosen")
        print("Goal Point not chosen")
        cv2.waitKey(0)

        #check if the chosen point is obstacle or not

        
    cv2.destroyAllWindows()
    print(goal_point_map)

    goal = img_to_real_convert(goal_point_map[0], goal_point_map[1])

    #goal = [25386.3, 20828.299999999996]
    a = finding_nearest_node(start, R)
    n_start = len(waypoints)
    waypoints.append(start)
    itr = iter(range(len(a)))
    for i in itr:
        n = len(waypoints)
        waypoints.append(a[i][1])
        # if waypoints[n] == a[i][1]:
        #     print('Waypoint appended correctly')
        #single direction
        #graph[n] = [a[i][0]]
        #print('graph',graph[n])

        #double direction
        if len(a[i]) == 3:
            graph[n] = [a[i][2]]
            #graph[a[i][2]].remove(a[i][0])
            #graph[a[i][2]].append(n)
            graph[a[i][0]].remove(a[i][2])
            graph[a[i][0]].append(n)
        elif len(a[i]) == 4:
            waypoints.append(a[i+1][1])
            graph[n] = [n+1]
            graph[n+1] = [a[i][2]]

            graph[a[i][0]].remove(a[i][2])
            #graph[a[i][2]].remove(a[i][0])

            graph[a[i][0]].append(n)
            #graph[a[i][2]].append(n+1)

            next(itr)



    start_nodes = []
    for i in range(n_start+1,len(waypoints)):
        start_nodes.append(i)
    graph[n_start] = start_nodes
    # b = [real_to_img_convert(i[1][0],i[1][1]) for i in a]
    # plot_path(b)
    a = finding_nearest_node(goal, R)
    n_goal = len(waypoints)
    waypoints.append(goal)
    itr = iter(range(len(a)))
    for i in itr:
        n = len(waypoints)
        waypoints.append(a[i][1])
        # if waypoints[n] == a[i][1]:
        #     print('Waypoint appended correctly')

        #Double direction
        if len(a[i]) == 3:
            graph[n] = [n_goal]
            #graph[a[i][2]].remove(a[i][0])
            #graph[a[i][2]].append(n)
            graph[a[i][0]].remove(a[i][2])
            graph[a[i][0]].append(n)
        elif len(a[i]) == 4:
            waypoints.append(a[i + 1][1])
            graph[n] = [n + 1, n_goal]
            graph[n + 1] = [ a[i][2] ]

            graph[a[i][0]].remove(a[i][2])
            #graph[a[i][2]].remove(a[i][0])

            graph[a[i][0]].append(n)
            graph[n+1].append(a[i][2])
            #graph[n_goal] = n+1 # pt1 -- t1 -- goal -- t2 -- pt2
            next(itr)
    #     # Single direction
    #     graph[a[i][2]].append(n)
    #     graph[n] = [n_goal]
    #graph[n_goal] = []
    return n_start,n_goal #Reutrns index of start and goal

def get_closest_node(current):
    min_dis = float('inf')
    node = -1
    for i in range(len(waypoints)):
        a = waypoints[i]
        dis = math.sqrt((current[0] - a[0])**2 + (current[1] - a[1])**2 )
        if dis < min_dis:
            min_dis = dis
            node = i
    return node

def get_start_pt_cb(msg):
    '''Callback function for getting the position of the robot'''
    global start_point, start_available
    x_robot = msg.pose.pose.position.x #/ 1000
    y_robot = msg.pose.pose.position.y #/ 1000
    start_point = (x_robot, y_robot)
    start_available = True
    
    #rospy.logerr(f'In callback, {start_point}')
    

def click_event(event, x, y, flags, params):
    '''Checking for left mouse clicks to choose the goal point'''
    global goal_point_map
    if event == cv2.EVENT_LBUTTONDOWN:
        goal_point_map = [x, y]

def astar(road_map, start, goal):

    def heuristic(node):
        return np.linalg.norm(np.array(node) - np.array(road_map[int(goal)]))

    # Initializing
    queue = []
    cost = {start: 0}
    heapq.heappush(queue, (0, start))
    parent = {start: None}


    while queue:
        _, current = heapq.heappop(queue)
        if current == goal:
            break

        neighbors = graph[current]



        for neighbor in neighbors:
            # if map_array[neighbor[0]][neighbor[1]] == 0:
            #     continue
            # Calculate the cost of reaching the neighbor
            neighbor_cost = cost[current] + np.linalg.norm(np.array(road_map[neighbor]) - np.array(road_map[current]))

            if neighbor not in cost or neighbor_cost < cost[neighbor]:
                #g
                cost[neighbor] = neighbor_cost
                #f = g+h
                priority = neighbor_cost + heuristic(road_map[neighbor])
                # adding new nodes to the q
                heapq.heappush(queue, (priority, neighbor))
                #connect node to its parent
                parent[neighbor] = current

    # Retrieve the path from the goal to the start
    path = []
    current = goal

    while current:
        path.append(current)
        current = parent[current]

    path.reverse()
    return path

def main():
    rospy.init_node('global_path_planner_node')  # Node initialization
    global_path_pub = rospy.Publisher("/global_path", PoseArray, queue_size=1,latch=True)  # Publisher for global path
    rospy.Subscriber("/pose_ekf", PoseWithCovarianceStamped, get_start_pt_cb) # Subscriber for getting the pose data

    start, goal = calculate_path() # gets goal point and adds the points to new graph

    path = astar(waypoints, start, goal)
    path_real_coord = []
    for i in path:
        path_real_coord.append(waypoints[i])
    print(path)
    

    if path is None:
        return
    if len(path) > 0:
        # Storing the position as a pose array to be used by other nodes
        path_msg = PoseArray()
        path_pts = []
        for pt in path_real_coord:
            global_path_pts = Pose()
            global_path_pts.position.x = pt[0]
            global_path_pts.position.y = pt[1]
            path_pts.append(global_path_pts)
            
        #starting point not required for gtg
        
        path_msg.poses = path_pts
        #rate.sleep()
        for _ in range(3):
            global_path_pub.publish(path_msg) 
            rospy.sleep(1)
        #rospy.sleep(10)
        #A plot to show the path on the map
        b = [real_to_img_convert(i[0], i[1]) for i in path_real_coord]
        print(path_real_coord)
        #plot_path(b)
        print('Path found and is being published')
        rospy.spin()
        #rospy.sleep(10)
        
    else:
        rospy.logerr('Goal is not reachable')
        print('Goal Not Reachable')

 



if __name__ == '__main__':
    main()
    
#'{position: {x: 1500.0, y: 6000.5, z: 0.0}, orientation: {x: 0.0, y: 0.0,z: 0.0,w: 0.0}}'
