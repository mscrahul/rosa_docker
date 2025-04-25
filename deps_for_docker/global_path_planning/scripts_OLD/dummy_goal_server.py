#!/usr/bin/env python3

import rospy
import time
from std_msgs.msg import Empty # Using Empty message as a simple trigger

# Import the custom service definition (make sure your package builds it)
try:
    from global_path_planning.srv import SetPixelGoal, SetPixelGoalResponse
except ImportError as e:
    rospy.logfatal(f"Failed to import SetPixelGoal service definition: {e}")
    rospy.logfatal("Did you build the workspace (catkin_make) and source devel/setup.bash?")
    import sys
    sys.exit(1)

# --- Configuration ---
SIMULATED_TRAVEL_TIME = 5.0 # Seconds to wait before returning success
USE_TRIGGER_TOPIC = False   # Set to True to wait for a message instead of fixed time
TRIGGER_TOPIC_NAME = "/dummy_goal_trigger" # Topic to listen on if USE_TRIGGER_TOPIC is True
SERVICE_TIMEOUT = 60.0      # Seconds to wait before timing out a service call

# --- Global variable to hold the trigger state ---
trigger_received = False
trigger_sub = None # Subscriber object

def trigger_callback(msg):
    """Callback function for the trigger topic."""
    global trigger_received
    rospy.loginfo("Dummy Goal Server: Received trigger!")
    trigger_received = True

def handle_set_pixel_goal(req):
    """Service handler for the dummy server."""
    global trigger_received

    rospy.loginfo(f"Dummy Goal Server: Received request for pixel ({req.pixel_x}, {req.pixel_y}).")
    rospy.loginfo("Simulating robot navigation...")

    response = SetPixelGoalResponse()
    response.reached_goal = False # Default to False
    response.message = "Processing goal..."

    start_time = rospy.Time.now()
    timeout_duration = rospy.Duration(SERVICE_TIMEOUT) # Use timeout from planner for consistency

    if USE_TRIGGER_TOPIC:
        # --- Wait for Trigger Topic ---
        trigger_received = False # Reset trigger for this request
        rospy.loginfo(f"Waiting for trigger message on topic '{TRIGGER_TOPIC_NAME}'...")
        while not trigger_received and not rospy.is_shutdown():
            # Check for timeout
            if (rospy.Time.now() - start_time) > timeout_duration:
                 rospy.logwarn("Dummy Goal Server: Timeout waiting for trigger.")
                 response.message = "Timeout waiting for external trigger."
                 response.reached_goal = False
                 return response
            try:
                rospy.sleep(0.2) # Check periodically
            except rospy.ROSInterruptException:
                rospy.logwarn("Dummy Goal Server: Shutdown during trigger wait.")
                response.message = "Shutdown during trigger wait."
                response.reached_goal = False
                return response

        if trigger_received:
            rospy.loginfo("Dummy Goal Server: Trigger received! Goal 'reached'.")
            response.reached_goal = True
            response.message = "Goal reached (triggered)."
        # If loop exited due to shutdown, response remains False

    else:
        # --- Wait for Fixed Time ---
        rospy.loginfo(f"Simulating travel time ({SIMULATED_TRAVEL_TIME} seconds)...")
        try:
            # Use rospy.sleep for a duration, checking for shutdown
            end_time = rospy.Time.now() + rospy.Duration(SIMULATED_TRAVEL_TIME)
            rate = rospy.Rate(10) # Check shutdown status periodically
            while rospy.Time.now() < end_time and not rospy.is_shutdown():
                rate.sleep()

            if rospy.is_shutdown():
                 rospy.logwarn("Dummy Goal Server: Shutdown during simulated travel.")
                 response.message = "Shutdown during simulated travel."
                 response.reached_goal = False
            else:
                 rospy.loginfo("Dummy Goal Server: Simulated travel complete. Goal 'reached'.")
                 response.reached_goal = True
                 response.message = "Goal reached (simulated time)."

        except rospy.ROSInterruptException:
             rospy.logwarn("Dummy Goal Server: Shutdown during simulated travel sleep.")
             response.message = "Shutdown during simulated travel sleep."
             response.reached_goal = False

    return response

def dummy_server_node():
    """Initializes the node and starts the service server."""
    # --- Declare globals FIRST ---
    global SIMULATED_TRAVEL_TIME, USE_TRIGGER_TOPIC, TRIGGER_TOPIC_NAME, SERVICE_TIMEOUT
    global trigger_sub # Also declare trigger_sub if modified globally

    rospy.init_node('dummy_set_pixel_goal_server')
    node_name = rospy.get_name()
    rospy.loginfo(f"{node_name} started.")

    # --- Now read parameters and assign to the (now declared) globals ---
    sim_time_param = rospy.get_param("~simulated_travel_time", SIMULATED_TRAVEL_TIME)
    use_trigger_param = rospy.get_param("~use_trigger", USE_TRIGGER_TOPIC)
    trigger_topic_param = rospy.get_param("~trigger_topic", TRIGGER_TOPIC_NAME)
    timeout_param = rospy.get_param("~service_timeout", SERVICE_TIMEOUT)

    # Update globals based on parameters
    SIMULATED_TRAVEL_TIME = sim_time_param
    USE_TRIGGER_TOPIC = use_trigger_param
    TRIGGER_TOPIC_NAME = trigger_topic_param
    SERVICE_TIMEOUT = timeout_param

    rospy.loginfo(f"Simulated travel time: {SIMULATED_TRAVEL_TIME}s")
    rospy.loginfo(f"Wait for trigger topic: {USE_TRIGGER_TOPIC}")
    if USE_TRIGGER_TOPIC:
        rospy.loginfo(f"Trigger topic name: {TRIGGER_TOPIC_NAME}")
        # Initialize subscriber only if needed
        trigger_sub = rospy.Subscriber(TRIGGER_TOPIC_NAME, Empty, trigger_callback)

    # Advertise the service
    try:
        service_name = 'set_pixel_goal' # Match the name used by the client
        s = rospy.Service(service_name, SetPixelGoal, handle_set_pixel_goal)
        rospy.loginfo(f"Dummy service '{rospy.resolve_name(service_name)}' is ready.")
    except Exception as e:
        rospy.logfatal(f"Failed to create dummy service: {e}", exc_info=True)
        return # Exit if service creation fails

    rospy.loginfo("Dummy server spinning, waiting for requests...")
    rospy.spin() # Keep the node alive until shutdown

    rospy.loginfo(f"{node_name} shutting down.")

if __name__ == "__main__":
    try:
        dummy_server_node()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logfatal(f"Unhandled exception in dummy server main: {e}", exc_info=True)
