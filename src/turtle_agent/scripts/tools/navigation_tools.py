#!/usr/bin/env python3.9
# ~/catkin_ws/src/rosa/src/turtle_agent/scripts/tools/navigation_tools.py

import rospy
from langchain.agents import tool

# --- Import YOUR service definition ---
# This relies on global_path_planning being built by catkin build inside the container
try:
    # Correct import path based on catkin build structure
    from global_path_planning.srv import SetPixelGoal, SetPixelGoalRequest
except ImportError as e:
    rospy.logerr(f"CRITICAL ERROR [ROSA Tool]: Cannot import SetPixelGoal service definition: {e}. "
                 f"Build inside Docker likely failed to process the srv file. "
                 f"Check Dockerfile COPY and catkin build steps.")
    # Define dummy classes to prevent further import errors, but tool will fail
    class SetPixelGoal: pass
    class SetPixelGoalRequest: pass
    # Or raise the exception again if you prefer immediate failure
    # raise ImportError("Failed to import SetPixelGoal service definition") from e

# --- Define Known Locations with PIXEL Coordinates ---
# Use the exact names you want the user/LLM to use.
# These should match the pixel coordinates expected by your /set_pixel_goal service.
# Example - REPLACE WITH YOUR ACTUAL PIXEL COORDINATES
KNOWN_LOCATIONS_PIXELS = {
    "professor room": (100, 150), # Example pixel coords
    "water point": (300, 400),    # Example pixel coords
    "laboratory1": (500, 550),   # Example pixel coords
    "laboratory2": (500, 450),   # Example pixel coords
    "lift 1": (50, 600),         # Example pixel coords
    "elevator 1": (50, 600),     # Alias
    "lift 2": (700, 300),        # Example pixel coords
    "elevator 2": (700, 300),    # Alias
    "innovation hall": (400, 600),# Example pixel coords
    # Add other locations with their PIXEL coordinates
}

@tool
def navigate_to_named_location(location_name: str) -> str:
    """
    Sends the robot to a predefined named location by looking up its known
    pixel coordinates and calling the /set_pixel_goal ROS service.
    Use 'list_known_locations' to see available names.

    :param location_name: The exact name of the destination (e.g., 'professor room', 'lift 1'). Case-insensitive.
    """
    service_name = "/set_pixel_goal" # Service name provided by your robot's node
    rospy.loginfo(f"[ROSA Tool] Request: Navigate to named location '{location_name}'.")

    if SetPixelGoal is None or SetPixelGoalRequest is None:
        err_msg = "Error: SetPixelGoal service type not available (Import failed inside Docker)."
        rospy.logerr(f"[ROSA Tool] {err_msg}")
        return err_msg

    normalized_name = location_name.lower().strip()
    pixel_coords = KNOWN_LOCATIONS_PIXELS.get(normalized_name)

    if pixel_coords is None:
        known_names = list(KNOWN_LOCATIONS_PIXELS.keys())
        # Filter out aliases for display if needed, or just show all
        display_names = sorted(list(set(name for name in known_names)))
        error_msg = f"Error: Unknown location '{location_name}'. Known locations are: {', '.join(display_names)}."
        rospy.logwarn(f"[ROSA Tool] {error_msg}")
        return error_msg

    pixel_x, pixel_y = pixel_coords
    rospy.loginfo(f"[ROSA Tool] Found pixels ({pixel_x}, {pixel_y}). Calling service {service_name}...")

    try:
        # Wait for the service to become available (important!)
        rospy.wait_for_service(service_name, timeout=5.0)
        # Create a service proxy
        set_goal_proxy = rospy.ServiceProxy(service_name, SetPixelGoal)
        # Create the request message
        request = SetPixelGoalRequest()
        request.pixel_x = int(pixel_x) # Ensure integer type
        request.pixel_y = int(pixel_y) # Ensure integer type
        # Call the service
        response = set_goal_proxy(request)

        # Process the response
        if hasattr(response, 'success') and response.success: # Check if 'success' attribute exists
            feedback = response.message if hasattr(response, 'message') and response.message else "Goal accepted by planner."
            result_msg = f"Successfully sent navigation goal for '{location_name}' (Pixel: {pixel_x}, {pixel_y}). Response: {feedback}"
            rospy.loginfo(f"[ROSA Tool] {result_msg}")
            return result_msg
        elif hasattr(response, 'reached_goal') and response.reached_goal: # Check legacy attribute name
             feedback = response.message if hasattr(response, 'message') and response.message else "Goal accepted by planner."
             result_msg = f"Successfully sent navigation goal for '{location_name}' (Pixel: {pixel_x}, {pixel_y}). Response: {feedback}"
             rospy.loginfo(f"[ROSA Tool] {result_msg}")
             return result_msg
        else:
            feedback = response.message if hasattr(response, 'message') and response.message else "Planner rejected goal or failed."
            error_msg = f"Failed to set goal for '{location_name}' (Pixel: {pixel_x}, {pixel_y}). Response: {feedback}"
            rospy.logwarn(f"[ROSA Tool] {error_msg}")
            return error_msg

    except rospy.ServiceException as e:
        error_msg = f"Error calling service {service_name}: {e}"
        rospy.logerr(f"[ROSA Tool] {error_msg}")
        return error_msg
    except rospy.ROSException as e:
        error_msg = f"ROS Error connecting to service '{service_name}' (Timeout? Is service running?): {e}"
        rospy.logerr(f"[ROSA Tool] {error_msg}")
        return error_msg
    except AttributeError as e:
         # This might happen if the service definition wasn't imported correctly
         error_msg = f"AttributeError calling service '{service_name}'. Likely service definition import failed: {e}"
         rospy.logerr(f"[ROSA Tool] {error_msg}")
         return error_msg
    except Exception as e:
        error_msg = f"Unexpected error in tool '{navigate_to_named_location.__name__}': {type(e).__name__} - {e}"
        rospy.logerr(f"[ROSA Tool] {error_msg}")
        return f"Error: {error_msg}"

@tool
def list_known_locations() -> str:
    """Lists the names of predefined locations the robot can be sent to using the navigate_to_named_location tool."""
    names = list(KNOWN_LOCATIONS_PIXELS.keys())
    if not names:
        return "I do not have any specific named locations stored."
    # Filter out aliases if they exist and you only want unique destinations shown
    unique_names = sorted(list(set(name for name in names))) # Simple approach: show all defined keys
    return f"I can navigate to these specific named locations: {', '.join(unique_names)}."