import numpy as np
from scipy.ndimage import label
import math

def plan_debris_collection_route(grid, start_pos, budget):
    """
    Analyzes a grid map to find an optimized debris collection route and initial travel direction.

    This function performs all steps:
    1. Identifies debris clusters ('nodes') from the input grid.
    2. Calculates prizes (size) and locations (centroid) for each node.
    3. Uses a greedy insertion heuristic to build a high-value route within the travel budget.
    4. Determines the initial travel direction from the start point.
    5. Prints a full report of the plan.

    Args:
        grid (list[list[int]]): The 2D map with 1s for debris and 0s for water.
        start_pos (tuple[int, int]): The (row, col) starting coordinate.
        budget (int): The maximum allowed travel distance.
    """

    # --- Nested Helper Function for Direction ---
    def get_direction(start_node_pos, first_stop_pos):
        """Calculates the approximate compass direction from a start to an end point."""
        # Note: In grid coordinates, y increases downwards.
        # dy = y2 - y1 (row change)
        # dx = x2 - x1 (column change)
        dy = first_stop_pos[0] - start_node_pos[0]
        dx = first_stop_pos[1] - start_node_pos[1]

        if dx == 0 and dy == 0:
            return "Stationary"
        
        # Calculate angle in degrees: 0 is East, 90 is North, 180 is West, -90 is South
        angle = math.degrees(math.atan2(-dy, dx))
        
        # Normalize angle to be within [0, 360)
        if angle < 0:
            angle += 360

        # Define compass directions and their angle ranges
        directions = {
            "East": (0, 22.5), "North-East": (22.5, 67.5),
            "North": (67.5, 112.5), "North-West": (112.5, 157.5),
            "West": (157.5, 202.5), "South-West": (202.5, 247.5),
            "South": (247.5, 292.5), "South-East": (292.5, 337.5)
        }
        
        for direction, (lower, upper) in directions.items():
            if lower <= angle < upper:
                return direction
        return "East" # Catches the case where angle is between 337.5 and 360

    
# --- Main Execution ---
if __name__ == '__main__':
    # Define the world grid, start position, and budget
    world_grid = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    
    START_POS = (0, 0)
    TRAVEL_BUDGET = 25

    # Call the main function to plan the route and print the report
    plan_debris_collection_route(world_grid, START_POS, TRAVEL_BUDGET)