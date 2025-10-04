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

    # --- Step 1: Preprocessing - Find Nodes and Prizes ---
    labeled_array, num_features = label(grid)
    nodes = {0: {"pos": start_pos, "prize": 0, "id": 0}}
    
    for i in range(1, num_features + 1):
        coords = np.argwhere(labeled_array == i)
        if coords.size > 0:
            center_pos = tuple(np.mean(coords, axis=0).round().astype(int))
            nodes[i] = {"pos": center_pos, "prize": len(coords), "id": i}

    print(f"Analysis complete: Found {len(nodes)-1} debris clusters (nodes).")
    for nid, data in nodes.items():
        if nid > 0:
            print(f"  - Node {nid}: Position={data['pos']}, Prize={data['prize']}")
    print("-" * 30)

    # --- Step 2: Algorithm - Greedy Insertion Heuristic ---
    def manhattan_distance(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    tour = [nodes[0], nodes[0]] 
    total_prize = 0
    total_distance = 0
    unvisited_node_ids = set(n for n in nodes if n != 0)

    while True:
        best_insertion = None
        best_score = -1
        for node_id in unvisited_node_ids:
            node_to_insert = nodes[node_id]
            for i in range(len(tour) - 1):
                prev_node, next_node = tour[i], tour[i+1]
                cost_increase = (manhattan_distance(prev_node['pos'], node_to_insert['pos']) +
                                 manhattan_distance(node_to_insert['pos'], next_node['pos']) -
                                 manhattan_distance(prev_node['pos'], next_node['pos']))
                
                if total_distance + cost_increase <= budget:
                    score = node_to_insert['prize'] / (cost_increase + 1e-9)
                    if score > best_score:
                        best_score = score
                        best_insertion = {"node": node_to_insert, "index": i + 1, "cost": cost_increase}
        
        if best_insertion:
            node = best_insertion['node']
            tour.insert(best_insertion['index'], node)
            total_distance += best_insertion['cost']
            total_prize += node['prize']
            unvisited_node_ids.remove(node['id'])
        else:
            break
            
    # --- Step 3: Output the Results ---
    print("MISSION PLAN:")
    print("="*30)
    
    if len(tour) > 2:
        start_node = tour[0]
        first_destination_node = tour[1]
        initial_direction = get_direction(start_node['pos'], first_destination_node['pos'])
        
        print(f"🚀 Initial Move: Head {initial_direction} towards Node {first_destination_node['id']} at {first_destination_node['pos']}.")
        print(f"✔️ Optimal Path Found: {[node['id'] for node in tour]}")
        print(f"📍 Path Coordinates: {[node['pos'] for node in tour]}")
        print(f"🏆 Total Prize Collected: {total_prize}")
        print(f"⛽ Total Distance Traveled: {total_distance:.2f} (Budget was {budget})")
    else:
        print("⚠️ No viable route found within the given travel budget.")
        print(f"   Consider increasing the budget of {budget}.")

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