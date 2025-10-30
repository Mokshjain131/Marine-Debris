#!/usr/bin/env python3
"""
Marine Debris Collection Route Planner with Map Visualization

This extension:
1. Loads the geolocation hashmap of debris locations
2. Plans optimal debris collection routes using Orienteering Problem approach
3. Visualizes debris locations on an interactive world map
4. Shows ship routing paths on the map
5. Exports interactive HTML maps for viewing

Usage:
    python extension.py --geolocation_map debris_geolocation_map.json
"""

import numpy as np
from scipy.ndimage import label
import math
import json
import folium
from folium import plugins
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DebrisRouteVisualizer:
    """Visualizes marine debris locations and collection routes on maps"""
    
    def __init__(self, geolocation_file: str, use_per_patch: bool = False):
        """Load geolocation data"""
        print(f"Loading geolocation data from: {geolocation_file}")
        
        with open(geolocation_file, 'r') as f:
            self.data = json.load(f)
        
        # Check if this is per-patch data or aggregated data
        if 'patch_map' in self.data:
            # Per-patch format
            print("[INFO] Using per-patch data format")
            self.use_per_patch = True
            patch_map = self.data['patch_map']
            self.metadata = self.data.get('metadata', {})
            
            # Extract all patches with coordinates
            self.debris_locations = []
            self.all_locations = []
            
            for patch_name, patch_data in patch_map.items():
                coords = patch_data.get('coordinates')
                if coords and coords['lat'] is not None and coords['lon'] is not None:
                    location_info = {
                        'id': patch_name,
                        'lat': coords['lat'],
                        'lon': coords['lon'],
                        'has_debris': patch_data.get('has_debris', False),
                        'confidence': patch_data.get('confidence', 0.0),
                        'date': patch_data.get('date'),
                        'utm_tile': patch_data.get('utm_tile'),
                        'patch_index': patch_data.get('patch_index', 0),
                        'tile_number': patch_data.get('tile_number', 0),
                        'location_id': patch_data.get('location_id', '')
                    }
                    
                    self.all_locations.append(location_info)
                    
                    if location_info['has_debris']:
                        self.debris_locations.append(location_info)
        else:
            # Aggregated format
            print("[INFO] Using aggregated location data format")
            self.use_per_patch = False
            self.geolocation_map = self.data['geolocation_map']
            self.metadata = self.data.get('metadata', {})
            
            # Extract debris locations with coordinates
            self.debris_locations = []
            self.all_locations = []
            
            for location_id, location_data in self.geolocation_map.items():
                if isinstance(location_data, dict):
                    # Detailed format
                    coords = location_data.get('coordinates')
                    if coords and coords['lat'] is not None and coords['lon'] is not None:
                        location_info = {
                            'id': location_id,
                            'lat': coords['lat'],
                            'lon': coords['lon'],
                            'has_debris': location_data.get('has_debris', False),
                            'confidence': location_data.get('confidence', 0.0),
                            'date': location_data.get('date'),
                            'utm_tile': location_data.get('utm_tile')
                        }
                        
                        self.all_locations.append(location_info)
                        
                        if location_info['has_debris']:
                            self.debris_locations.append(location_info)
        
        print(f"[OK] Loaded {len(self.all_locations)} {'patches' if self.use_per_patch else 'locations'}")
        print(f"[OK] Found {len(self.debris_locations)} {'patches' if self.use_per_patch else 'locations'} with debris")
    
    
    def create_world_map(self, output_file: str = 'debris_world_map.html'):
        """
        Create an interactive world map showing all debris locations
        
        Red markers: Debris detected
        Green markers: No debris
        """
        print(f"\nCreating world map visualization...")
        
        if not self.all_locations:
            print("[WARNING] No locations with coordinates found")
            return
        
        # Calculate map center (mean of all locations)
        center_lat = np.mean([loc['lat'] for loc in self.all_locations])
        center_lon = np.mean([loc['lon'] for loc in self.all_locations])
        
        # Create map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=3,
            tiles='OpenStreetMap'
        )
        
        # Add markers for each location
        debris_count = 0
        clean_count = 0
        
        for loc in self.all_locations:
            if loc['has_debris']:
                # Red marker for debris
                color = 'red'
                icon = 'warning-sign'
                debris_count += 1
            else:
                # Green marker for clean
                color = 'green'
                icon = 'ok-sign'
                clean_count += 1
            
            # Build popup HTML
            if self.use_per_patch:
                popup_html = f"""
                <div style="font-family: Arial; width: 220px;">
                    <h4 style="margin: 5px 0; color: {'#d32f2f' if loc['has_debris'] else '#388e3c'};">
                        {'⚠️ DEBRIS DETECTED' if loc['has_debris'] else '✓ Clean'}
                    </h4>
                    <hr style="margin: 5px 0;">
                    <b>Patch:</b> {loc['id']}<br>
                    <b>Location:</b> {loc.get('location_id', 'N/A')}<br>
                    <b>Tile #:</b> {loc.get('tile_number', 'N/A')}<br>
                    <b>Test Index:</b> {loc.get('patch_index', 'N/A')}<br>
                    <b>Coordinates:</b> ({loc['lat']:.2f}, {loc['lon']:.2f})<br>
                    <b>Confidence:</b> {loc['confidence']:.1%}<br>
                    <b>Date:</b> {loc['date'] or 'N/A'}<br>
                    <b>UTM Tile:</b> {loc['utm_tile'] or 'N/A'}
                </div>
                """
            else:
                popup_html = f"""
                <div style="font-family: Arial; width: 200px;">
                    <h4 style="margin: 5px 0; color: {'#d32f2f' if loc['has_debris'] else '#388e3c'};">
                        {'⚠️ DEBRIS DETECTED' if loc['has_debris'] else '✓ Clean'}
                    </h4>
                    <hr style="margin: 5px 0;">
                    <b>Location:</b> {loc['id']}<br>
                    <b>Coordinates:</b> ({loc['lat']:.2f}, {loc['lon']:.2f})<br>
                    <b>Confidence:</b> {loc['confidence']:.1%}<br>
                    <b>Date:</b> {loc['date'] or 'N/A'}<br>
                    <b>UTM Tile:</b> {loc['utm_tile'] or 'N/A'}
                </div>
                """
            
            folium.Marker(
                location=[loc['lat'], loc['lon']],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"{loc['id']} ({'Debris' if loc['has_debris'] else 'Clean'})",
                icon=folium.Icon(color=color, icon=icon, prefix='glyphicon')
            ).add_to(m)
        
        # Add marker cluster for better performance with many markers
        # marker_cluster = plugins.MarkerCluster().add_to(m)
        
        # Add legend
        legend_title = "Marine Debris Map (Test Patches)" if self.use_per_patch else "Marine Debris Map"
        legend_html = f'''
        <div style="position: fixed; 
                    top: 10px; right: 10px; width: 220px; 
                    background-color: white; border: 2px solid grey;
                    z-index: 9999; font-size: 14px; padding: 10px;
                    box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
            <h4 style="margin: 0 0 10px 0;">{legend_title}</h4>
            <p style="margin: 5px 0;">
                <i class="glyphicon glyphicon-warning-sign" style="color: red;"></i> 
                Debris Detected: <b>{debris_count}</b>
            </p>
            <p style="margin: 5px 0;">
                <i class="glyphicon glyphicon-ok-sign" style="color: green;"></i> 
                Clean {'Patches' if self.use_per_patch else 'Areas'}: <b>{clean_count}</b>
            </p>
            <hr style="margin: 5px 0;">
            <p style="margin: 5px 0; font-size: 11px;">
                <b>Dataset:</b> {'Per-patch (208)' if self.use_per_patch else 'Aggregated (11)'}<br>
                <b>Threshold:</b> 0.45<br>
                <b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}
            </p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add fullscreen button
        plugins.Fullscreen().add_to(m)
        
        # Save map
        m.save(output_file)
        print(f"[OK] World map saved to: {output_file}")
        if self.use_per_patch:
            print(f"    - Debris patches: {debris_count}")
            print(f"    - Clean patches: {clean_count}")
        else:
            print(f"    - Debris locations: {debris_count}")
            print(f"    - Clean locations: {clean_count}")
        
        return m
    
    
    def create_route_map(self, route: List[Dict], output_file: str = 'debris_route_map.html'):
        """
        Create map showing the optimal collection route
        
        Args:
            route: List of location dicts in visit order
            output_file: Output HTML file path
        """
        print(f"\nCreating route map visualization...")
        
        if not route or len(route) < 2:
            print("[WARNING] Route too short to visualize")
            return
        
        # Calculate map center
        center_lat = np.mean([loc['lat'] for loc in route])
        center_lon = np.mean([loc['lon'] for loc in route])
        
        # Create map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=4,
            tiles='OpenStreetMap'
        )
        
        # Add route line
        route_coords = [(loc['lat'], loc['lon']) for loc in route]
        
        folium.PolyLine(
            route_coords,
            color='blue',
            weight=3,
            opacity=0.8,
            popup='Collection Route'
        ).add_to(m)
        
        # Add numbered markers for each stop
        for idx, loc in enumerate(route):
            # Start point: Green
            if idx == 0:
                color = 'green'
                icon_color = 'white'
                icon = 'play'
            # End point: Black
            elif idx == len(route) - 1:
                color = 'black'
                icon_color = 'white'
                icon = 'stop'
            # Debris stops: Red
            else:
                color = 'red'
                icon_color = 'white'
                icon = 'trash'
            
            popup_html = f"""
            <div style="font-family: Arial; width: 220px;">
                <h4 style="margin: 5px 0; color: #1976d2;">
                    Stop #{idx + 1}
                </h4>
                <hr style="margin: 5px 0;">
                <b>Location:</b> {loc.get('id', 'Start/End')}<br>
                <b>Coordinates:</b> ({loc['lat']:.2f}, {loc['lon']:.2f})<br>
                {'<b>Prize:</b> ' + str(loc.get('prize', 0)) + ' debris units<br>' if 'prize' in loc else ''}
                {'<b>Confidence:</b> ' + f"{loc.get('confidence', 0):.1%}" + '<br>' if 'confidence' in loc else ''}
            </div>
            """
            
            folium.Marker(
                location=[loc['lat'], loc['lon']],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"Stop {idx + 1}: {loc.get('id', 'Start/End')}",
                icon=folium.Icon(color=color, icon=icon, prefix='glyphicon', icon_color=icon_color)
            ).add_to(m)
        
        # Add legend with route info
        total_distance = sum(
            self._haversine_distance(
                route[i]['lat'], route[i]['lon'],
                route[i+1]['lat'], route[i+1]['lon']
            )
            for i in range(len(route) - 1)
        )
        
        total_prize = sum(loc.get('prize', 0) for loc in route)
        
        legend_html = f'''
        <div style="position: fixed; 
                    top: 10px; right: 10px; width: 250px; 
                    background-color: white; border: 2px solid grey;
                    z-index: 9999; font-size: 14px; padding: 10px;
                    box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
            <h4 style="margin: 0 0 10px 0;">Collection Route</h4>
            <p style="margin: 5px 0;"><b>Total Stops:</b> {len(route)}</p>
            <p style="margin: 5px 0;"><b>Total Distance:</b> {total_distance:.1f} km</p>
            <p style="margin: 5px 0;"><b>Total Prize:</b> {total_prize}</p>
            <hr style="margin: 5px 0;">
            <p style="margin: 5px 0; font-size: 12px;">
                <i class="glyphicon glyphicon-play" style="color: green;"></i> Start Point<br>
                <i class="glyphicon glyphicon-trash" style="color: red;"></i> Debris Collection<br>
                <i class="glyphicon glyphicon-stop" style="color: black;"></i> End Point
            </p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add fullscreen button
        plugins.Fullscreen().add_to(m)
        
        # Save map
        m.save(output_file)
        print(f"[OK] Route map saved to: {output_file}")
        print(f"    - Total stops: {len(route)}")
        print(f"    - Total distance: {total_distance:.1f} km")
        print(f"    - Total prize: {total_prize}")
        
        return m
    
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points on Earth in kilometers"""
        R = 6371  # Earth radius in km
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c


class DebrisRoutePlanner:
    """Plans optimal debris collection routes using Orienteering Problem approach"""
    
    def __init__(self, visualizer: DebrisRouteVisualizer):
        self.visualizer = visualizer
        self.debris_locations = visualizer.debris_locations
    
    
    def plan_collection_route(
        self, 
        start_location: Dict,
        budget_km: float = 1000.0,
        return_to_start: bool = True
    ) -> Tuple[List[Dict], float, float]:
        """
        Plan optimal debris collection route
        
        Args:
            start_location: Starting location dict with 'lat', 'lon'
            budget_km: Maximum travel distance in kilometers
            return_to_start: Whether to return to start location
        
        Returns:
            (route, total_distance, total_prize)
        """
        print(f"\nPlanning debris collection route...")
        print(f"  Start location: ({start_location['lat']:.2f}, {start_location['lon']:.2f})")
        print(f"  Budget: {budget_km:.1f} km")
        print(f"  Return to start: {return_to_start}")
        
        if not self.debris_locations:
            print("[WARNING] No debris locations to visit")
            return [start_location], 0.0, 0.0
        
        # Add prize to debris locations (use confidence as proxy)
        nodes = [start_location]
        for loc in self.debris_locations:
            node = loc.copy()
            # Prize based on confidence (higher confidence = higher priority)
            node['prize'] = int(node.get('confidence', 0.5) * 100)
            nodes.append(node)
        
        # Greedy insertion heuristic for Orienteering Problem
        tour = [nodes[0]]
        if return_to_start:
            tour.append(nodes[0])
        
        total_prize = 0
        total_distance = 0
        unvisited = set(range(1, len(nodes)))
        
        while True:
            best_insertion = None
            best_score = -1
            
            for node_idx in unvisited:
                node = nodes[node_idx]
                
                # Try inserting at each position in tour
                for i in range(len(tour) - (1 if return_to_start else 0)):
                    prev_node = tour[i]
                    next_node = tour[i+1] if i+1 < len(tour) else tour[-1]
                    
                    # Calculate cost increase
                    cost_to_node = self._distance(prev_node, node)
                    cost_from_node = self._distance(node, next_node)
                    cost_direct = self._distance(prev_node, next_node)
                    cost_increase = cost_to_node + cost_from_node - cost_direct
                    
                    # Check budget constraint
                    if total_distance + cost_increase <= budget_km:
                        # Score: prize per unit distance
                        score = node['prize'] / (cost_increase + 1e-6)
                        
                        if score > best_score:
                            best_score = score
                            best_insertion = {
                                'node': node,
                                'index': i + 1,
                                'cost': cost_increase,
                                'node_idx': node_idx
                            }
            
            # Insert best node
            if best_insertion:
                node = best_insertion['node']
                tour.insert(best_insertion['index'], node)
                total_distance += best_insertion['cost']
                total_prize += node['prize']
                unvisited.remove(best_insertion['node_idx'])
                
                print(f"  + Added {node['id']} (prize: {node['prize']}, distance: +{best_insertion['cost']:.1f} km)")
            else:
                break
        
        print(f"\n[OK] Route planning complete")
        print(f"    - Stops: {len(tour)}")
        print(f"    - Total distance: {total_distance:.1f} km (budget: {budget_km:.1f} km)")
        print(f"    - Total prize: {total_prize}")
        if total_distance > 0:
            print(f"    - Efficiency: {total_prize/total_distance:.2f} prize/km")
        
        return tour, total_distance, total_prize
    
    
    def _distance(self, loc1: Dict, loc2: Dict) -> float:
        """Calculate distance between two locations in km"""
        return self.visualizer._haversine_distance(
            loc1['lat'], loc1['lon'],
            loc2['lat'], loc2['lon']
        )
    
    
    def get_initial_direction(self, route: List[Dict]) -> str:
        """Get compass direction from start to first stop"""
        if len(route) < 2:
            return "Stationary"
        
        start = route[0]
        first_stop = route[1]
        
        dlat = first_stop['lat'] - start['lat']
        dlon = first_stop['lon'] - start['lon']
        
        if abs(dlat) < 0.01 and abs(dlon) < 0.01:
            return "Stationary"
        
        # Calculate bearing
        angle = math.degrees(math.atan2(dlon, dlat))
        
        # Normalize to [0, 360)
        if angle < 0:
            angle += 360
        
        # Convert to compass directions
        directions = [
            "North", "North-East", "East", "South-East",
            "South", "South-West", "West", "North-West"
        ]
        
        idx = int((angle + 22.5) / 45) % 8
        return directions[idx]


def main():
    parser = argparse.ArgumentParser(description='Marine Debris Route Planner and Visualizer')
    parser.add_argument('--geolocation_map', default='debris_geolocation_map.json',
                       help='Path to geolocation map JSON file (aggregated or per-patch)')
    parser.add_argument('--per_patch', action='store_true',
                       help='Use per-patch data (test_patch_predictions.json)')
    parser.add_argument('--start_lat', type=float, default=12.0,
                       help='Starting latitude')
    parser.add_argument('--start_lon', type=float, default=-87.0,
                       help='Starting longitude')
    parser.add_argument('--budget', type=float, default=1000.0,
                       help='Travel budget in kilometers')
    parser.add_argument('--no_return', action='store_true',
                       help='Do not return to start location')
    parser.add_argument('--skip_route', action='store_true',
                       help='Skip route planning, only create world map')
    
    args = parser.parse_args()
    
    # Auto-select per-patch file if flag is set
    if args.per_patch:
        args.geolocation_map = 'test_patch_predictions.json'
    
    print("="*70)
    print("MARINE DEBRIS COLLECTION ROUTE PLANNER")
    print("="*70)
    
    # Initialize visualizer
    visualizer = DebrisRouteVisualizer(args.geolocation_map)
    
    # Create world map
    visualizer.create_world_map('marine_debris_world_map.html')
    
    if not args.skip_route and visualizer.debris_locations:
        # Initialize route planner
        planner = DebrisRoutePlanner(visualizer)
        
        # Plan collection route
        start_location = {
            'id': 'START',
            'lat': args.start_lat,
            'lon': args.start_lon,
            'has_debris': False,
            'prize': 0
        }
        
        route, distance, prize = planner.plan_collection_route(
            start_location,
            budget_km=args.budget,
            return_to_start=not args.no_return
        )
        
        # Get initial direction
        if len(route) > 1:
            direction = planner.get_initial_direction(route)
            print(f"\n📍 Initial heading: {direction}")
            print(f"   First stop: {route[1]['id']} at ({route[1]['lat']:.2f}, {route[1]['lon']:.2f})")
        
        # Create route map
        visualizer.create_route_map(route, 'marine_debris_route_map.html')
        
        # Print route summary
        print("\n" + "="*70)
        print("ROUTE SUMMARY")
        print("="*70)
        for idx, stop in enumerate(route):
            if idx == 0:
                print(f"START: {stop['id']} at ({stop['lat']:.2f}, {stop['lon']:.2f})")
            elif idx == len(route) - 1 and stop['id'] == 'START':
                print(f"END: Return to start")
            else:
                print(f"Stop {idx}: {stop['id']} - Prize: {stop.get('prize', 0)} - Confidence: {stop.get('confidence', 0):.1%}")
        print("="*70)
    
    print(f"\n✓ Complete! Open the HTML files in your browser:")
    print(f"  - World Map: marine_debris_world_map.html")
    if not args.skip_route and visualizer.debris_locations:
        print(f"  - Route Map: marine_debris_route_map.html")


if __name__ == '__main__':
    main()
