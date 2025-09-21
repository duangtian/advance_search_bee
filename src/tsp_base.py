"""
TSP Base Classes and Utilities

This module contains the fundamental classes and utility functions
for representing and solving the Traveling Salesman Problem.
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class City:
    """Represents a city with coordinates and an identifier."""
    id: int
    x: float
    y: float
    name: Optional[str] = None

    def distance_to(self, other: 'City') -> float:
        """Calculate Euclidean distance to another city."""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def __str__(self) -> str:
        return f"City({self.id}, {self.x:.2f}, {self.y:.2f})"


class TSPInstance:
    """Represents a TSP problem instance with cities and distance matrix."""
    
    def __init__(self, cities: List[City]):
        self.cities = cities
        self.num_cities = len(cities)
        self.distance_matrix = self._build_distance_matrix()
    
    def _build_distance_matrix(self) -> np.ndarray:
        """Build a symmetric distance matrix between all cities."""
        matrix = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(i + 1, self.num_cities):
                dist = self.cities[i].distance_to(self.cities[j])
                matrix[i][j] = matrix[j][i] = dist
        return matrix
    
    def get_distance(self, city1_idx: int, city2_idx: int) -> float:
        """Get distance between two cities by their indices."""
        return self.distance_matrix[city1_idx][city2_idx]
    
    def calculate_tour_distance(self, tour: List[int]) -> float:
        """Calculate total distance for a given tour."""
        if len(tour) != self.num_cities:
            raise ValueError(f"Tour must contain exactly {self.num_cities} cities")
        
        total_distance = 0.0
        for i in range(len(tour)):
            current_city = tour[i]
            next_city = tour[(i + 1) % len(tour)]  # Return to start
            total_distance += self.get_distance(current_city, next_city)
        
        return total_distance
    
    def is_valid_tour(self, tour: List[int]) -> bool:
        """Check if a tour is valid (visits each city exactly once)."""
        return (len(tour) == self.num_cities and 
                set(tour) == set(range(self.num_cities)))
    
    def generate_random_tour(self) -> List[int]:
        """Generate a random valid tour."""
        tour = list(range(self.num_cities))
        random.shuffle(tour)
        return tour
    
    def get_nearest_neighbor_tour(self, start_city: int = 0) -> List[int]:
        """Generate a tour using nearest neighbor heuristic."""
        tour = [start_city]
        unvisited = set(range(self.num_cities)) - {start_city}
        
        current_city = start_city
        while unvisited:
            nearest_city = min(unvisited, 
                             key=lambda city: self.get_distance(current_city, city))
            tour.append(nearest_city)
            unvisited.remove(nearest_city)
            current_city = nearest_city
        
        return tour
    
    def visualize_tour(self, tour: List[int], title: str = "TSP Tour", 
                      save_path: Optional[str] = None) -> None:
        """Visualize a tour on a 2D plot."""
        if not self.is_valid_tour(tour):
            raise ValueError("Invalid tour provided")
        
        plt.figure(figsize=(10, 8))
        
        # Plot cities
        x_coords = [self.cities[i].x for i in tour]
        y_coords = [self.cities[i].y for i in tour]
        
        # Add the starting city at the end to complete the loop
        x_coords.append(x_coords[0])
        y_coords.append(y_coords[0])
        
        # Plot the tour path
        plt.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.7, label='Tour Path')
        
        # Plot cities as points
        for i, city in enumerate(self.cities):
            plt.scatter(city.x, city.y, c='red', s=100, zorder=5)
            plt.annotate(str(city.id), (city.x, city.y), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
        distance = self.calculate_tour_distance(tour)
        plt.title(f"{title}\nTotal Distance: {distance:.2f}")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class TSPGenerator:
    """Utility class for generating TSP problem instances."""
    
    @staticmethod
    def generate_random_cities(num_cities: int, width: float = 100.0, 
                              height: float = 100.0, seed: Optional[int] = None) -> List[City]:
        """Generate random cities within a rectangular area."""
        if seed is not None:
            random.seed(seed)
        
        cities = []
        for i in range(num_cities):
            x = random.uniform(0, width)
            y = random.uniform(0, height)
            cities.append(City(id=i, x=x, y=y))
        
        return cities
    
    @staticmethod
    def generate_circular_cities(num_cities: int, radius: float = 50.0, 
                               center: Tuple[float, float] = (50.0, 50.0),
                               noise: float = 0.0) -> List[City]:
        """Generate cities arranged in a circle with optional noise."""
        cities = []
        angle_step = 2 * math.pi / num_cities
        
        for i in range(num_cities):
            angle = i * angle_step
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            
            # Add noise if specified
            if noise > 0:
                x += random.uniform(-noise, noise)
                y += random.uniform(-noise, noise)
            
            cities.append(City(id=i, x=x, y=y))
        
        return cities
    
    @staticmethod
    def generate_clustered_cities(num_cities: int, num_clusters: int = 3,
                                cluster_radius: float = 15.0,
                                area_width: float = 100.0,
                                area_height: float = 100.0) -> List[City]:
        """Generate cities in clusters."""
        cities = []
        cities_per_cluster = num_cities // num_clusters
        remaining_cities = num_cities % num_clusters
        
        # Generate cluster centers
        cluster_centers = []
        for _ in range(num_clusters):
            center_x = random.uniform(cluster_radius, area_width - cluster_radius)
            center_y = random.uniform(cluster_radius, area_height - cluster_radius)
            cluster_centers.append((center_x, center_y))
        
        city_id = 0
        for cluster_idx in range(num_clusters):
            center_x, center_y = cluster_centers[cluster_idx]
            cities_in_this_cluster = cities_per_cluster
            if cluster_idx < remaining_cities:
                cities_in_this_cluster += 1
            
            for _ in range(cities_in_this_cluster):
                # Generate city within cluster radius
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(0, cluster_radius)
                x = center_x + distance * math.cos(angle)
                y = center_y + distance * math.sin(angle)
                
                cities.append(City(id=city_id, x=x, y=y))
                city_id += 1
        
        return cities


def load_tsp_file(filepath: str) -> TSPInstance:
    """Load TSP instance from a standard TSPLIB format file."""
    # This is a simplified loader - real TSPLIB files have more complex formats
    cities = []
    with open(filepath, 'r') as file:
        lines = file.readlines()
        
        # Find the coordinate section
        coord_start = -1
        for i, line in enumerate(lines):
            if 'NODE_COORD_SECTION' in line:
                coord_start = i + 1
                break
        
        if coord_start == -1:
            raise ValueError("Could not find coordinate section in file")
        
        for line in lines[coord_start:]:
            line = line.strip()
            if line == 'EOF' or not line:
                break
            
            parts = line.split()
            if len(parts) >= 3:
                city_id = int(parts[0]) - 1  # Convert to 0-based indexing
                x = float(parts[1])
                y = float(parts[2])
                cities.append(City(id=city_id, x=x, y=y))
    
    return TSPInstance(cities)


# Utility functions for algorithm evaluation
def evaluate_solution(tsp_instance: TSPInstance, tour: List[int]) -> dict:
    """Evaluate a solution and return performance metrics."""
    if not tsp_instance.is_valid_tour(tour):
        raise ValueError("Invalid tour provided")
    
    distance = tsp_instance.calculate_tour_distance(tour)
    
    # Calculate some additional metrics
    metrics = {
        'distance': distance,
        'num_cities': len(tour),
        'tour': tour.copy()
    }
    
    return metrics


def compare_algorithms_results(results: dict) -> None:
    """Compare results from multiple algorithms."""
    print("\n" + "="*60)
    print("ALGORITHM COMPARISON RESULTS")
    print("="*60)
    
    algorithms = list(results.keys())
    best_distance = min(results[alg]['distance'] for alg in algorithms)
    
    for algorithm in algorithms:
        result = results[algorithm]
        distance = result['distance']
        improvement = ((distance - best_distance) / best_distance) * 100
        
        print(f"\n{algorithm}:")
        print(f"  Distance: {distance:.2f}")
        print(f"  Gap from best: {improvement:.2f}%")
        if 'time' in result:
            print(f"  Runtime: {result['time']:.3f} seconds")
        if 'iterations' in result:
            print(f"  Iterations: {result['iterations']}")