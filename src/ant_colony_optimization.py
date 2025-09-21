"""
Ant Colony Optimization for Traveling Salesman Problem

This module implements the Ant Colony System (ACS) algorithm for solving TSP,
inspired by the foraging behavior of ants.
"""

import random
import numpy as np
from typing import List, Tuple, Optional
from .tsp_base import TSPInstance
import time


class AntColonyOptimizationTSP:
    """Ant Colony Optimization implementation for solving TSP."""
    
    def __init__(self, tsp_instance: TSPInstance, num_ants: int = None,
                 alpha: float = 1.0, beta: float = 5.0, rho: float = 0.1,
                 q0: float = 0.9, xi: float = 0.1, initial_pheromone: float = None):
        """
        Initialize the Ant Colony Optimization algorithm.
        
        Args:
            tsp_instance: The TSP problem instance
            num_ants: Number of ants (default: number of cities)
            alpha: Pheromone importance factor
            beta: Heuristic importance factor (distance)
            rho: Global pheromone evaporation rate
            q0: Exploitation vs exploration parameter
            xi: Local pheromone evaporation rate
            initial_pheromone: Initial pheromone level
        """
        self.tsp_instance = tsp_instance
        self.num_cities = tsp_instance.num_cities
        self.num_ants = num_ants or self.num_cities
        
        # Algorithm parameters
        self.alpha = alpha
        self.beta = beta
        self.rho = rho  # Global evaporation rate
        self.q0 = q0    # Exploitation probability
        self.xi = xi    # Local evaporation rate
        
        # Initialize pheromone matrix
        if initial_pheromone is None:
            # Use nearest neighbor tour to estimate initial pheromone
            nn_tour = tsp_instance.get_nearest_neighbor_tour()
            nn_distance = tsp_instance.calculate_tour_distance(nn_tour)
            self.initial_pheromone = 1.0 / (self.num_cities * nn_distance)
        else:
            self.initial_pheromone = initial_pheromone
        
        self.pheromone_matrix = np.full((self.num_cities, self.num_cities), 
                                       self.initial_pheromone)
        
        # Heuristic information (inverse of distance)
        self.heuristic_matrix = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    distance = tsp_instance.get_distance(i, j)
                    self.heuristic_matrix[i][j] = 1.0 / distance if distance > 0 else 0
        
        # Best solution tracking
        self.best_tour = None
        self.best_distance = float('inf')
        self.history = []
    
    def construct_ant_tour(self, ant_id: int) -> List[int]:
        """Construct a tour for a single ant using ACO rules."""
        tour = []
        unvisited = set(range(self.num_cities))
        
        # Start from a random city (or specific starting city)
        current_city = random.randint(0, self.num_cities - 1)
        tour.append(current_city)
        unvisited.remove(current_city)
        
        # Construct the rest of the tour
        while unvisited:
            next_city = self._select_next_city(current_city, unvisited)
            
            # Local pheromone update (ACS rule)
            self._local_pheromone_update(current_city, next_city)
            
            tour.append(next_city)
            unvisited.remove(next_city)
            current_city = next_city
        
        return tour
    
    def _select_next_city(self, current_city: int, unvisited: set) -> int:
        """Select the next city for an ant to visit."""
        if random.random() < self.q0:
            # Exploitation: choose the best city
            return self._exploitation_selection(current_city, unvisited)
        else:
            # Exploration: probabilistic selection
            return self._exploration_selection(current_city, unvisited)
    
    def _exploitation_selection(self, current_city: int, unvisited: set) -> int:
        """Select the city with highest pheromone * heuristic value."""
        best_city = None
        best_value = -1
        
        for city in unvisited:
            pheromone = self.pheromone_matrix[current_city][city]
            heuristic = self.heuristic_matrix[current_city][city]
            value = pheromone * (heuristic ** self.beta)
            
            if value > best_value:
                best_value = value
                best_city = city
        
        return best_city
    
    def _exploration_selection(self, current_city: int, unvisited: set) -> int:
        """Select the next city probabilistically."""
        probabilities = []
        cities = list(unvisited)
        
        # Calculate selection probabilities
        total_weight = 0
        for city in cities:
            pheromone = self.pheromone_matrix[current_city][city]
            heuristic = self.heuristic_matrix[current_city][city]
            weight = (pheromone ** self.alpha) * (heuristic ** self.beta)
            probabilities.append(weight)
            total_weight += weight
        
        if total_weight == 0:
            return random.choice(cities)
        
        # Normalize probabilities
        probabilities = [p / total_weight for p in probabilities]
        
        # Roulette wheel selection
        r = random.random()
        cumulative_prob = 0
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if r <= cumulative_prob:
                return cities[i]
        
        return cities[-1]  # Fallback
    
    def _local_pheromone_update(self, city1: int, city2: int) -> None:
        """Perform local pheromone update (ACS rule)."""
        current_pheromone = self.pheromone_matrix[city1][city2]
        new_pheromone = (1 - self.xi) * current_pheromone + self.xi * self.initial_pheromone
        
        self.pheromone_matrix[city1][city2] = new_pheromone
        self.pheromone_matrix[city2][city1] = new_pheromone  # Symmetric
    
    def global_pheromone_update(self, best_tour: List[int], best_distance: float) -> None:
        """Perform global pheromone update on the best tour."""
        # Evaporation
        self.pheromone_matrix *= (1 - self.rho)
        
        # Reinforce best tour
        pheromone_deposit = 1.0 / best_distance
        for i in range(len(best_tour)):
            city1 = best_tour[i]
            city2 = best_tour[(i + 1) % len(best_tour)]
            
            self.pheromone_matrix[city1][city2] += self.rho * pheromone_deposit
            self.pheromone_matrix[city2][city1] += self.rho * pheromone_deposit
    
    def apply_local_search(self, tour: List[int]) -> List[int]:
        """Apply 2-opt local search to improve a tour."""
        improved_tour = tour.copy()
        improved = True
        
        while improved:
            improved = False
            current_distance = self.tsp_instance.calculate_tour_distance(improved_tour)
            
            for i in range(len(tour) - 1):
                for j in range(i + 2, len(tour)):
                    # Skip if it would reverse the entire tour
                    if j == len(tour) - 1 and i == 0:
                        continue
                    
                    # Create new tour by reversing segment
                    new_tour = improved_tour.copy()
                    new_tour[i+1:j+1] = new_tour[i+1:j+1][::-1]
                    
                    new_distance = self.tsp_instance.calculate_tour_distance(new_tour)
                    if new_distance < current_distance:
                        improved_tour = new_tour
                        current_distance = new_distance
                        improved = True
                        break
                
                if improved:
                    break
        
        return improved_tour
    
    def solve(self, max_iterations: int = 1000, convergence_threshold: int = 100,
              apply_local_search: bool = True, verbose: bool = True) -> dict:
        """
        Solve the TSP using Ant Colony Optimization.
        
        Args:
            max_iterations: Maximum number of iterations
            convergence_threshold: Stop if no improvement for this many iterations
            apply_local_search: Whether to apply local search to solutions
            verbose: Whether to print progress information
        
        Returns:
            Dictionary containing the solution and statistics
        """
        start_time = time.time()
        
        no_improvement_count = 0
        last_best_distance = self.best_distance
        
        if verbose:
            print(f"Starting ACO with {self.num_ants} ants...")
        
        for iteration in range(max_iterations):
            iteration_tours = []
            iteration_distances = []
            
            # Each ant constructs a tour
            for ant in range(self.num_ants):
                tour = self.construct_ant_tour(ant)
                distance = self.tsp_instance.calculate_tour_distance(tour)
                
                # Apply local search if enabled
                if apply_local_search:
                    tour = self.apply_local_search(tour)
                    distance = self.tsp_instance.calculate_tour_distance(tour)
                
                iteration_tours.append(tour)
                iteration_distances.append(distance)
                
                # Update global best
                if distance < self.best_distance:
                    self.best_distance = distance
                    self.best_tour = tour.copy()
            
            # Find best tour of this iteration
            best_iteration_idx = np.argmin(iteration_distances)
            best_iteration_tour = iteration_tours[best_iteration_idx]
            best_iteration_distance = iteration_distances[best_iteration_idx]
            
            # Global pheromone update
            self.global_pheromone_update(best_iteration_tour, best_iteration_distance)
            
            # Track progress
            avg_distance = np.mean(iteration_distances)
            self.history.append({
                'iteration': iteration,
                'best_distance': self.best_distance,
                'iteration_best': best_iteration_distance,
                'avg_distance': avg_distance,
                'pheromone_variance': np.var(self.pheromone_matrix)
            })
            
            # Check for improvement
            if self.best_distance < last_best_distance:
                no_improvement_count = 0
                last_best_distance = self.best_distance
            else:
                no_improvement_count += 1
            
            # Print progress
            if verbose and iteration % 50 == 0:
                print(f"Iteration {iteration}: Best = {self.best_distance:.2f}, "
                      f"Avg = {avg_distance:.2f}")
            
            # Check convergence
            if no_improvement_count >= convergence_threshold:
                if verbose:
                    print(f"Converged after {iteration} iterations")
                break
        
        end_time = time.time()
        
        result = {
            'tour': self.best_tour,
            'distance': self.best_distance,
            'iterations': iteration + 1,
            'time': end_time - start_time,
            'history': self.history,
            'algorithm': 'Ant Colony Optimization',
            'final_pheromone_matrix': self.pheromone_matrix.copy()
        }
        
        if verbose:
            print(f"Final best distance: {self.best_distance:.2f}")
            print(f"Total runtime: {result['time']:.3f} seconds")
        
        return result
    
    def get_convergence_plot(self) -> None:
        """Plot the convergence history."""
        import matplotlib.pyplot as plt
        
        iterations = [entry['iteration'] for entry in self.history]
        best_distances = [entry['best_distance'] for entry in self.history]
        avg_distances = [entry['avg_distance'] for entry in self.history]
        
        plt.figure(figsize=(12, 5))
        
        # Convergence plot
        plt.subplot(1, 2, 1)
        plt.plot(iterations, best_distances, 'b-', linewidth=2, label='Best Distance')
        plt.plot(iterations, avg_distances, 'r--', alpha=0.7, label='Average Distance')
        plt.title('ACO Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Distance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Pheromone variance plot
        plt.subplot(1, 2, 2)
        pheromone_vars = [entry['pheromone_variance'] for entry in self.history]
        plt.plot(iterations, pheromone_vars, 'g-', linewidth=2)
        plt.title('Pheromone Variance')
        plt.xlabel('Iteration')
        plt.ylabel('Variance')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_pheromone_matrix(self) -> None:
        """Visualize the final pheromone matrix as a heatmap."""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 8))
        plt.imshow(self.pheromone_matrix, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Pheromone Level')
        plt.title('Pheromone Matrix Heatmap')
        plt.xlabel('City Index')
        plt.ylabel('City Index')
        plt.show()


class MaxMinAntSystem(AntColonyOptimizationTSP):
    """
    Max-Min Ant System variant of ACO with pheromone bounds.
    """
    
    def __init__(self, tsp_instance: TSPInstance, num_ants: int = None,
                 alpha: float = 1.0, beta: float = 2.0, rho: float = 0.02,
                 **kwargs):
        super().__init__(tsp_instance, num_ants, alpha, beta, rho, **kwargs)
        
        # Calculate pheromone bounds
        nn_tour = tsp_instance.get_nearest_neighbor_tour()
        nn_distance = tsp_instance.calculate_tour_distance(nn_tour)
        
        self.tau_max = 1.0 / (rho * nn_distance)
        self.tau_min = self.tau_max / (2 * self.num_cities)
        
        # Initialize pheromone matrix to tau_max
        self.pheromone_matrix.fill(self.tau_max)
    
    def global_pheromone_update(self, best_tour: List[int], best_distance: float) -> None:
        """Perform global pheromone update with bounds."""
        # Evaporation
        self.pheromone_matrix *= (1 - self.rho)
        
        # Reinforce best tour
        pheromone_deposit = 1.0 / best_distance
        for i in range(len(best_tour)):
            city1 = best_tour[i]
            city2 = best_tour[(i + 1) % len(best_tour)]
            
            self.pheromone_matrix[city1][city2] += self.rho * pheromone_deposit
            self.pheromone_matrix[city2][city1] += self.rho * pheromone_deposit
        
        # Apply bounds
        self.pheromone_matrix = np.clip(self.pheromone_matrix, self.tau_min, self.tau_max)


# Example usage and testing functions
def test_ant_colony_optimization():
    """Test the Ant Colony Optimization with a sample problem."""
    from .tsp_base import TSPGenerator
    
    # Generate a test problem
    cities = TSPGenerator.generate_random_cities(20, seed=42)
    tsp_instance = TSPInstance(cities)
    
    # Create and run ACO
    aco = AntColonyOptimizationTSP(tsp_instance, num_ants=20)
    result = aco.solve(max_iterations=200, verbose=True)
    
    # Visualize results
    tsp_instance.visualize_tour(result['tour'], 
                               title=f"ACO Solution (Distance: {result['distance']:.2f})")
    aco.get_convergence_plot()
    aco.visualize_pheromone_matrix()
    
    return result


def test_max_min_ant_system():
    """Test the Max-Min Ant System variant."""
    from .tsp_base import TSPGenerator
    
    # Generate a test problem
    cities = TSPGenerator.generate_random_cities(20, seed=42)
    tsp_instance = TSPInstance(cities)
    
    # Create and run MMAS
    mmas = MaxMinAntSystem(tsp_instance, num_ants=20)
    result = mmas.solve(max_iterations=200, verbose=True)
    
    # Visualize results
    tsp_instance.visualize_tour(result['tour'], 
                               title=f"MMAS Solution (Distance: {result['distance']:.2f})")
    mmas.get_convergence_plot()
    
    return result


if __name__ == "__main__":
    print("Testing Ant Colony Optimization...")
    test_ant_colony_optimization()
    
    print("\nTesting Max-Min Ant System...")
    test_max_min_ant_system()