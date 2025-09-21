"""
Simulated Annealing for Traveling Salesman Problem

This module implements Simulated Annealing with various cooling schedules
and neighborhood search operators for solving TSP.
"""

import random
import math
import numpy as np
from typing import List, Tuple, Optional, Callable
from .tsp_base import TSPInstance
import time


class SimulatedAnnealingTSP:
    """Simulated Annealing implementation for solving TSP."""
    
    def __init__(self, tsp_instance: TSPInstance, initial_temperature: float = None,
                 final_temperature: float = 0.01, cooling_rate: float = 0.95):
        """
        Initialize the Simulated Annealing algorithm.
        
        Args:
            tsp_instance: The TSP problem instance
            initial_temperature: Initial temperature (auto-calculated if None)
            final_temperature: Final temperature
            cooling_rate: Cooling rate for geometric cooling
        """
        self.tsp_instance = tsp_instance
        self.num_cities = tsp_instance.num_cities
        
        # Temperature parameters
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.cooling_rate = cooling_rate
        
        # Current solution
        self.current_tour = None
        self.current_cost = float('inf')
        
        # Best solution found
        self.best_tour = None
        self.best_cost = float('inf')
        
        # History tracking
        self.history = []
        
        # If initial temperature not provided, estimate it
        if self.initial_temperature is None:
            self.initial_temperature = self._estimate_initial_temperature()
    
    def _estimate_initial_temperature(self, sample_size: int = 100) -> float:
        """
        Estimate initial temperature based on the average cost difference
        of random neighboring solutions.
        """
        # Generate random tour
        sample_tour = self.tsp_instance.generate_random_tour()
        
        cost_differences = []
        for _ in range(sample_size):
            # Generate neighbor
            neighbor = self._get_random_neighbor(sample_tour)
            
            # Calculate cost difference
            original_cost = self.tsp_instance.calculate_tour_distance(sample_tour)
            neighbor_cost = self.tsp_instance.calculate_tour_distance(neighbor)
            cost_diff = abs(neighbor_cost - original_cost)
            
            if cost_diff > 0:
                cost_differences.append(cost_diff)
        
        if not cost_differences:
            return 100.0
        
        avg_cost_diff = np.mean(cost_differences)
        # Set initial temperature so that initially 80% of worse solutions are accepted
        return -avg_cost_diff / math.log(0.8)
    
    def _get_random_neighbor(self, tour: List[int]) -> List[int]:
        """Generate a random neighbor using various operators."""
        operators = [
            self._swap_operator,
            self._two_opt_operator,
            self._insert_operator,
            self._reverse_operator
        ]
        
        operator = random.choice(operators)
        return operator(tour)
    
    def _swap_operator(self, tour: List[int]) -> List[int]:
        """Swap two random cities."""
        neighbor = tour.copy()
        if len(neighbor) > 1:
            i, j = random.sample(range(len(neighbor)), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        return neighbor
    
    def _two_opt_operator(self, tour: List[int]) -> List[int]:
        """Apply 2-opt operation (reverse a segment)."""
        neighbor = tour.copy()
        if len(neighbor) > 3:
            i, j = sorted(random.sample(range(len(neighbor)), 2))
            if j - i > 1:  # Ensure we have a segment to reverse
                neighbor[i:j+1] = neighbor[i:j+1][::-1]
        return neighbor
    
    def _insert_operator(self, tour: List[int]) -> List[int]:
        """Remove a city and insert it at a different position."""
        neighbor = tour.copy()
        if len(neighbor) > 2:
            # Remove city from random position
            remove_idx = random.randint(0, len(neighbor) - 1)
            city = neighbor.pop(remove_idx)
            
            # Insert at random position
            insert_idx = random.randint(0, len(neighbor))
            neighbor.insert(insert_idx, city)
        return neighbor
    
    def _reverse_operator(self, tour: List[int]) -> List[int]:
        """Reverse a random segment of the tour."""
        neighbor = tour.copy()
        if len(neighbor) > 2:
            start = random.randint(0, len(neighbor) - 2)
            end = random.randint(start + 1, len(neighbor) - 1)
            neighbor[start:end+1] = neighbor[start:end+1][::-1]
        return neighbor
    
    def _acceptance_probability(self, current_cost: float, neighbor_cost: float, 
                               temperature: float) -> float:
        """Calculate the acceptance probability for a solution."""
        if neighbor_cost < current_cost:
            return 1.0  # Always accept better solutions
        
        if temperature <= 0:
            return 0.0
        
        return math.exp(-(neighbor_cost - current_cost) / temperature)
    
    def _linear_cooling(self, initial_temp: float, final_temp: float, 
                       current_iteration: int, max_iterations: int) -> float:
        """Linear cooling schedule."""
        return initial_temp - (initial_temp - final_temp) * (current_iteration / max_iterations)
    
    def _geometric_cooling(self, current_temp: float, cooling_rate: float) -> float:
        """Geometric cooling schedule."""
        return current_temp * cooling_rate
    
    def _exponential_cooling(self, initial_temp: float, current_iteration: int, 
                            cooling_factor: float = 0.95) -> float:
        """Exponential cooling schedule."""
        return initial_temp * (cooling_factor ** current_iteration)
    
    def _logarithmic_cooling(self, initial_temp: float, current_iteration: int) -> float:
        """Logarithmic cooling schedule."""
        return initial_temp / (1 + math.log(1 + current_iteration))
    
    def _adaptive_cooling(self, current_temp: float, acceptance_rate: float, 
                         target_acceptance_rate: float = 0.2) -> float:
        """Adaptive cooling based on acceptance rate."""
        if acceptance_rate > target_acceptance_rate:
            return current_temp * 0.9  # Cool faster if accepting too many
        else:
            return current_temp * 0.99  # Cool slower if accepting too few
    
    def solve(self, max_iterations: int = 10000, max_iterations_per_temp: int = 100,
              cooling_schedule: str = 'geometric', reheat_interval: int = 0,
              verbose: bool = True) -> dict:
        """
        Solve the TSP using Simulated Annealing.
        
        Args:
            max_iterations: Maximum total iterations
            max_iterations_per_temp: Iterations per temperature level
            cooling_schedule: Cooling schedule ('linear', 'geometric', 'exponential', 'logarithmic', 'adaptive')
            reheat_interval: Interval for reheating (0 = no reheating)
            verbose: Whether to print progress information
        
        Returns:
            Dictionary containing the solution and statistics
        """
        start_time = time.time()
        
        # Initialize solution
        self.current_tour = self.tsp_instance.get_nearest_neighbor_tour()
        self.current_cost = self.tsp_instance.calculate_tour_distance(self.current_tour)
        
        # Initialize best solution
        self.best_tour = self.current_tour.copy()
        self.best_cost = self.current_cost
        
        # Initialize temperature
        current_temperature = self.initial_temperature
        
        # Statistics
        total_evaluations = 0
        total_accepted = 0
        total_rejected = 0
        last_improvement_iteration = 0
        
        if verbose:
            print(f"Starting Simulated Annealing...")
            print(f"Initial temperature: {current_temperature:.2f}")
            print(f"Initial cost: {self.current_cost:.2f}")
        
        iteration = 0
        while iteration < max_iterations and current_temperature > self.final_temperature:
            # Track acceptance rate for this temperature level
            accepted_at_temp = 0
            iterations_at_temp = 0
            
            # Iterate at current temperature
            for temp_iter in range(max_iterations_per_temp):
                if iteration >= max_iterations:
                    break
                
                # Generate neighbor
                neighbor_tour = self._get_random_neighbor(self.current_tour)
                neighbor_cost = self.tsp_instance.calculate_tour_distance(neighbor_tour)
                
                total_evaluations += 1
                
                # Calculate acceptance probability
                accept_prob = self._acceptance_probability(self.current_cost, neighbor_cost, 
                                                         current_temperature)
                
                # Accept or reject
                if random.random() < accept_prob:
                    self.current_tour = neighbor_tour
                    self.current_cost = neighbor_cost
                    total_accepted += 1
                    accepted_at_temp += 1
                    
                    # Update best solution
                    if neighbor_cost < self.best_cost:
                        self.best_cost = neighbor_cost
                        self.best_tour = neighbor_tour.copy()
                        last_improvement_iteration = iteration
                else:
                    total_rejected += 1
                
                iterations_at_temp += 1
                iteration += 1
                
                # Track history every 100 iterations
                if iteration % 100 == 0:
                    acceptance_rate = total_accepted / total_evaluations if total_evaluations > 0 else 0
                    self.history.append({
                        'iteration': iteration,
                        'temperature': current_temperature,
                        'current_cost': self.current_cost,
                        'best_cost': self.best_cost,
                        'acceptance_rate': acceptance_rate
                    })
            
            # Calculate acceptance rate for this temperature
            temp_acceptance_rate = accepted_at_temp / iterations_at_temp if iterations_at_temp > 0 else 0
            
            # Update temperature based on cooling schedule
            if cooling_schedule == 'linear':
                current_temperature = self._linear_cooling(
                    self.initial_temperature, self.final_temperature, 
                    iteration, max_iterations
                )
            elif cooling_schedule == 'geometric':
                current_temperature = self._geometric_cooling(current_temperature, self.cooling_rate)
            elif cooling_schedule == 'exponential':
                current_temperature = self._exponential_cooling(self.initial_temperature, iteration // max_iterations_per_temp)
            elif cooling_schedule == 'logarithmic':
                current_temperature = self._logarithmic_cooling(self.initial_temperature, iteration // max_iterations_per_temp)
            elif cooling_schedule == 'adaptive':
                current_temperature = self._adaptive_cooling(current_temperature, temp_acceptance_rate)
            else:
                current_temperature = self._geometric_cooling(current_temperature, self.cooling_rate)
            
            # Reheating mechanism
            if (reheat_interval > 0 and iteration % reheat_interval == 0 and 
                iteration - last_improvement_iteration > reheat_interval // 2):
                current_temperature = self.initial_temperature * 0.5
                if verbose:
                    print(f"Reheating at iteration {iteration}")
            
            # Print progress
            if verbose and iteration % 1000 == 0:
                acceptance_rate = total_accepted / total_evaluations if total_evaluations > 0 else 0
                print(f"Iteration {iteration}: T = {current_temperature:.3f}, "
                      f"Current = {self.current_cost:.2f}, Best = {self.best_cost:.2f}, "
                      f"Accept Rate = {acceptance_rate:.3f}")
        
        end_time = time.time()
        
        # Final statistics
        final_acceptance_rate = total_accepted / total_evaluations if total_evaluations > 0 else 0
        
        result = {
            'tour': self.best_tour,
            'distance': self.best_cost,
            'iterations': iteration,
            'time': end_time - start_time,
            'history': self.history,
            'algorithm': 'Simulated Annealing',
            'final_temperature': current_temperature,
            'total_evaluations': total_evaluations,
            'acceptance_rate': final_acceptance_rate,
            'cooling_schedule': cooling_schedule
        }
        
        if verbose:
            print(f"Final best distance: {self.best_cost:.2f}")
            print(f"Final temperature: {current_temperature:.6f}")
            print(f"Total evaluations: {total_evaluations}")
            print(f"Overall acceptance rate: {final_acceptance_rate:.3f}")
            print(f"Total runtime: {result['time']:.3f} seconds")
        
        return result
    
    def get_convergence_plot(self) -> None:
        """Plot the convergence history."""
        import matplotlib.pyplot as plt
        
        iterations = [entry['iteration'] for entry in self.history]
        temperatures = [entry['temperature'] for entry in self.history]
        current_costs = [entry['current_cost'] for entry in self.history]
        best_costs = [entry['best_cost'] for entry in self.history]
        acceptance_rates = [entry['acceptance_rate'] for entry in self.history]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Cost convergence
        ax1.plot(iterations, current_costs, 'b-', alpha=0.6, label='Current Cost')
        ax1.plot(iterations, best_costs, 'r-', linewidth=2, label='Best Cost')
        ax1.set_title('Cost Convergence')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Cost')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Temperature schedule
        ax2.semilogy(iterations, temperatures, 'g-', linewidth=2)
        ax2.set_title('Temperature Schedule')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Temperature (log scale)')
        ax2.grid(True, alpha=0.3)
        
        # Acceptance rate
        ax3.plot(iterations, acceptance_rates, 'purple', linewidth=2)
        ax3.set_title('Acceptance Rate')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Acceptance Rate')
        ax3.grid(True, alpha=0.3)
        
        # Cost vs Temperature
        ax4.scatter(temperatures, current_costs, c=iterations, cmap='viridis', alpha=0.6)
        ax4.set_title('Cost vs Temperature')
        ax4.set_xlabel('Temperature')
        ax4.set_ylabel('Current Cost')
        ax4.set_xscale('log')
        colorbar = plt.colorbar(ax4.scatter(temperatures, current_costs, c=iterations, cmap='viridis', alpha=0.6), ax=ax4)
        colorbar.set_label('Iteration')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class MultiStartSimulatedAnnealing(SimulatedAnnealingTSP):
    """Multi-start Simulated Annealing with multiple independent runs."""
    
    def solve_multistart(self, num_starts: int = 5, **kwargs) -> dict:
        """
        Solve using multiple independent SA runs.
        
        Args:
            num_starts: Number of independent runs
            **kwargs: Arguments passed to individual SA runs
        
        Returns:
            Dictionary containing the best solution and statistics
        """
        verbose = kwargs.get('verbose', True)
        kwargs['verbose'] = False  # Suppress individual run output
        
        best_overall_tour = None
        best_overall_cost = float('inf')
        all_results = []
        
        if verbose:
            print(f"Running {num_starts} independent SA starts...")
        
        start_time = time.time()
        
        for start_idx in range(num_starts):
            if verbose:
                print(f"Start {start_idx + 1}/{num_starts}...")
            
            # Reset for new start
            self.current_tour = None
            self.current_cost = float('inf')
            self.best_tour = None
            self.best_cost = float('inf')
            self.history = []
            
            # Run SA
            result = self.solve(**kwargs)
            all_results.append(result)
            
            # Update overall best
            if result['distance'] < best_overall_cost:
                best_overall_cost = result['distance']
                best_overall_tour = result['tour'].copy()
            
            if verbose:
                print(f"  Result: {result['distance']:.2f}")
        
        end_time = time.time()
        
        # Compile multi-start results
        distances = [r['distance'] for r in all_results]
        times = [r['time'] for r in all_results]
        
        multi_result = {
            'tour': best_overall_tour,
            'distance': best_overall_cost,
            'time': end_time - start_time,
            'algorithm': 'Multi-Start Simulated Annealing',
            'num_starts': num_starts,
            'all_distances': distances,
            'best_distance': min(distances),
            'worst_distance': max(distances),
            'avg_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'avg_time_per_start': np.mean(times),
            'individual_results': all_results
        }
        
        if verbose:
            print(f"\nMulti-start Results:")
            print(f"Best distance: {multi_result['best_distance']:.2f}")
            print(f"Worst distance: {multi_result['worst_distance']:.2f}")
            print(f"Average distance: {multi_result['avg_distance']:.2f} ± {multi_result['std_distance']:.2f}")
            print(f"Total time: {multi_result['time']:.3f} seconds")
        
        return multi_result


# Example usage and testing functions
def test_simulated_annealing():
    """Test the Simulated Annealing with a sample problem."""
    from .tsp_base import TSPGenerator
    
    # Generate a test problem
    cities = TSPGenerator.generate_random_cities(20, seed=42)
    tsp_instance = TSPInstance(cities)
    
    # Test different cooling schedules
    cooling_schedules = ['geometric', 'linear', 'exponential', 'adaptive']
    results = {}
    
    for schedule in cooling_schedules:
        print(f"\nTesting {schedule} cooling schedule...")
        sa = SimulatedAnnealingTSP(tsp_instance)
        result = sa.solve(max_iterations=5000, cooling_schedule=schedule, verbose=False)
        results[schedule] = result
        print(f"{schedule}: {result['distance']:.2f}")
    
    # Find best result
    best_schedule = min(results.keys(), key=lambda s: results[s]['distance'])
    best_result = results[best_schedule]
    
    print(f"\nBest cooling schedule: {best_schedule}")
    print(f"Best distance: {best_result['distance']:.2f}")
    
    # Visualize best result
    sa = SimulatedAnnealingTSP(tsp_instance)
    sa.history = best_result['history']
    tsp_instance.visualize_tour(best_result['tour'], 
                               title=f"SA Solution ({best_schedule}) - Distance: {best_result['distance']:.2f}")
    sa.get_convergence_plot()
    
    return best_result


def test_multistart_simulated_annealing():
    """Test the Multi-start Simulated Annealing."""
    from .tsp_base import TSPGenerator
    
    # Generate a test problem
    cities = TSPGenerator.generate_random_cities(20, seed=42)
    tsp_instance = TSPInstance(cities)
    
    # Create and run Multi-start SA
    mssa = MultiStartSimulatedAnnealing(tsp_instance)
    result = mssa.solve_multistart(num_starts=5, max_iterations=3000, 
                                  cooling_schedule='geometric', verbose=True)
    
    # Visualize results
    tsp_instance.visualize_tour(result['tour'], 
                               title=f"Multi-start SA Solution - Distance: {result['distance']:.2f}")
    
    return result


if __name__ == "__main__":
    print("Testing Simulated Annealing...")
    test_simulated_annealing()
    
    print("\n" + "="*60)
    print("Testing Multi-start Simulated Annealing...")
    test_multistart_simulated_annealing()