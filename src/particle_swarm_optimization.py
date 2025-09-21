"""
Particle Swarm Optimization for Traveling Salesman Problem

This module implements PSO adapted for the discrete TSP problem using 
position and velocity representations suitable for permutation problems.
"""

import random
import numpy as np
from typing import List, Tuple, Optional
from .tsp_base import TSPInstance
import time


class ParticleSwarmOptimizationTSP:
    """Particle Swarm Optimization implementation for solving TSP."""
    
    def __init__(self, tsp_instance: TSPInstance, num_particles: int = 30,
                 w: float = 0.9, w_damp: float = 0.99, c1: float = 2.0, c2: float = 2.0):
        """
        Initialize the Particle Swarm Optimization algorithm.
        
        Args:
            tsp_instance: The TSP problem instance
            num_particles: Number of particles in the swarm
            w: Inertia weight
            w_damp: Inertia weight damping ratio
            c1: Personal learning factor
            c2: Global learning factor
        """
        self.tsp_instance = tsp_instance
        self.num_cities = tsp_instance.num_cities
        self.num_particles = num_particles
        
        # PSO parameters
        self.w = w          # Inertia weight
        self.w_damp = w_damp  # Inertia weight damping
        self.c1 = c1        # Personal learning factor
        self.c2 = c2        # Global learning factor
        
        # Initialize particles
        self.particles = []
        self.personal_best_positions = []
        self.personal_best_costs = []
        self.global_best_position = None
        self.global_best_cost = float('inf')
        
        # History tracking
        self.history = []
        
        self._initialize_swarm()
    
    def _initialize_swarm(self) -> None:
        """Initialize the particle swarm."""
        for i in range(self.num_particles):
            # Initialize position as a random tour
            if i == 0:
                # First particle uses nearest neighbor heuristic
                position = self.tsp_instance.get_nearest_neighbor_tour()
            else:
                position = self.tsp_instance.generate_random_tour()
            
            # Initialize velocity as random swap operations
            velocity = self._generate_random_velocity()
            
            # Calculate cost
            cost = self.tsp_instance.calculate_tour_distance(position)
            
            # Create particle
            particle = {
                'position': position,
                'velocity': velocity,
                'cost': cost
            }
            
            self.particles.append(particle)
            
            # Initialize personal best
            self.personal_best_positions.append(position.copy())
            self.personal_best_costs.append(cost)
            
            # Update global best
            if cost < self.global_best_cost:
                self.global_best_cost = cost
                self.global_best_position = position.copy()
    
    def _generate_random_velocity(self) -> List[Tuple[int, int]]:
        """Generate a random velocity as a list of swap operations."""
        num_swaps = random.randint(1, self.num_cities // 2)
        velocity = []
        
        for _ in range(num_swaps):
            i, j = random.sample(range(self.num_cities), 2)
            velocity.append((i, j))
        
        return velocity
    
    def _apply_velocity(self, position: List[int], velocity: List[Tuple[int, int]]) -> List[int]:
        """Apply velocity (swap operations) to a position."""
        new_position = position.copy()
        
        for swap in velocity:
            i, j = swap
            if 0 <= i < len(new_position) and 0 <= j < len(new_position):
                new_position[i], new_position[j] = new_position[j], new_position[i]
        
        return new_position
    
    def _calculate_swap_sequence(self, source: List[int], target: List[int]) -> List[Tuple[int, int]]:
        """Calculate the sequence of swaps needed to transform source into target."""
        if len(source) != len(target):
            return []
        
        swaps = []
        current = source.copy()
        
        for i in range(len(target)):
            if current[i] != target[i]:
                # Find where target[i] is in current
                j = current.index(target[i])
                if j != i:
                    # Swap to put target[i] in position i
                    current[i], current[j] = current[j], current[i]
                    swaps.append((i, j))
        
        return swaps
    
    def _combine_velocities(self, v1: List[Tuple[int, int]], v2: List[Tuple[int, int]], 
                           weight: float) -> List[Tuple[int, int]]:
        """Combine two velocities with a given weight."""
        combined = []
        
        # Add v1 with probability based on weight
        for swap in v1:
            if random.random() < weight:
                combined.append(swap)
        
        # Add v2 with remaining probability
        for swap in v2:
            if random.random() < (1 - weight):
                combined.append(swap)
        
        return combined
    
    def _limit_velocity(self, velocity: List[Tuple[int, int]], max_length: int) -> List[Tuple[int, int]]:
        """Limit the velocity to a maximum number of swaps."""
        if len(velocity) <= max_length:
            return velocity
        
        # Randomly sample swaps to keep
        return random.sample(velocity, max_length)
    
    def _update_particle_velocity(self, particle_idx: int) -> None:
        """Update the velocity of a particle."""
        particle = self.particles[particle_idx]
        current_position = particle['position']
        current_velocity = particle['velocity']
        
        # Calculate personal best influence
        personal_best_swaps = self._calculate_swap_sequence(
            current_position, self.personal_best_positions[particle_idx]
        )
        
        # Calculate global best influence
        global_best_swaps = self._calculate_swap_sequence(
            current_position, self.global_best_position
        )
        
        # Update velocity using PSO formula (adapted for discrete space)
        # v = w*v + c1*r1*(pbest - position) + c2*r2*(gbest - position)
        
        # Inertia component
        inertia_velocity = []
        for swap in current_velocity:
            if random.random() < self.w:
                inertia_velocity.append(swap)
        
        # Personal best component
        personal_velocity = []
        for swap in personal_best_swaps:
            if random.random() < self.c1 * random.random():
                personal_velocity.append(swap)
        
        # Global best component
        global_velocity = []
        for swap in global_best_swaps:
            if random.random() < self.c2 * random.random():
                global_velocity.append(swap)
        
        # Combine all components
        new_velocity = inertia_velocity + personal_velocity + global_velocity
        
        # Remove duplicate swaps and limit velocity
        unique_swaps = list(set(new_velocity))
        max_velocity_length = max(3, self.num_cities // 4)
        new_velocity = self._limit_velocity(unique_swaps, max_velocity_length)
        
        particle['velocity'] = new_velocity
    
    def _update_particle_position(self, particle_idx: int) -> None:
        """Update the position of a particle."""
        particle = self.particles[particle_idx]
        
        # Apply velocity to current position
        new_position = self._apply_velocity(particle['position'], particle['velocity'])
        
        # Calculate new cost
        new_cost = self.tsp_instance.calculate_tour_distance(new_position)
        
        # Update particle
        particle['position'] = new_position
        particle['cost'] = new_cost
        
        # Update personal best
        if new_cost < self.personal_best_costs[particle_idx]:
            self.personal_best_costs[particle_idx] = new_cost
            self.personal_best_positions[particle_idx] = new_position.copy()
        
        # Update global best
        if new_cost < self.global_best_cost:
            self.global_best_cost = new_cost
            self.global_best_position = new_position.copy()
    
    def _apply_local_search(self, position: List[int]) -> List[int]:
        """Apply 2-opt local search to improve a position."""
        improved_position = position.copy()
        improved = True
        
        while improved:
            improved = False
            current_cost = self.tsp_instance.calculate_tour_distance(improved_position)
            
            for i in range(len(position) - 1):
                for j in range(i + 2, len(position)):
                    if j == len(position) - 1 and i == 0:
                        continue
                    
                    # Create new tour by reversing segment
                    new_position = improved_position.copy()
                    new_position[i+1:j+1] = new_position[i+1:j+1][::-1]
                    
                    new_cost = self.tsp_instance.calculate_tour_distance(new_position)
                    if new_cost < current_cost:
                        improved_position = new_position
                        improved = True
                        break
                
                if improved:
                    break
        
        return improved_position
    
    def _diversify_swarm(self) -> None:
        """Diversify the swarm to avoid premature convergence."""
        # Keep the best few particles, randomize others
        num_to_keep = max(2, self.num_particles // 4)
        
        # Sort particles by cost
        particle_indices = sorted(range(self.num_particles), 
                                key=lambda i: self.particles[i]['cost'])
        
        # Randomize worst particles
        for i in range(num_to_keep, self.num_particles):
            idx = particle_indices[i]
            
            # Generate new random position
            new_position = self.tsp_instance.generate_random_tour()
            new_cost = self.tsp_instance.calculate_tour_distance(new_position)
            
            # Update particle
            self.particles[idx]['position'] = new_position
            self.particles[idx]['cost'] = new_cost
            self.particles[idx]['velocity'] = self._generate_random_velocity()
    
    def solve(self, max_iterations: int = 1000, convergence_threshold: int = 50,
              apply_local_search: bool = True, diversify_interval: int = 100,
              verbose: bool = True) -> dict:
        """
        Solve the TSP using Particle Swarm Optimization.
        
        Args:
            max_iterations: Maximum number of iterations
            convergence_threshold: Stop if no improvement for this many iterations
            apply_local_search: Whether to apply local search
            diversify_interval: Interval for swarm diversification
            verbose: Whether to print progress information
        
        Returns:
            Dictionary containing the solution and statistics
        """
        start_time = time.time()
        
        no_improvement_count = 0
        last_best_cost = self.global_best_cost
        
        if verbose:
            print(f"Starting PSO with {self.num_particles} particles...")
            print(f"Initial best cost: {self.global_best_cost:.2f}")
        
        for iteration in range(max_iterations):
            # Update all particles
            for i in range(self.num_particles):
                self._update_particle_velocity(i)
                self._update_particle_position(i)
            
            # Apply local search to best particles
            if apply_local_search and iteration % 10 == 0:
                # Apply local search to top particles
                particle_indices = sorted(range(self.num_particles), 
                                        key=lambda i: self.particles[i]['cost'])
                
                for i in range(min(3, self.num_particles)):
                    idx = particle_indices[i]
                    improved_position = self._apply_local_search(self.particles[idx]['position'])
                    improved_cost = self.tsp_instance.calculate_tour_distance(improved_position)
                    
                    if improved_cost < self.particles[idx]['cost']:
                        self.particles[idx]['position'] = improved_position
                        self.particles[idx]['cost'] = improved_cost
                        
                        # Update personal best
                        if improved_cost < self.personal_best_costs[idx]:
                            self.personal_best_costs[idx] = improved_cost
                            self.personal_best_positions[idx] = improved_position.copy()
                        
                        # Update global best
                        if improved_cost < self.global_best_cost:
                            self.global_best_cost = improved_cost
                            self.global_best_position = improved_position.copy()
            
            # Diversify swarm if needed
            if diversify_interval > 0 and iteration % diversify_interval == 0 and iteration > 0:
                if no_improvement_count > diversify_interval // 2:
                    self._diversify_swarm()
                    if verbose:
                        print(f"Diversified swarm at iteration {iteration}")
            
            # Update inertia weight
            self.w *= self.w_damp
            
            # Track progress
            current_costs = [p['cost'] for p in self.particles]
            avg_cost = np.mean(current_costs)
            std_cost = np.std(current_costs)
            
            self.history.append({
                'iteration': iteration,
                'global_best_cost': self.global_best_cost,
                'avg_cost': avg_cost,
                'std_cost': std_cost,
                'inertia_weight': self.w
            })
            
            # Check for improvement
            if self.global_best_cost < last_best_cost:
                no_improvement_count = 0
                last_best_cost = self.global_best_cost
            else:
                no_improvement_count += 1
            
            # Print progress
            if verbose and iteration % 50 == 0:
                print(f"Iteration {iteration}: Best = {self.global_best_cost:.2f}, "
                      f"Avg = {avg_cost:.2f}, w = {self.w:.3f}")
            
            # Check convergence
            if no_improvement_count >= convergence_threshold:
                if verbose:
                    print(f"Converged after {iteration} iterations")
                break
        
        end_time = time.time()
        
        result = {
            'tour': self.global_best_position,
            'distance': self.global_best_cost,
            'iterations': iteration + 1,
            'time': end_time - start_time,
            'history': self.history,
            'algorithm': 'Particle Swarm Optimization'
        }
        
        if verbose:
            print(f"Final best distance: {self.global_best_cost:.2f}")
            print(f"Total runtime: {result['time']:.3f} seconds")
        
        return result
    
    def get_convergence_plot(self) -> None:
        """Plot the convergence history."""
        import matplotlib.pyplot as plt
        
        iterations = [entry['iteration'] for entry in self.history]
        best_costs = [entry['global_best_cost'] for entry in self.history]
        avg_costs = [entry['avg_cost'] for entry in self.history]
        inertia_weights = [entry['inertia_weight'] for entry in self.history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Convergence plot
        ax1.plot(iterations, best_costs, 'b-', linewidth=2, label='Best Cost')
        ax1.plot(iterations, avg_costs, 'r--', alpha=0.7, label='Average Cost')
        ax1.set_title('PSO Convergence')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Cost')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Inertia weight plot
        ax2.plot(iterations, inertia_weights, 'g-', linewidth=2)
        ax2.set_title('Inertia Weight')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Weight')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_swarm_diversity_plot(self) -> None:
        """Plot the swarm diversity over iterations."""
        import matplotlib.pyplot as plt
        
        iterations = [entry['iteration'] for entry in self.history]
        std_costs = [entry['std_cost'] for entry in self.history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, std_costs, 'purple', linewidth=2)
        plt.title('Swarm Diversity (Standard Deviation of Costs)')
        plt.xlabel('Iteration')
        plt.ylabel('Standard Deviation')
        plt.grid(True, alpha=0.3)
        plt.show()


# Example usage and testing functions
def test_particle_swarm_optimization():
    """Test the Particle Swarm Optimization with a sample problem."""
    from .tsp_base import TSPGenerator
    
    # Generate a test problem
    cities = TSPGenerator.generate_random_cities(20, seed=42)
    tsp_instance = TSPInstance(cities)
    
    # Create and run PSO
    pso = ParticleSwarmOptimizationTSP(tsp_instance, num_particles=20)
    result = pso.solve(max_iterations=200, verbose=True)
    
    # Visualize results
    tsp_instance.visualize_tour(result['tour'], 
                               title=f"PSO Solution (Distance: {result['distance']:.2f})")
    pso.get_convergence_plot()
    pso.get_swarm_diversity_plot()
    
    return result


if __name__ == "__main__":
    test_particle_swarm_optimization()