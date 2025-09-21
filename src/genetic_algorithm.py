"""
Genetic Algorithm for Traveling Salesman Problem

This module implements a genetic algorithm specifically designed for solving
the TSP using order-based crossover and mutation operators.
"""

import random
import numpy as np
from typing import List, Tuple, Optional
from .tsp_base import TSPInstance, evaluate_solution
import time


class GeneticAlgorithmTSP:
    """Genetic Algorithm implementation for solving TSP."""
    
    def __init__(self, tsp_instance: TSPInstance, population_size: int = 100,
                 elite_size: int = 20, mutation_rate: float = 0.01,
                 crossover_rate: float = 0.8):
        """
        Initialize the Genetic Algorithm.
        
        Args:
            tsp_instance: The TSP problem instance
            population_size: Size of the population
            elite_size: Number of elite individuals to keep
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
        """
        self.tsp_instance = tsp_instance
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        self.population = []
        self.fitness_scores = []
        self.best_tour = None
        self.best_distance = float('inf')
        self.history = []
    
    def initialize_population(self) -> None:
        """Initialize the population with random tours and some heuristic solutions."""
        self.population = []
        
        # Add some random tours
        for _ in range(self.population_size - 2):
            tour = self.tsp_instance.generate_random_tour()
            self.population.append(tour)
        
        # Add nearest neighbor solutions starting from different cities
        for start_city in [0, self.tsp_instance.num_cities // 2]:
            if start_city < self.tsp_instance.num_cities:
                nn_tour = self.tsp_instance.get_nearest_neighbor_tour(start_city)
                self.population.append(nn_tour)
    
    def calculate_fitness(self, tour: List[int]) -> float:
        """Calculate fitness score for a tour (inverse of distance)."""
        distance = self.tsp_instance.calculate_tour_distance(tour)
        return 1.0 / (1.0 + distance)  # Higher fitness for shorter tours
    
    def evaluate_population(self) -> None:
        """Evaluate fitness for all individuals in the population."""
        self.fitness_scores = []
        for tour in self.population:
            fitness = self.calculate_fitness(tour)
            self.fitness_scores.append(fitness)
            
            # Update best solution
            distance = self.tsp_instance.calculate_tour_distance(tour)
            if distance < self.best_distance:
                self.best_distance = distance
                self.best_tour = tour.copy()
    
    def tournament_selection(self, tournament_size: int = 3) -> List[int]:
        """Select an individual using tournament selection."""
        tournament_indices = random.sample(range(self.population_size), tournament_size)
        tournament_fitness = [self.fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
        return self.population[winner_idx].copy()
    
    def roulette_wheel_selection(self) -> List[int]:
        """Select an individual using roulette wheel selection."""
        total_fitness = sum(self.fitness_scores)
        if total_fitness == 0:
            return random.choice(self.population).copy()
        
        spin = random.uniform(0, total_fitness)
        current_sum = 0
        
        for i, fitness in enumerate(self.fitness_scores):
            current_sum += fitness
            if current_sum >= spin:
                return self.population[i].copy()
        
        return self.population[-1].copy()
    
    def order_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        Perform Order Crossover (OX) - preserves the relative order of cities.
        """
        size = len(parent1)
        
        # Choose two random cut points
        start, end = sorted(random.sample(range(size), 2))
        
        # Create offspring
        offspring1 = [-1] * size
        offspring2 = [-1] * size
        
        # Copy the substring from parent1 to offspring1 and parent2 to offspring2
        offspring1[start:end] = parent1[start:end]
        offspring2[start:end] = parent2[start:end]
        
        # Fill remaining positions
        self._fill_offspring_ox(offspring1, parent2, start, end)
        self._fill_offspring_ox(offspring2, parent1, start, end)
        
        return offspring1, offspring2
    
    def _fill_offspring_ox(self, offspring: List[int], parent: List[int], 
                          start: int, end: int) -> None:
        """Helper method for order crossover."""
        size = len(offspring)
        parent_idx = end
        offspring_idx = end
        
        while -1 in offspring:
            if parent[parent_idx % size] not in offspring:
                offspring[offspring_idx % size] = parent[parent_idx % size]
                offspring_idx += 1
            parent_idx += 1
    
    def pmx_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        Perform Partially Mapped Crossover (PMX).
        """
        size = len(parent1)
        
        # Choose two random cut points
        start, end = sorted(random.sample(range(size), 2))
        
        # Create offspring as copies of parents
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        
        # Create mapping
        mapping1 = {}
        mapping2 = {}
        
        for i in range(start, end):
            # Swap elements and create mapping
            city1, city2 = parent1[i], parent2[i]
            offspring1[i], offspring2[i] = city2, city1
            mapping1[city2] = city1
            mapping2[city1] = city2
        
        # Resolve conflicts
        self._resolve_pmx_conflicts(offspring1, mapping1, start, end)
        self._resolve_pmx_conflicts(offspring2, mapping2, start, end)
        
        return offspring1, offspring2
    
    def _resolve_pmx_conflicts(self, offspring: List[int], mapping: dict, 
                              start: int, end: int) -> None:
        """Helper method to resolve conflicts in PMX crossover."""
        for i in range(len(offspring)):
            if i < start or i >= end:
                while offspring[i] in offspring[start:end]:
                    offspring[i] = mapping[offspring[i]]
    
    def swap_mutation(self, tour: List[int]) -> List[int]:
        """Perform swap mutation - swap two random cities."""
        mutated_tour = tour.copy()
        if len(mutated_tour) > 1:
            idx1, idx2 = random.sample(range(len(mutated_tour)), 2)
            mutated_tour[idx1], mutated_tour[idx2] = mutated_tour[idx2], mutated_tour[idx1]
        return mutated_tour
    
    def inversion_mutation(self, tour: List[int]) -> List[int]:
        """Perform inversion mutation - reverse a segment of the tour."""
        mutated_tour = tour.copy()
        if len(mutated_tour) > 2:
            start, end = sorted(random.sample(range(len(mutated_tour)), 2))
            mutated_tour[start:end+1] = mutated_tour[start:end+1][::-1]
        return mutated_tour
    
    def scramble_mutation(self, tour: List[int]) -> List[int]:
        """Perform scramble mutation - randomly shuffle a segment."""
        mutated_tour = tour.copy()
        if len(mutated_tour) > 2:
            start, end = sorted(random.sample(range(len(mutated_tour)), 2))
            segment = mutated_tour[start:end+1]
            random.shuffle(segment)
            mutated_tour[start:end+1] = segment
        return mutated_tour
    
    def local_search_2opt(self, tour: List[int], max_improvements: int = 10) -> List[int]:
        """Apply 2-opt local search to improve a tour."""
        improved_tour = tour.copy()
        current_distance = self.tsp_instance.calculate_tour_distance(improved_tour)
        improvements = 0
        
        for i in range(len(tour) - 1):
            for j in range(i + 2, len(tour)):
                if j == len(tour) - 1 and i == 0:
                    continue  # Skip if it would reverse the entire tour
                
                # Create new tour by reversing the segment between i and j
                new_tour = improved_tour.copy()
                new_tour[i+1:j+1] = new_tour[i+1:j+1][::-1]
                
                new_distance = self.tsp_instance.calculate_tour_distance(new_tour)
                if new_distance < current_distance:
                    improved_tour = new_tour
                    current_distance = new_distance
                    improvements += 1
                    
                    if improvements >= max_improvements:
                        return improved_tour
        
        return improved_tour
    
    def evolve_generation(self) -> None:
        """Evolve one generation of the population."""
        new_population = []
        
        # Elite selection - keep best individuals
        elite_indices = sorted(range(self.population_size), 
                             key=lambda i: self.fitness_scores[i], reverse=True)[:self.elite_size]
        for idx in elite_indices:
            new_population.append(self.population[idx].copy())
        
        # Generate offspring to fill the rest of the population
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            # Crossover
            if random.random() < self.crossover_rate:
                offspring1, offspring2 = self.order_crossover(parent1, parent2)
            else:
                offspring1, offspring2 = parent1.copy(), parent2.copy()
            
            # Mutation
            if random.random() < self.mutation_rate:
                offspring1 = self.swap_mutation(offspring1)
            if random.random() < self.mutation_rate:
                offspring2 = self.inversion_mutation(offspring2)
            
            # Add offspring to new population
            new_population.extend([offspring1, offspring2])
        
        # Trim to exact population size
        self.population = new_population[:self.population_size]
    
    def solve(self, max_generations: int = 1000, convergence_threshold: int = 100,
              apply_local_search: bool = True, verbose: bool = True) -> dict:
        """
        Solve the TSP using Genetic Algorithm.
        
        Args:
            max_generations: Maximum number of generations
            convergence_threshold: Stop if no improvement for this many generations
            apply_local_search: Whether to apply local search to elite solutions
            verbose: Whether to print progress information
        
        Returns:
            Dictionary containing the solution and statistics
        """
        start_time = time.time()
        
        # Initialize
        self.initialize_population()
        self.evaluate_population()
        
        no_improvement_count = 0
        last_best_distance = self.best_distance
        
        if verbose:
            print(f"Initial best distance: {self.best_distance:.2f}")
        
        # Evolution loop
        for generation in range(max_generations):
            # Evolve population
            self.evolve_generation()
            self.evaluate_population()
            
            # Apply local search to elite solutions
            if apply_local_search and generation % 10 == 0:
                for i in range(min(5, self.elite_size)):
                    elite_idx = np.argmax(self.fitness_scores)
                    improved_tour = self.local_search_2opt(self.population[elite_idx])
                    self.population[elite_idx] = improved_tour
                
                self.evaluate_population()
            
            # Track progress
            self.history.append({
                'generation': generation,
                'best_distance': self.best_distance,
                'avg_fitness': np.mean(self.fitness_scores),
                'diversity': self._calculate_diversity()
            })
            
            # Check for improvement
            if self.best_distance < last_best_distance:
                no_improvement_count = 0
                last_best_distance = self.best_distance
            else:
                no_improvement_count += 1
            
            # Print progress
            if verbose and generation % 50 == 0:
                print(f"Generation {generation}: Best = {self.best_distance:.2f}, "
                      f"Avg Fitness = {np.mean(self.fitness_scores):.6f}")
            
            # Check convergence
            if no_improvement_count >= convergence_threshold:
                if verbose:
                    print(f"Converged after {generation} generations")
                break
        
        end_time = time.time()
        
        result = {
            'tour': self.best_tour,
            'distance': self.best_distance,
            'generations': generation + 1,
            'time': end_time - start_time,
            'history': self.history,
            'algorithm': 'Genetic Algorithm'
        }
        
        if verbose:
            print(f"Final best distance: {self.best_distance:.2f}")
            print(f"Total runtime: {result['time']:.3f} seconds")
        
        return result
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity based on unique tours."""
        unique_tours = set()
        for tour in self.population:
            # Use a normalized representation for comparison
            min_idx = tour.index(min(tour))
            normalized_tour = tuple(tour[min_idx:] + tour[:min_idx])
            unique_tours.add(normalized_tour)
        
        return len(unique_tours) / self.population_size
    
    def get_convergence_plot(self) -> None:
        """Plot the convergence history."""
        import matplotlib.pyplot as plt
        
        generations = [entry['generation'] for entry in self.history]
        best_distances = [entry['best_distance'] for entry in self.history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(generations, best_distances, 'b-', linewidth=2)
        plt.title('Genetic Algorithm Convergence')
        plt.xlabel('Generation')
        plt.ylabel('Best Distance')
        plt.grid(True, alpha=0.3)
        plt.show()


# Example usage and testing functions
def test_genetic_algorithm():
    """Test the Genetic Algorithm with a sample problem."""
    from .tsp_base import TSPGenerator
    
    # Generate a test problem
    cities = TSPGenerator.generate_random_cities(20, seed=42)
    tsp_instance = TSPInstance(cities)
    
    # Create and run GA
    ga = GeneticAlgorithmTSP(tsp_instance, population_size=50, elite_size=10)
    result = ga.solve(max_generations=200, verbose=True)
    
    # Visualize results
    tsp_instance.visualize_tour(result['tour'], 
                               title=f"GA Solution (Distance: {result['distance']:.2f})")
    ga.get_convergence_plot()
    
    return result


if __name__ == "__main__":
    test_genetic_algorithm()