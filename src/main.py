"""
Main Application for Nature-Inspired TSP Solvers

This module provides a comprehensive interface for running and comparing
different nature-inspired algorithms on TSP instances.
"""

import sys
import os
import argparse
import json
import time
from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib.pyplot as plt

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tsp_base import TSPInstance, TSPGenerator, compare_algorithms_results
from genetic_algorithm import GeneticAlgorithmTSP
from ant_colony_optimization import AntColonyOptimizationTSP, MaxMinAntSystem
from particle_swarm_optimization import ParticleSwarmOptimizationTSP
from simulated_annealing import SimulatedAnnealingTSP, MultiStartSimulatedAnnealing


class TSPBenchmark:
    """Comprehensive benchmarking suite for TSP algorithms."""
    
    def __init__(self, tsp_instance: TSPInstance):
        self.tsp_instance = tsp_instance
        self.results = {}
        
    def run_genetic_algorithm(self, **kwargs) -> Dict[str, Any]:
        """Run Genetic Algorithm with specified parameters."""
        params = {
            'population_size': 100,
            'elite_size': 20,
            'mutation_rate': 0.01,
            'crossover_rate': 0.8,
            'max_generations': 500,
            'convergence_threshold': 50,
            'apply_local_search': True,
            'verbose': False
        }
        params.update(kwargs)
        
        print(f"Running Genetic Algorithm...")
        ga = GeneticAlgorithmTSP(
            self.tsp_instance,
            population_size=params['population_size'],
            elite_size=params['elite_size'],
            mutation_rate=params['mutation_rate'],
            crossover_rate=params['crossover_rate']
        )
        
        result = ga.solve(
            max_generations=params['max_generations'],
            convergence_threshold=params['convergence_threshold'],
            apply_local_search=params['apply_local_search'],
            verbose=params['verbose']
        )
        
        return result
    
    def run_ant_colony_optimization(self, variant='standard', **kwargs) -> Dict[str, Any]:
        """Run Ant Colony Optimization with specified parameters."""
        params = {
            'num_ants': None,  # Will default to number of cities
            'alpha': 1.0,
            'beta': 5.0,
            'rho': 0.1,
            'q0': 0.9,
            'xi': 0.1,
            'max_iterations': 500,
            'convergence_threshold': 50,
            'apply_local_search': True,
            'verbose': False
        }
        params.update(kwargs)
        
        if variant == 'mmas':
            print(f"Running Max-Min Ant System...")
            aco = MaxMinAntSystem(
                self.tsp_instance,
                num_ants=params['num_ants'],
                alpha=params['alpha'],
                beta=params['beta'],
                rho=params['rho']
            )
        else:
            print(f"Running Ant Colony Optimization...")
            aco = AntColonyOptimizationTSP(
                self.tsp_instance,
                num_ants=params['num_ants'],
                alpha=params['alpha'],
                beta=params['beta'],
                rho=params['rho'],
                q0=params['q0'],
                xi=params['xi']
            )
        
        result = aco.solve(
            max_iterations=params['max_iterations'],
            convergence_threshold=params['convergence_threshold'],
            apply_local_search=params['apply_local_search'],
            verbose=params['verbose']
        )
        
        return result
    
    def run_particle_swarm_optimization(self, **kwargs) -> Dict[str, Any]:
        """Run Particle Swarm Optimization with specified parameters."""
        params = {
            'num_particles': 30,
            'w': 0.9,
            'w_damp': 0.99,
            'c1': 2.0,
            'c2': 2.0,
            'max_iterations': 500,
            'convergence_threshold': 50,
            'apply_local_search': True,
            'diversify_interval': 100,
            'verbose': False
        }
        params.update(kwargs)
        
        print(f"Running Particle Swarm Optimization...")
        pso = ParticleSwarmOptimizationTSP(
            self.tsp_instance,
            num_particles=params['num_particles'],
            w=params['w'],
            w_damp=params['w_damp'],
            c1=params['c1'],
            c2=params['c2']
        )
        
        result = pso.solve(
            max_iterations=params['max_iterations'],
            convergence_threshold=params['convergence_threshold'],
            apply_local_search=params['apply_local_search'],
            diversify_interval=params['diversify_interval'],
            verbose=params['verbose']
        )
        
        return result
    
    def run_simulated_annealing(self, multistart=False, **kwargs) -> Dict[str, Any]:
        """Run Simulated Annealing with specified parameters."""
        params = {
            'initial_temperature': None,
            'final_temperature': 0.01,
            'cooling_rate': 0.95,
            'max_iterations': 5000,
            'max_iterations_per_temp': 100,
            'cooling_schedule': 'geometric',
            'reheat_interval': 0,
            'verbose': False
        }
        params.update(kwargs)
        
        if multistart:
            multistart_params = {
                'num_starts': kwargs.get('num_starts', 3)
            }
            print(f"Running Multi-start Simulated Annealing...")
            sa = MultiStartSimulatedAnnealing(
                self.tsp_instance,
                initial_temperature=params['initial_temperature'],
                final_temperature=params['final_temperature'],
                cooling_rate=params['cooling_rate']
            )
            
            result = sa.solve_multistart(
                num_starts=multistart_params['num_starts'],
                max_iterations=params['max_iterations'],
                max_iterations_per_temp=params['max_iterations_per_temp'],
                cooling_schedule=params['cooling_schedule'],
                reheat_interval=params['reheat_interval'],
                verbose=params['verbose']
            )
        else:
            print(f"Running Simulated Annealing...")
            sa = SimulatedAnnealingTSP(
                self.tsp_instance,
                initial_temperature=params['initial_temperature'],
                final_temperature=params['final_temperature'],
                cooling_rate=params['cooling_rate']
            )
            
            result = sa.solve(
                max_iterations=params['max_iterations'],
                max_iterations_per_temp=params['max_iterations_per_temp'],
                cooling_schedule=params['cooling_schedule'],
                reheat_interval=params['reheat_interval'],
                verbose=params['verbose']
            )
        
        return result
    
    def run_all_algorithms(self, save_results: bool = True) -> Dict[str, Dict[str, Any]]:
        """Run all algorithms with default parameters."""
        algorithms = {
            'Genetic Algorithm': self.run_genetic_algorithm,
            'Ant Colony Optimization': self.run_ant_colony_optimization,
            'Max-Min Ant System': lambda: self.run_ant_colony_optimization(variant='mmas'),
            'Particle Swarm Optimization': self.run_particle_swarm_optimization,
            'Simulated Annealing': self.run_simulated_annealing,
            'Multi-start SA': lambda: self.run_simulated_annealing(multistart=True)
        }
        
        results = {}
        total_start_time = time.time()
        
        print(f"Running comprehensive benchmark on {self.tsp_instance.num_cities} cities...")
        print("=" * 60)
        
        for name, algorithm_func in algorithms.items():
            start_time = time.time()
            try:
                result = algorithm_func()
                end_time = time.time()
                
                print(f"{name}: {result['distance']:.2f} (Time: {end_time - start_time:.2f}s)")
                results[name] = result
                
            except Exception as e:
                print(f"{name}: FAILED - {str(e)}")
                results[name] = {'error': str(e), 'distance': float('inf')}
        
        total_end_time = time.time()
        print("=" * 60)
        print(f"Total benchmark time: {total_end_time - total_start_time:.2f} seconds")
        
        self.results = results
        
        if save_results:
            self.save_results()
        
        return results
    
    def save_results(self, filename: Optional[str] = None) -> None:
        """Save results to a JSON file."""
        if filename is None:
            filename = f"tsp_results_{self.tsp_instance.num_cities}cities_{int(time.time())}.json"
        
        # Prepare results for JSON serialization
        json_results = {}
        for alg_name, result in self.results.items():
            if 'error' in result:
                json_results[alg_name] = result
            else:
                json_results[alg_name] = {
                    'distance': result['distance'],
                    'time': result['time'],
                    'algorithm': result.get('algorithm', alg_name),
                    'tour': result['tour']
                }
                
                # Add algorithm-specific metrics
                if 'iterations' in result:
                    json_results[alg_name]['iterations'] = result['iterations']
                if 'generations' in result:
                    json_results[alg_name]['generations'] = result['generations']
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Results saved to {filename}")
    
    def create_comparison_plots(self, save_plots: bool = True) -> None:
        """Create comparison plots for all algorithms."""
        if not self.results:
            print("No results to plot. Run algorithms first.")
            return
        
        # Filter out failed results
        valid_results = {name: result for name, result in self.results.items() 
                        if 'error' not in result}
        
        if not valid_results:
            print("No valid results to plot.")
            return
        
        # Create comparison plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Distance comparison
        algorithms = list(valid_results.keys())
        distances = [valid_results[alg]['distance'] for alg in algorithms]
        
        bars1 = ax1.bar(range(len(algorithms)), distances, color='lightblue', edgecolor='navy')
        ax1.set_title('Distance Comparison')
        ax1.set_xlabel('Algorithm')
        ax1.set_ylabel('Total Distance')
        ax1.set_xticks(range(len(algorithms)))
        ax1.set_xticklabels(algorithms, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{distances[i]:.1f}', ha='center', va='bottom')
        
        # 2. Runtime comparison
        times = [valid_results[alg]['time'] for alg in algorithms]
        
        bars2 = ax2.bar(range(len(algorithms)), times, color='lightgreen', edgecolor='darkgreen')
        ax2.set_title('Runtime Comparison')
        ax2.set_xlabel('Algorithm')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_xticks(range(len(algorithms)))
        ax2.set_xticklabels(algorithms, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{times[i]:.2f}s', ha='center', va='bottom')
        
        # 3. Distance vs Runtime scatter plot
        ax3.scatter(times, distances, c=range(len(algorithms)), cmap='viridis', s=100)
        for i, alg in enumerate(algorithms):
            ax3.annotate(alg, (times[i], distances[i]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        ax3.set_title('Distance vs Runtime')
        ax3.set_xlabel('Runtime (seconds)')
        ax3.set_ylabel('Distance')
        ax3.grid(True, alpha=0.3)
        
        # 4. Relative performance (normalized)
        min_distance = min(distances)
        min_time = min(times)
        
        normalized_distances = [(d - min_distance) / min_distance * 100 for d in distances]
        normalized_times = [(t - min_time) / min_time * 100 for t in times]
        
        x = np.arange(len(algorithms))
        width = 0.35
        
        bars3 = ax4.bar(x - width/2, normalized_distances, width, label='Distance Gap (%)', 
                       color='lightcoral', edgecolor='darkred')
        bars4 = ax4.bar(x + width/2, normalized_times, width, label='Time Overhead (%)', 
                       color='lightsalmon', edgecolor='darkorange')
        
        ax4.set_title('Relative Performance (% over best)')
        ax4.set_xlabel('Algorithm')
        ax4.set_ylabel('Percentage over best')
        ax4.set_xticks(x)
        ax4.set_xticklabels(algorithms, rotation=45, ha='right')
        ax4.legend()
        
        plt.tight_layout()
        
        if save_plots:
            filename = f"tsp_comparison_{self.tsp_instance.num_cities}cities.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Comparison plots saved to {filename}")
        
        plt.show()
    
    def print_detailed_comparison(self) -> None:
        """Print detailed comparison of all algorithms."""
        if not self.results:
            print("No results to compare. Run algorithms first.")
            return
        
        # Filter valid results
        valid_results = {name: result for name, result in self.results.items() 
                        if 'error' not in result}
        
        if not valid_results:
            print("No valid results to compare.")
            return
        
        print("\n" + "="*80)
        print("DETAILED ALGORITHM COMPARISON")
        print("="*80)
        
        # Find best solutions
        best_distance = min(result['distance'] for result in valid_results.values())
        best_time = min(result['time'] for result in valid_results.values())
        
        # Print header
        print(f"{'Algorithm':<25} {'Distance':<12} {'Gap %':<8} {'Time (s)':<10} {'Efficiency':<12}")
        print("-" * 80)
        
        # Sort by distance
        sorted_results = sorted(valid_results.items(), key=lambda x: x[1]['distance'])
        
        for alg_name, result in sorted_results:
            distance = result['distance']
            time_taken = result['time']
            
            gap_percent = ((distance - best_distance) / best_distance) * 100
            efficiency = best_distance / (distance * time_taken)  # Quality per unit time
            
            print(f"{alg_name:<25} {distance:<12.2f} {gap_percent:<8.2f} {time_taken:<10.3f} {efficiency:<12.4f}")
        
        print("-" * 80)
        print(f"Best distance: {best_distance:.2f}")
        print(f"Best time: {best_time:.3f} seconds")
        
        # Statistical summary
        distances = [result['distance'] for result in valid_results.values()]
        times = [result['time'] for result in valid_results.values()]
        
        print(f"\nStatistical Summary:")
        print(f"Distance - Mean: {np.mean(distances):.2f}, Std: {np.std(distances):.2f}")
        print(f"Time - Mean: {np.mean(times):.3f}s, Std: {np.std(times):.3f}s")


def create_sample_problems() -> Dict[str, TSPInstance]:
    """Create a set of sample TSP problems for testing."""
    problems = {}
    
    # Small random problem
    cities_small = TSPGenerator.generate_random_cities(10, seed=42)
    problems['Small Random (10 cities)'] = TSPInstance(cities_small)
    
    # Medium random problem
    cities_medium = TSPGenerator.generate_random_cities(20, seed=42)
    problems['Medium Random (20 cities)'] = TSPInstance(cities_medium)
    
    # Large random problem
    cities_large = TSPGenerator.generate_random_cities(30, seed=42)
    problems['Large Random (30 cities)'] = TSPInstance(cities_large)
    
    # Circular arrangement
    cities_circular = TSPGenerator.generate_circular_cities(15, radius=50, noise=5)
    problems['Circular (15 cities)'] = TSPInstance(cities_circular)
    
    # Clustered cities
    cities_clustered = TSPGenerator.generate_clustered_cities(20, num_clusters=4)
    problems['Clustered (20 cities)'] = TSPInstance(cities_clustered)
    
    return problems


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description='Nature-Inspired TSP Solver')
    parser.add_argument('--problem', type=str, choices=['small', 'medium', 'large', 'circular', 'clustered', 'all'],
                       default='medium', help='Problem type to solve')
    parser.add_argument('--algorithm', type=str, 
                       choices=['ga', 'aco', 'mmas', 'pso', 'sa', 'mssa', 'all'],
                       default='all', help='Algorithm to run')
    parser.add_argument('--visualize', action='store_true', help='Show visualization')
    parser.add_argument('--save', action='store_true', help='Save results and plots')
    parser.add_argument('--cities', type=int, help='Number of cities (for custom problem)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create or select problem
    if args.cities:
        print(f"Creating custom problem with {args.cities} cities...")
        cities = TSPGenerator.generate_random_cities(args.cities, seed=args.seed)
        tsp_instance = TSPInstance(cities)
        problem_name = f"Custom ({args.cities} cities)"
    else:
        problems = create_sample_problems()
        
        if args.problem == 'all':
            # Run on all problems
            for problem_name, tsp_instance in problems.items():
                print(f"\n{'='*60}")
                print(f"SOLVING: {problem_name}")
                print(f"{'='*60}")
                
                benchmark = TSPBenchmark(tsp_instance)
                results = benchmark.run_all_algorithms(save_results=args.save)
                benchmark.print_detailed_comparison()
                
                if args.visualize:
                    benchmark.create_comparison_plots(save_plots=args.save)
            return
        else:
            problem_map = {
                'small': 'Small Random (10 cities)',
                'medium': 'Medium Random (20 cities)',
                'large': 'Large Random (30 cities)',
                'circular': 'Circular (15 cities)',
                'clustered': 'Clustered (20 cities)'
            }
            problem_name = problem_map[args.problem]
            tsp_instance = problems[problem_name]
    
    print(f"Solving: {problem_name}")
    print(f"Number of cities: {tsp_instance.num_cities}")
    
    # Create benchmark instance
    benchmark = TSPBenchmark(tsp_instance)
    
    # Run specified algorithm(s)
    if args.algorithm == 'all':
        results = benchmark.run_all_algorithms(save_results=args.save)
        benchmark.print_detailed_comparison()
        
        if args.visualize:
            benchmark.create_comparison_plots(save_plots=args.save)
    else:
        # Run single algorithm
        algorithm_map = {
            'ga': benchmark.run_genetic_algorithm,
            'aco': benchmark.run_ant_colony_optimization,
            'mmas': lambda: benchmark.run_ant_colony_optimization(variant='mmas'),
            'pso': benchmark.run_particle_swarm_optimization,
            'sa': benchmark.run_simulated_annealing,
            'mssa': lambda: benchmark.run_simulated_annealing(multistart=True)
        }
        
        algorithm_func = algorithm_map[args.algorithm]
        result = algorithm_func()
        
        print(f"\nResult: {result['distance']:.2f}")
        print(f"Time: {result['time']:.3f} seconds")
        
        if args.visualize:
            tsp_instance.visualize_tour(result['tour'], 
                                       title=f"{result['algorithm']} - Distance: {result['distance']:.2f}")


if __name__ == "__main__":
    main()