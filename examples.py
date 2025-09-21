"""
Example Usage: Quick Start Guide

This script demonstrates how to use the nature-inspired TSP solvers
with various problem types and algorithm configurations.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.tsp_base import TSPGenerator, TSPInstance
from src.main import TSPBenchmark


def example_1_basic_usage():
    """Example 1: Basic usage with a small problem."""
    print("="*60)
    print("EXAMPLE 1: Basic Usage")
    print("="*60)
    
    # Generate a small random problem
    cities = TSPGenerator.generate_random_cities(12, seed=42)
    tsp_instance = TSPInstance(cities)
    
    print(f"Created TSP instance with {tsp_instance.num_cities} cities")
    
    # Create benchmark and run all algorithms
    benchmark = TSPBenchmark(tsp_instance)
    results = benchmark.run_all_algorithms()
    
    # Print comparison
    benchmark.print_detailed_comparison()
    
    # Visualize the best solution
    best_algorithm = min(results.keys(), key=lambda k: results[k].get('distance', float('inf')))
    best_result = results[best_algorithm]
    
    print(f"\nBest solution found by: {best_algorithm}")
    print(f"Distance: {best_result['distance']:.2f}")
    
    tsp_instance.visualize_tour(
        best_result['tour'], 
        title=f"Best Solution: {best_algorithm} (Distance: {best_result['distance']:.2f})"
    )


def example_2_individual_algorithms():
    """Example 2: Running individual algorithms with custom parameters."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Individual Algorithms")
    print("="*60)
    
    # Create a medium-sized problem
    cities = TSPGenerator.generate_random_cities(15, seed=123)
    tsp_instance = TSPInstance(cities)
    
    benchmark = TSPBenchmark(tsp_instance)
    
    # Run Genetic Algorithm with custom parameters
    print("Running Genetic Algorithm with custom parameters...")
    ga_result = benchmark.run_genetic_algorithm(
        population_size=50,
        elite_size=10,
        mutation_rate=0.02,
        max_generations=300,
        verbose=True
    )
    
    # Run Ant Colony Optimization
    print("\nRunning Ant Colony Optimization...")
    aco_result = benchmark.run_ant_colony_optimization(
        num_ants=15,
        alpha=1.5,
        beta=3.0,
        max_iterations=200,
        verbose=True
    )
    
    # Compare results
    print(f"\nGA Result: {ga_result['distance']:.2f} (Time: {ga_result['time']:.2f}s)")
    print(f"ACO Result: {aco_result['distance']:.2f} (Time: {aco_result['time']:.2f}s)")


def example_3_different_problem_types():
    """Example 3: Testing different problem types."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Different Problem Types")
    print("="*60)
    
    # Create different problem types
    problems = {
        'Random': TSPInstance(TSPGenerator.generate_random_cities(15, seed=42)),
        'Circular': TSPInstance(TSPGenerator.generate_circular_cities(15, radius=50, noise=3)),
        'Clustered': TSPInstance(TSPGenerator.generate_clustered_cities(15, num_clusters=3))
    }
    
    # Test each problem type with a fast algorithm
    for problem_name, tsp_instance in problems.items():
        print(f"\nSolving {problem_name} problem...")
        
        benchmark = TSPBenchmark(tsp_instance)
        
        # Run only Genetic Algorithm for speed
        result = benchmark.run_genetic_algorithm(
            population_size=30,
            max_generations=100,
            verbose=False
        )
        
        print(f"{problem_name}: Distance = {result['distance']:.2f}, Time = {result['time']:.2f}s")
        
        # Visualize the solution
        tsp_instance.visualize_tour(
            result['tour'], 
            title=f"{problem_name} Problem - GA Solution (Distance: {result['distance']:.2f})"
        )


def example_4_algorithm_comparison():
    """Example 4: Detailed algorithm comparison with plots."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Algorithm Comparison with Plots")
    print("="*60)
    
    # Create a moderately challenging problem
    cities = TSPGenerator.generate_clustered_cities(18, num_clusters=4, cluster_radius=12)
    tsp_instance = TSPInstance(cities)
    
    print(f"Created clustered problem with {tsp_instance.num_cities} cities")
    
    # Run comprehensive benchmark
    benchmark = TSPBenchmark(tsp_instance)
    results = benchmark.run_all_algorithms()
    
    # Print detailed comparison
    benchmark.print_detailed_comparison()
    
    # Create comparison plots
    benchmark.create_comparison_plots(save_plots=False)


def example_5_parameter_tuning():
    """Example 5: Parameter tuning example."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Parameter Tuning")
    print("="*60)
    
    # Create test problem
    cities = TSPGenerator.generate_random_cities(12, seed=999)
    tsp_instance = TSPInstance(cities)
    
    # Test different GA parameters
    mutation_rates = [0.005, 0.01, 0.02, 0.05]
    results = {}
    
    print("Testing different mutation rates for Genetic Algorithm:")
    
    for rate in mutation_rates:
        benchmark = TSPBenchmark(tsp_instance)
        result = benchmark.run_genetic_algorithm(
            mutation_rate=rate,
            max_generations=200,
            verbose=False
        )
        results[f"GA (mutation_rate={rate})"] = result
        print(f"Mutation rate {rate}: Distance = {result['distance']:.2f}")
    
    # Find best parameters
    best_config = min(results.keys(), key=lambda k: results[k]['distance'])
    print(f"\nBest configuration: {best_config}")
    print(f"Best distance: {results[best_config]['distance']:.2f}")


if __name__ == "__main__":
    print("Nature-Inspired TSP Solver - Example Usage")
    print("This script demonstrates various ways to use the TSP solver package.")
    
    try:
        example_1_basic_usage()
        example_2_individual_algorithms()
        example_3_different_problem_types()
        example_4_algorithm_comparison()
        example_5_parameter_tuning()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("Check the generated plots and outputs above.")
        print("="*60)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have installed the required dependencies:")
        print("pip install numpy matplotlib")