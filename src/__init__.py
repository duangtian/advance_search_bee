"""
Nature-Inspired Algorithms for Traveling Salesman Problem

This package implements various nature-inspired optimization algorithms
for solving the Traveling Salesman Problem (TSP).

Available algorithms:
- Genetic Algorithm (GA)
- Ant Colony Optimization (ACO)
- Max-Min Ant System (MMAS)
- Particle Swarm Optimization (PSO)
- Simulated Annealing (SA)
- Multi-start Simulated Annealing (MSSA)

Usage:
    from src.main import TSPBenchmark
    from src.tsp_base import TSPGenerator, TSPInstance
    
    # Create a problem
    cities = TSPGenerator.generate_random_cities(20, seed=42)
    tsp_instance = TSPInstance(cities)
    
    # Run benchmark
    benchmark = TSPBenchmark(tsp_instance)
    results = benchmark.run_all_algorithms()
"""

__version__ = "1.0.0"
__author__ = "Nature-Inspired TSP Solver"

# Import main classes for easy access
from .tsp_base import City, TSPInstance, TSPGenerator
from .genetic_algorithm import GeneticAlgorithmTSP
from .ant_colony_optimization import AntColonyOptimizationTSP, MaxMinAntSystem
from .particle_swarm_optimization import ParticleSwarmOptimizationTSP
from .simulated_annealing import SimulatedAnnealingTSP, MultiStartSimulatedAnnealing
from .main import TSPBenchmark

__all__ = [
    'City',
    'TSPInstance', 
    'TSPGenerator',
    'GeneticAlgorithmTSP',
    'AntColonyOptimizationTSP',
    'MaxMinAntSystem',
    'ParticleSwarmOptimizationTSP',
    'SimulatedAnnealingTSP',
    'MultiStartSimulatedAnnealing',
    'TSPBenchmark'
]