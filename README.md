# Nature-Inspired Algorithms for Traveling Salesman Problem

A comprehensive implementation of various nature-inspired optimization algorithms for solving the Traveling Salesman Problem (TSP). This project demonstrates how biological and physical phenomena can inspire powerful optimization techniques.

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## 🚀 Features

### Implemented Algorithms

1. **Genetic Algorithm (GA)**
   - Order-based crossover (OX) and Partially Mapped Crossover (PMX)
   - Multiple mutation operators (swap, inversion, scramble)
   - Tournament and roulette wheel selection
   - Elite preservation and local search integration

2. **Ant Colony Optimization (ACO)**
   - Standard Ant Colony System (ACS)
   - Max-Min Ant System (MMAS) variant
   - Pheromone trail management with evaporation
   - Exploitation vs exploration balance

3. **Particle Swarm Optimization (PSO)**
   - Discrete PSO adapted for TSP
   - Velocity represented as swap operations
   - Inertia weight damping
   - Swarm diversification mechanisms

4. **Simulated Annealing (SA)**
   - Multiple cooling schedules (geometric, linear, exponential, adaptive)
   - Various neighborhood operators (2-opt, swap, insert, reverse)
   - Multi-start variant for improved robustness
   - Reheating mechanisms

### Key Features

- **Comprehensive Benchmarking**: Compare all algorithms on the same problem instances
- **Visualization**: Plot tours, convergence curves, and algorithm comparisons
- **Problem Generation**: Multiple TSP instance generators (random, circular, clustered)
- **Parameter Tuning**: Easy configuration of algorithm parameters
- **Performance Metrics**: Detailed analysis of solution quality and runtime
- **Extensible Design**: Easy to add new algorithms or problem variants

## 📦 Installation

### Prerequisites
- Python 3.7 or higher
- NumPy
- Matplotlib

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Quick Install
```bash
pip install numpy matplotlib
```

## 🔧 Usage

### Basic Usage

```python
from src.tsp_base import TSPGenerator, TSPInstance
from src.main import TSPBenchmark

# Create a TSP problem
cities = TSPGenerator.generate_random_cities(20, seed=42)
tsp_instance = TSPInstance(cities)

# Run all algorithms
benchmark = TSPBenchmark(tsp_instance)
results = benchmark.run_all_algorithms()

# Print comparison
benchmark.print_detailed_comparison()

# Create visualization
benchmark.create_comparison_plots()
```

### Command Line Interface

```bash
# Run all algorithms on a medium problem
python src/main.py --problem medium --algorithm all --visualize

# Run specific algorithm
python src/main.py --algorithm ga --cities 25 --visualize

# Custom problem size
python src/main.py --cities 30 --algorithm all --save
```

### Individual Algorithm Usage

```python
from src.genetic_algorithm import GeneticAlgorithmTSP

# Configure and run Genetic Algorithm
ga = GeneticAlgorithmTSP(
    tsp_instance, 
    population_size=100,
    elite_size=20,
    mutation_rate=0.01
)

result = ga.solve(max_generations=500, verbose=True)
print(f"Best distance: {result['distance']:.2f}")

# Visualize the solution
tsp_instance.visualize_tour(result['tour'])
ga.get_convergence_plot()
```

## 📊 Algorithm Comparison

Here's a typical performance comparison on a 20-city random instance:

| Algorithm | Distance | Time (s) | Gap % | Efficiency |
|-----------|----------|----------|-------|------------|
| Genetic Algorithm | 89.45 | 2.34 | 0.00 | 0.1837 |
| Ant Colony Optimization | 92.18 | 3.21 | 3.05 | 0.1342 |
| Max-Min Ant System | 90.76 | 2.89 | 1.46 | 0.1551 |
| Particle Swarm Optimization | 94.32 | 1.98 | 5.44 | 0.1920 |
| Simulated Annealing | 91.23 | 1.67 | 2.00 | 0.2280 |
| Multi-start SA | 88.97 | 5.12 | -0.54 | 0.0871 |

## 🧬 Algorithm Details

### Genetic Algorithm
- **Inspiration**: Natural evolution and survival of the fittest
- **Key Concepts**: Population, crossover, mutation, selection
- **Strengths**: Good balance of exploration and exploitation
- **Best For**: Medium to large instances with adequate runtime

### Ant Colony Optimization
- **Inspiration**: Foraging behavior of ant colonies
- **Key Concepts**: Pheromone trails, probabilistic path construction
- **Strengths**: Strong convergence properties, good for sparse graphs
- **Best For**: Problems where paths have natural interpretations

### Particle Swarm Optimization
- **Inspiration**: Flocking behavior of birds and fish
- **Key Concepts**: Social learning, velocity, position updates
- **Strengths**: Fast convergence, simple implementation
- **Best For**: Continuous optimization adapted for discrete problems

### Simulated Annealing
- **Inspiration**: Metallurgical annealing process
- **Key Concepts**: Temperature, cooling schedule, probabilistic acceptance
- **Strengths**: Simple, good local search, avoids local optima
- **Best For**: Quick solutions, local refinement

## 🎯 Problem Types

The solver supports various TSP instance types:

1. **Random Cities**: Uniformly distributed points
2. **Circular Arrangement**: Cities arranged in a circle (with optional noise)
3. **Clustered Cities**: Cities grouped in clusters
4. **Custom Instances**: Load from TSPLIB format files

```python
# Generate different problem types
random_cities = TSPGenerator.generate_random_cities(20)
circular_cities = TSPGenerator.generate_circular_cities(15, radius=50)
clustered_cities = TSPGenerator.generate_clustered_cities(20, num_clusters=4)
```

## 📈 Performance Analysis

### Convergence Tracking
All algorithms track detailed convergence history:
- Best solution over time
- Population/swarm diversity
- Algorithm-specific metrics (temperature, pheromone levels, etc.)

### Visualization Options
- Tour visualization with city coordinates
- Convergence plots
- Algorithm comparison charts
- Performance scatter plots

### Statistical Analysis
- Distance gap from best solution
- Runtime efficiency metrics
- Solution quality consistency
- Parameter sensitivity analysis

## 🔬 Advanced Features

### Parameter Tuning
```python
# Genetic Algorithm parameter sweep
for mutation_rate in [0.005, 0.01, 0.02, 0.05]:
    result = benchmark.run_genetic_algorithm(
        mutation_rate=mutation_rate,
        max_generations=200
    )
    print(f"Mutation rate {mutation_rate}: {result['distance']:.2f}")
```

### Multi-objective Optimization
The framework can be extended for multi-objective TSP variants:
- Distance vs. time trade-offs
- Risk-aware routing
- Energy-efficient paths

### Hybrid Algorithms
Combine multiple approaches:
```python
# GA with local search
ga_result = benchmark.run_genetic_algorithm(apply_local_search=True)

# Multi-start SA
mssa_result = benchmark.run_simulated_annealing(multistart=True, num_starts=5)
```

## 📝 Examples

See `examples.py` for comprehensive usage examples:

1. **Basic Usage**: Quick start with default parameters
2. **Individual Algorithms**: Custom parameter configuration
3. **Problem Types**: Testing different TSP variants
4. **Algorithm Comparison**: Detailed benchmarking with plots
5. **Parameter Tuning**: Optimization of algorithm parameters

Run examples:
```bash
python examples.py
```

## 🏗️ Project Structure

```
advance_search_bee/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── tsp_base.py              # Core TSP classes and utilities
│   ├── genetic_algorithm.py     # Genetic Algorithm implementation
│   ├── ant_colony_optimization.py # ACO and MMAS implementations
│   ├── particle_swarm_optimization.py # PSO implementation
│   ├── simulated_annealing.py   # SA and Multi-start SA
│   └── main.py                  # Main application and benchmarking
├── examples.py                  # Usage examples
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **New Algorithms**: Implement additional nature-inspired algorithms
2. **Problem Variants**: Add support for other TSP variants (ATSP, mTSP, etc.)
3. **Performance**: Optimize algorithm implementations
4. **Visualization**: Enhanced plotting and analysis tools
5. **Documentation**: Additional examples and tutorials

## 📚 References

### Books and Papers
1. Dorigo, M., & Stützle, T. (2004). *Ant Colony Optimization*. MIT Press.
2. Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*. Addison-Wesley.
3. Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. *Proceedings of ICNN'95*.
4. Kirkpatrick, S., et al. (1983). Optimization by simulated annealing. *Science*, 220(4598), 671-680.

### Online Resources
- [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/): Standard TSP benchmark instances
- [Concorde TSP Solver](http://www.math.uwaterloo.ca/tsp/concorde/): Optimal TSP solver
- [Nature-Inspired Computing](https://en.wikipedia.org/wiki/Natural_computing): Wikipedia overview

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by the excellent work in nature-inspired computing research
- TSP problem formulation based on classical operations research literature
- Visualization tools built on matplotlib
- Performance analysis inspired by optimization benchmarking practices

## 📞 Contact

For questions, suggestions, or collaborations, please open an issue on GitHub.

---

**Happy Optimizing!** 🐝🐜🦅