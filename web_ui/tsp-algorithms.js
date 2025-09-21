// TSP Algorithm Implementations for Web UI
class TSPBase {
    constructor() {
        this.cities = [];
        this.distances = {};
        this.bestTour = null;
        this.bestDistance = Infinity;
        this.history = [];
    }

    calculateDistance(city1, city2) {
        const dx = city1.x - city2.x;
        const dy = city1.y - city2.y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    buildDistanceMatrix() {
        this.distances = {};
        for (let i = 0; i < this.cities.length; i++) {
            for (let j = 0; j < this.cities.length; j++) {
                if (i !== j) {
                    const key = `${i}-${j}`;
                    this.distances[key] = this.calculateDistance(this.cities[i], this.cities[j]);
                }
            }
        }
    }

    getTourDistance(tour) {
        let totalDistance = 0;
        for (let i = 0; i < tour.length; i++) {
            const from = tour[i];
            const to = tour[(i + 1) % tour.length];
            const key = `${from}-${to}`;
            totalDistance += this.distances[key];
        }
        return totalDistance;
    }

    generateRandomTour() {
        const tour = Array.from({length: this.cities.length}, (_, i) => i);
        // Fisher-Yates shuffle
        for (let i = tour.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [tour[i], tour[j]] = [tour[j], tour[i]];
        }
        return tour;
    }

    updateBest(tour, distance) {
        if (distance < this.bestDistance) {
            this.bestDistance = distance;
            this.bestTour = [...tour];
            return true;
        }
        return false;
    }
}

class GeneticAlgorithm extends TSPBase {
    constructor(params = {}) {
        super();
        this.populationSize = params.populationSize || 100;
        this.eliteSize = params.eliteSize || 20;
        this.mutationRate = params.mutationRate || 0.01;
        this.maxGenerations = params.maxGenerations || 500;
        this.population = [];
        this.generation = 0;
        this.isRunning = false;
    }

    initializePopulation() {
        this.population = [];
        for (let i = 0; i < this.populationSize; i++) {
            this.population.push(this.generateRandomTour());
        }
    }

    evaluatePopulation() {
        return this.population.map(tour => ({
            tour: tour,
            fitness: 1 / this.getTourDistance(tour),
            distance: this.getTourDistance(tour)
        })).sort((a, b) => b.fitness - a.fitness);
    }

    selection(rankedPop) {
        const selectionResults = [];
        const df = rankedPop.map((individual, index) => ({
            ...individual,
            cumProb: rankedPop.slice(0, index + 1).reduce((sum, ind) => sum + ind.fitness, 0)
        }));
        
        const totalFitness = df[df.length - 1].cumProb;
        
        for (let i = 0; i < this.eliteSize; i++) {
            selectionResults.push(rankedPop[i].tour);
        }
        
        for (let i = 0; i < this.populationSize - this.eliteSize; i++) {
            const pick = Math.random() * totalFitness;
            for (let j = 0; j < df.length; j++) {
                if (pick <= df[j].cumProb) {
                    selectionResults.push(df[j].tour);
                    break;
                }
            }
        }
        
        return selectionResults;
    }

    crossover(parent1, parent2) {
        const start = Math.floor(Math.random() * parent1.length);
        const end = Math.floor(Math.random() * parent1.length);
        const [geneA, geneB] = start < end ? [start, end] : [end, start];
        
        const child = new Array(parent1.length).fill(-1);
        
        // Copy substring from parent1
        for (let i = geneA; i <= geneB; i++) {
            child[i] = parent1[i];
        }
        
        // Fill remaining positions from parent2
        let parent2Index = 0;
        for (let i = 0; i < child.length; i++) {
            if (child[i] === -1) {
                while (child.includes(parent2[parent2Index])) {
                    parent2Index++;
                }
                child[i] = parent2[parent2Index];
                parent2Index++;
            }
        }
        
        return child;
    }

    mutate(individual) {
        for (let swapped = 0; swapped < individual.length; swapped++) {
            if (Math.random() < this.mutationRate) {
                const swapWith = Math.floor(Math.random() * individual.length);
                [individual[swapped], individual[swapWith]] = [individual[swapWith], individual[swapped]];
            }
        }
        return individual;
    }

    async step() {
        if (this.generation === 0) {
            this.initializePopulation();
        }

        const rankedPop = this.evaluatePopulation();
        const matingpool = this.selection(rankedPop);
        const children = [];

        for (let i = 0; i < this.eliteSize; i++) {
            children.push(matingpool[i]);
        }

        for (let i = 0; i < matingpool.length - this.eliteSize; i++) {
            const parent1 = matingpool[Math.floor(Math.random() * this.eliteSize)];
            const parent2 = matingpool[Math.floor(Math.random() * this.eliteSize)];
            const child = this.mutate(this.crossover(parent1, parent2));
            children.push(child);
        }

        this.population = children;
        this.generation++;

        // Update best solution
        const best = rankedPop[0];
        this.updateBest(best.tour, best.distance);

        return {
            generation: this.generation,
            bestDistance: this.bestDistance,
            avgFitness: rankedPop.reduce((sum, ind) => sum + ind.fitness, 0) / rankedPop.length,
            converged: this.generation >= this.maxGenerations
        };
    }
}

class AntColonyOptimization extends TSPBase {
    constructor(params = {}) {
        super();
        this.numAnts = params.numAnts || 20;
        this.alpha = params.alpha || 1.0; // pheromone importance
        this.beta = params.beta || 5.0; // distance importance
        this.rho = params.rho || 0.1; // evaporation rate
        this.maxIterations = params.maxIterations || 100;
        this.pheromones = {};
        this.iteration = 0;
    }

    initializePheromones() {
        this.pheromones = {};
        for (let i = 0; i < this.cities.length; i++) {
            for (let j = 0; j < this.cities.length; j++) {
                if (i !== j) {
                    this.pheromones[`${i}-${j}`] = 0.1;
                }
            }
        }
    }

    constructAntSolution() {
        const tour = [];
        const visited = new Set();
        let current = Math.floor(Math.random() * this.cities.length);
        
        tour.push(current);
        visited.add(current);

        while (tour.length < this.cities.length) {
            const probabilities = [];
            let totalProb = 0;

            for (let next = 0; next < this.cities.length; next++) {
                if (!visited.has(next)) {
                    const pheromone = this.pheromones[`${current}-${next}`] || 0.1;
                    const distance = this.distances[`${current}-${next}`];
                    const prob = Math.pow(pheromone, this.alpha) * Math.pow(1.0 / distance, this.beta);
                    probabilities.push({ city: next, prob: prob });
                    totalProb += prob;
                }
            }

            // Roulette wheel selection
            const rand = Math.random() * totalProb;
            let cumulative = 0;
            for (const choice of probabilities) {
                cumulative += choice.prob;
                if (rand <= cumulative) {
                    current = choice.city;
                    tour.push(current);
                    visited.add(current);
                    break;
                }
            }
        }

        return tour;
    }

    updatePheromones(antTours) {
        // Evaporation
        for (const key in this.pheromones) {
            this.pheromones[key] *= (1 - this.rho);
        }

        // Deposition
        for (const tourData of antTours) {
            const { tour, distance } = tourData;
            const pheromoneDeposit = 1.0 / distance;
            
            for (let i = 0; i < tour.length; i++) {
                const from = tour[i];
                const to = tour[(i + 1) % tour.length];
                const key = `${from}-${to}`;
                this.pheromones[key] = (this.pheromones[key] || 0) + pheromoneDeposit;
            }
        }
    }

    async step() {
        if (this.iteration === 0) {
            this.initializePheromones();
        }

        const antTours = [];
        
        // Construct solutions for all ants
        for (let ant = 0; ant < this.numAnts; ant++) {
            const tour = this.constructAntSolution();
            const distance = this.getTourDistance(tour);
            antTours.push({ tour, distance });
            
            // Update global best
            this.updateBest(tour, distance);
        }

        // Update pheromones
        this.updatePheromones(antTours);
        this.iteration++;

        const avgDistance = antTours.reduce((sum, td) => sum + td.distance, 0) / antTours.length;

        return {
            iteration: this.iteration,
            bestDistance: this.bestDistance,
            avgDistance: avgDistance,
            converged: this.iteration >= this.maxIterations
        };
    }
}

class ParticleSwarmOptimization extends TSPBase {
    constructor(params = {}) {
        super();
        this.numParticles = params.numParticles || 30;
        this.w = params.w || 0.9; // inertia weight
        this.c1 = params.c1 || 2.0; // cognitive parameter
        this.c2 = params.c2 || 2.0; // social parameter
        this.maxIterations = params.maxIterations || 100;
        this.particles = [];
        this.globalBestPosition = null;
        this.globalBestDistance = Infinity;
        this.iteration = 0;
    }

    initializeParticles() {
        this.particles = [];
        for (let i = 0; i < this.numParticles; i++) {
            const position = this.generateRandomTour();
            const particle = {
                position: position,
                velocity: this.generateRandomSwapSequence(),
                bestPosition: [...position],
                bestDistance: this.getTourDistance(position)
            };
            
            if (particle.bestDistance < this.globalBestDistance) {
                this.globalBestDistance = particle.bestDistance;
                this.globalBestPosition = [...position];
                this.updateBest(position, particle.bestDistance);
            }
            
            this.particles.push(particle);
        }
    }

    generateRandomSwapSequence() {
        const numSwaps = Math.floor(Math.random() * 5) + 1;
        const swaps = [];
        for (let i = 0; i < numSwaps; i++) {
            const a = Math.floor(Math.random() * this.cities.length);
            const b = Math.floor(Math.random() * this.cities.length);
            if (a !== b) {
                swaps.push([a, b]);
            }
        }
        return swaps;
    }

    applySwaps(tour, swaps) {
        const newTour = [...tour];
        for (const [a, b] of swaps) {
            [newTour[a], newTour[b]] = [newTour[b], newTour[a]];
        }
        return newTour;
    }

    updateVelocity(particle) {
        // Simplified velocity update for TSP
        const r1 = Math.random();
        const r2 = Math.random();
        
        // Generate new random swaps with reduced intensity
        const newVelocity = [];
        
        // Cognitive component
        if (r1 < this.c1 / 10) {
            newVelocity.push(...this.generateRandomSwapSequence());
        }
        
        // Social component  
        if (r2 < this.c2 / 10) {
            newVelocity.push(...this.generateRandomSwapSequence());
        }
        
        // Limit velocity
        particle.velocity = newVelocity.slice(0, Math.min(3, newVelocity.length));
    }

    async step() {
        if (this.iteration === 0) {
            this.initializeParticles();
        }

        for (const particle of this.particles) {
            // Update velocity
            this.updateVelocity(particle);
            
            // Update position
            particle.position = this.applySwaps(particle.position, particle.velocity);
            const distance = this.getTourDistance(particle.position);
            
            // Update personal best
            if (distance < particle.bestDistance) {
                particle.bestDistance = distance;
                particle.bestPosition = [...particle.position];
            }
            
            // Update global best
            if (distance < this.globalBestDistance) {
                this.globalBestDistance = distance;
                this.globalBestPosition = [...particle.position];
                this.updateBest(particle.position, distance);
            }
        }

        // Update inertia weight
        this.w *= 0.99;
        this.iteration++;

        const avgDistance = this.particles.reduce((sum, p) => sum + this.getTourDistance(p.position), 0) / this.particles.length;

        return {
            iteration: this.iteration,
            bestDistance: this.bestDistance,
            avgDistance: avgDistance,
            inertiaWeight: this.w,
            converged: this.iteration >= this.maxIterations
        };
    }
}

class SimulatedAnnealing extends TSPBase {
    constructor(params = {}) {
        super();
        this.initialTemp = params.initialTemp || 100.0;
        this.coolingRate = params.coolingRate || 0.95;
        this.minTemp = params.minTemp || 0.01;
        this.maxIterations = params.maxIterations || 10000;
        this.currentSolution = null;
        this.currentDistance = Infinity;
        this.temperature = this.initialTemp;
        this.iteration = 0;
    }

    initializeSolution() {
        this.currentSolution = this.generateRandomTour();
        this.currentDistance = this.getTourDistance(this.currentSolution);
        this.updateBest(this.currentSolution, this.currentDistance);
    }

    generateNeighbor(solution) {
        const neighbor = [...solution];
        const i = Math.floor(Math.random() * neighbor.length);
        const j = Math.floor(Math.random() * neighbor.length);
        [neighbor[i], neighbor[j]] = [neighbor[j], neighbor[i]];
        return neighbor;
    }

    acceptanceProbability(currentEnergy, newEnergy, temperature) {
        if (newEnergy < currentEnergy) {
            return 1.0;
        }
        return Math.exp((currentEnergy - newEnergy) / temperature);
    }

    async step() {
        if (this.iteration === 0) {
            this.initializeSolution();
        }

        if (this.temperature > this.minTemp && this.iteration < this.maxIterations) {
            const neighbor = this.generateNeighbor(this.currentSolution);
            const neighborDistance = this.getTourDistance(neighbor);
            
            const acceptProb = this.acceptanceProbability(this.currentDistance, neighborDistance, this.temperature);
            
            if (Math.random() < acceptProb) {
                this.currentSolution = neighbor;
                this.currentDistance = neighborDistance;
            }
            
            // Update global best
            this.updateBest(this.currentSolution, this.currentDistance);
            
            // Cool down
            this.temperature *= this.coolingRate;
            this.iteration++;
        }

        return {
            iteration: this.iteration,
            temperature: this.temperature,
            currentDistance: this.currentDistance,
            bestDistance: this.bestDistance,
            converged: this.temperature <= this.minTemp || this.iteration >= this.maxIterations
        };
    }
}

// Algorithm factory
class AlgorithmFactory {
    static create(algorithmType, params = {}) {
        switch (algorithmType) {
            case 'genetic':
                return new GeneticAlgorithm(params);
            case 'aco':
                return new AntColonyOptimization(params);
            case 'pso':
                return new ParticleSwarmOptimization(params);
            case 'sa':
                return new SimulatedAnnealing(params);
            default:
                throw new Error(`Unknown algorithm type: ${algorithmType}`);
        }
    }
}

// Export for use in main app
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { AlgorithmFactory, TSPBase };
}