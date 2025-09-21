// Main Application Logic for TSP Web UI
class TSPApp {
    constructor() {
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.cities = [];
        this.currentAlgorithm = null;
        this.selectedAlgorithmType = null;
        this.isRunning = false;
        this.animationId = null;
        this.startTime = null;
        this.results = {};
        
        this.initializeApp();
    }

    initializeApp() {
        this.setupEventListeners();
        this.generateRandomCities();
        this.updateParameterPanel();
        this.resizeCanvas();
        this.draw();
    }

    setupEventListeners() {
        // Algorithm selection
        document.querySelectorAll('.algorithm-card').forEach(card => {
            card.addEventListener('click', () => {
                document.querySelectorAll('.algorithm-card').forEach(c => c.classList.remove('selected'));
                card.classList.add('selected');
                this.selectedAlgorithmType = card.dataset.algorithm;
                this.updateParameterPanel();
                this.logMessage(`Selected: ${card.querySelector('strong').textContent}`);
            });
        });

        // City count slider
        const cityCountSlider = document.getElementById('cityCount');
        const cityCountDisplay = document.getElementById('cityCountDisplay');
        cityCountSlider.addEventListener('input', (e) => {
            cityCountDisplay.textContent = e.target.value;
        });
        cityCountSlider.addEventListener('change', () => {
            this.generateRandomCities();
        });

        // Window resize
        window.addEventListener('resize', () => {
            this.resizeCanvas();
            this.draw();
        });

        // Canvas click to add cities manually
        this.canvas.addEventListener('click', (e) => {
            if (!this.isRunning) {
                const rect = this.canvas.getBoundingClientRect();
                const x = (e.clientX - rect.left) * (this.canvas.width / rect.width);
                const y = (e.clientY - rect.top) * (this.canvas.height / rect.height);
                
                if (this.cities.length < 50) {
                    this.cities.push({ x, y, id: this.cities.length });
                    document.getElementById('cityCount').value = this.cities.length;
                    document.getElementById('cityCountDisplay').textContent = this.cities.length;
                    this.draw();
                    this.logMessage(`Added city at (${Math.round(x)}, ${Math.round(y)}). Total: ${this.cities.length}`);
                }
            }
        });
    }

    resizeCanvas() {
        const container = this.canvas.parentElement;
        const containerWidth = container.clientWidth - 50; // padding
        this.canvas.width = containerWidth;
        this.canvas.height = 500;
        this.canvas.style.width = containerWidth + 'px';
        this.canvas.style.height = '500px';
    }

    generateRandomCities() {
        const count = parseInt(document.getElementById('cityCount').value);
        this.cities = [];
        
        const margin = 30;
        const width = this.canvas.width - 2 * margin;
        const height = this.canvas.height - 2 * margin;
        
        for (let i = 0; i < count; i++) {
            this.cities.push({
                x: margin + Math.random() * width,
                y: margin + Math.random() * height,
                id: i
            });
        }
        
        this.draw();
        this.logMessage(`Generated ${count} random cities`);
    }

    updateParameterPanel() {
        const parametersDiv = document.getElementById('algorithmParameters');
        
        if (!this.selectedAlgorithmType) {
            parametersDiv.innerHTML = '<p style="color: #666; text-align: center;">Select an algorithm to configure parameters</p>';
            return;
        }

        const parameterConfigs = {
            genetic: [
                { name: 'populationSize', label: 'Population Size', min: 50, max: 200, value: 100, step: 10 },
                { name: 'eliteSize', label: 'Elite Size', min: 10, max: 50, value: 20, step: 5 },
                { name: 'mutationRate', label: 'Mutation Rate', min: 0.001, max: 0.1, value: 0.01, step: 0.001 },
                { name: 'maxGenerations', label: 'Max Generations', min: 100, max: 1000, value: 500, step: 50 }
            ],
            aco: [
                { name: 'numAnts', label: 'Number of Ants', min: 10, max: 50, value: 20, step: 5 },
                { name: 'alpha', label: 'Alpha (Pheromone)', min: 0.5, max: 3.0, value: 1.0, step: 0.1 },
                { name: 'beta', label: 'Beta (Distance)', min: 1.0, max: 10.0, value: 5.0, step: 0.5 },
                { name: 'rho', label: 'Evaporation Rate', min: 0.01, max: 0.5, value: 0.1, step: 0.01 },
                { name: 'maxIterations', label: 'Max Iterations', min: 50, max: 500, value: 100, step: 25 }
            ],
            pso: [
                { name: 'numParticles', label: 'Number of Particles', min: 15, max: 50, value: 30, step: 5 },
                { name: 'w', label: 'Inertia Weight', min: 0.1, max: 1.5, value: 0.9, step: 0.1 },
                { name: 'c1', label: 'Cognitive Factor', min: 0.5, max: 4.0, value: 2.0, step: 0.1 },
                { name: 'c2', label: 'Social Factor', min: 0.5, max: 4.0, value: 2.0, step: 0.1 },
                { name: 'maxIterations', label: 'Max Iterations', min: 50, max: 500, value: 100, step: 25 }
            ],
            sa: [
                { name: 'initialTemp', label: 'Initial Temperature', min: 50, max: 200, value: 100, step: 10 },
                { name: 'coolingRate', label: 'Cooling Rate', min: 0.85, max: 0.99, value: 0.95, step: 0.01 },
                { name: 'minTemp', label: 'Min Temperature', min: 0.001, max: 1.0, value: 0.01, step: 0.001 },
                { name: 'maxIterations', label: 'Max Iterations', min: 1000, max: 20000, value: 10000, step: 1000 }
            ]
        };

        const params = parameterConfigs[this.selectedAlgorithmType];
        let html = '';
        
        params.forEach(param => {
            html += `
                <div class="parameter-group">
                    <label for="${param.name}">${param.label}:</label>
                    <input type="range" 
                           id="${param.name}" 
                           min="${param.min}" 
                           max="${param.max}" 
                           value="${param.value}" 
                           step="${param.step}"
                           onchange="updateParameterDisplay('${param.name}')">
                    <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                        <span style="font-size: 0.8em; color: #666;">${param.min}</span>
                        <span id="${param.name}Display" style="font-weight: 600;">${param.value}</span>
                        <span style="font-size: 0.8em; color: #666;">${param.max}</span>
                    </div>
                </div>
            `;
        });

        parametersDiv.innerHTML = html;
    }

    getAlgorithmParameters() {
        if (!this.selectedAlgorithmType) return {};
        
        const params = {};
        const inputs = document.querySelectorAll('#algorithmParameters input');
        
        inputs.forEach(input => {
            const value = parseFloat(input.value);
            params[input.id] = value;
        });
        
        return params;
    }

    async runSelectedAlgorithm() {
        if (!this.selectedAlgorithmType) {
            this.logMessage('❌ Please select an algorithm first!', 'error');
            return;
        }

        if (this.cities.length < 3) {
            this.logMessage('❌ Need at least 3 cities to solve TSP!', 'error');
            return;
        }

        this.isRunning = true;
        this.startTime = Date.now();
        
        // Update UI
        document.getElementById('runBtn').disabled = true;
        document.getElementById('runBtn').innerHTML = '<span class="status-indicator status-running"></span>Running...';
        this.updateProgress(0, 'Initializing algorithm...');
        
        // Create algorithm instance
        const params = this.getAlgorithmParameters();
        this.currentAlgorithm = AlgorithmFactory.create(this.selectedAlgorithmType, params);
        this.currentAlgorithm.cities = this.cities;
        this.currentAlgorithm.buildDistanceMatrix();
        
        this.logMessage(`🚀 Started ${this.getAlgorithmName()} with ${this.cities.length} cities`);
        this.logMessage(`Parameters: ${JSON.stringify(params)}`);
        
        // Start algorithm loop
        this.runAlgorithmStep();
    }

    async runAlgorithmStep() {
        if (!this.isRunning || !this.currentAlgorithm) return;

        try {
            const result = await this.currentAlgorithm.step();
            
            // Update metrics
            this.updateMetrics(result);
            
            // Update visualization
            if (this.currentAlgorithm.bestTour) {
                this.draw();
            }
            
            // Check convergence
            if (result.converged) {
                this.finishAlgorithm();
                return;
            }
            
            // Continue with next step
            setTimeout(() => this.runAlgorithmStep(), 50); // 50ms delay for visualization
            
        } catch (error) {
            this.logMessage(`❌ Error: ${error.message}`, 'error');
            this.stopAlgorithm();
        }
    }

    finishAlgorithm() {
        this.isRunning = false;
        const runtime = (Date.now() - this.startTime) / 1000;
        
        // Store results
        this.results[this.selectedAlgorithmType] = {
            distance: this.currentAlgorithm.bestDistance,
            runtime: runtime,
            tour: this.currentAlgorithm.bestTour
        };
        
        // Update UI
        document.getElementById('runBtn').disabled = false;
        document.getElementById('runBtn').innerHTML = '<span class="status-indicator status-complete"></span>Start Optimization';
        this.updateProgress(100, 'Optimization complete!');
        
        this.logMessage(`✅ ${this.getAlgorithmName()} completed in ${runtime.toFixed(2)}s`);
        this.logMessage(`🏆 Best distance found: ${this.currentAlgorithm.bestDistance.toFixed(2)}`);
        
        // Update comparison table
        this.updateComparisonTable();
        
        // Final draw
        this.draw();
    }

    stopAlgorithm() {
        this.isRunning = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        
        document.getElementById('runBtn').disabled = false;
        document.getElementById('runBtn').innerHTML = '<span class="status-indicator status-idle"></span>Start Optimization';
        this.updateProgress(0, 'Stopped');
        
        this.logMessage('⏹️ Algorithm stopped');
    }

    async runComparison() {
        if (this.cities.length < 3) {
            this.logMessage('❌ Need at least 3 cities for comparison!', 'error');
            return;
        }

        this.logMessage('🏁 Starting algorithm comparison...');
        
        const algorithms = ['genetic', 'aco', 'pso', 'sa'];
        const defaultParams = {
            genetic: { populationSize: 50, maxGenerations: 100 },
            aco: { numAnts: 15, maxIterations: 50 },
            pso: { numParticles: 20, maxIterations: 50 },
            sa: { maxIterations: 2000 }
        };

        for (const algoType of algorithms) {
            this.selectedAlgorithmType = algoType;
            
            // Highlight current algorithm
            document.querySelectorAll('.algorithm-card').forEach(card => {
                card.classList.remove('selected');
                if (card.dataset.algorithm === algoType) {
                    card.classList.add('selected');
                }
            });
            
            // Run algorithm with reduced parameters for speed
            const algorithm = AlgorithmFactory.create(algoType, defaultParams[algoType]);
            algorithm.cities = this.cities;
            algorithm.buildDistanceMatrix();
            
            const startTime = Date.now();
            let converged = false;
            
            while (!converged) {
                const result = await algorithm.step();
                converged = result.converged;
                
                // Update display
                if (algorithm.bestTour) {
                    this.currentAlgorithm = algorithm;
                    this.draw();
                }
                
                // Prevent infinite loops
                if (Date.now() - startTime > 10000) break; // 10 second timeout
            }
            
            const runtime = (Date.now() - startTime) / 1000;
            this.results[algoType] = {
                distance: algorithm.bestDistance,
                runtime: runtime,
                tour: algorithm.bestTour
            };
            
            this.logMessage(`✅ ${this.getAlgorithmName(algoType)} completed: ${algorithm.bestDistance.toFixed(2)} in ${runtime.toFixed(2)}s`);
        }
        
        this.updateComparisonTable();
        this.logMessage('🏆 Comparison complete!');
    }

    updateMetrics(result) {
        const runtime = (Date.now() - this.startTime) / 1000;
        
        document.getElementById('bestDistance').textContent = 
            this.currentAlgorithm.bestDistance.toFixed(2);
        document.getElementById('runtime').textContent = runtime.toFixed(1) + 's';
        
        if (result.generation !== undefined) {
            document.getElementById('currentIteration').textContent = result.generation;
            this.updateProgress((result.generation / result.maxGenerations) * 100, 
                `Generation ${result.generation}`);
        } else if (result.iteration !== undefined) {
            document.getElementById('currentIteration').textContent = result.iteration;
            const maxIter = this.currentAlgorithm.maxIterations || 100;
            this.updateProgress((result.iteration / maxIter) * 100, 
                `Iteration ${result.iteration}`);
        }
        
        // Algorithm-specific metrics
        if (result.avgFitness !== undefined) {
            document.getElementById('convergenceRate').textContent = 
                (result.avgFitness * 1000).toFixed(2);
        } else if (result.temperature !== undefined) {
            document.getElementById('convergenceRate').textContent = 
                result.temperature.toFixed(3) + '°';
        } else {
            document.getElementById('convergenceRate').textContent = '-';
        }
    }

    updateProgress(percentage, text) {
        document.getElementById('progressFill').style.width = percentage + '%';
        document.getElementById('progressText').textContent = text;
    }

    updateComparisonTable() {
        const tbody = document.getElementById('comparisonTableBody');
        const algorithms = [
            { type: 'genetic', icon: '🧬', name: 'Genetic Algorithm' },
            { type: 'aco', icon: '🐜', name: 'Ant Colony Optimization' },
            { type: 'pso', icon: '🦅', name: 'Particle Swarm Optimization' },
            { type: 'sa', icon: '🔥', name: 'Simulated Annealing' }
        ];

        // Find best distance for gap calculation
        const bestDistance = Math.min(...Object.values(this.results)
            .map(r => r.distance).filter(d => d !== undefined));

        tbody.innerHTML = algorithms.map(algo => {
            const result = this.results[algo.type];
            const status = result ? 'Complete' : 'Idle';
            const statusClass = result ? 'status-complete' : 'status-idle';
            const distance = result ? result.distance.toFixed(2) : '-';
            const runtime = result ? result.runtime.toFixed(2) + 's' : '-';
            const gap = result ? (((result.distance - bestDistance) / bestDistance) * 100).toFixed(2) + '%' : '-';

            return `
                <tr>
                    <td>${algo.icon} ${algo.name}</td>
                    <td><span class="status-indicator ${statusClass}"></span>${status}</td>
                    <td>${distance}</td>
                    <td>${runtime}</td>
                    <td>${gap}</td>
                </tr>
            `;
        }).join('');
    }

    draw() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw tour if available
        if (this.currentAlgorithm && this.currentAlgorithm.bestTour) {
            this.drawTour(this.currentAlgorithm.bestTour);
        }
        
        // Draw cities
        this.drawCities();
        
        // Draw info
        this.drawInfo();
    }

    drawTour(tour) {
        if (!tour || tour.length < 2) return;
        
        this.ctx.strokeStyle = '#667eea';
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([]);
        this.ctx.beginPath();
        
        const firstCity = this.cities[tour[0]];
        this.ctx.moveTo(firstCity.x, firstCity.y);
        
        for (let i = 1; i < tour.length; i++) {
            const city = this.cities[tour[i]];
            this.ctx.lineTo(city.x, city.y);
        }
        
        // Close the tour
        this.ctx.lineTo(firstCity.x, firstCity.y);
        this.ctx.stroke();
    }

    drawCities() {
        this.cities.forEach((city, index) => {
            // City circle
            this.ctx.fillStyle = '#4a5568';
            this.ctx.strokeStyle = '#fff';
            this.ctx.lineWidth = 2;
            this.ctx.beginPath();
            this.ctx.arc(city.x, city.y, 6, 0, 2 * Math.PI);
            this.ctx.fill();
            this.ctx.stroke();
            
            // City label
            this.ctx.fillStyle = '#2d3748';
            this.ctx.font = '12px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(index.toString(), city.x, city.y - 12);
        });
    }

    drawInfo() {
        if (this.currentAlgorithm && this.currentAlgorithm.bestDistance !== Infinity) {
            this.ctx.fillStyle = '#4a5568';
            this.ctx.font = '16px Arial';
            this.ctx.textAlign = 'left';
            this.ctx.fillText(
                `Best Distance: ${this.currentAlgorithm.bestDistance.toFixed(2)}`, 
                10, 25
            );
        }
    }

    getAlgorithmName(type = this.selectedAlgorithmType) {
        const names = {
            genetic: 'Genetic Algorithm',
            aco: 'Ant Colony Optimization',
            pso: 'Particle Swarm Optimization',
            sa: 'Simulated Annealing'
        };
        return names[type] || 'Unknown Algorithm';
    }

    logMessage(message, type = 'info') {
        const logArea = document.getElementById('logArea');
        const timestamp = new Date().toLocaleTimeString();
        const colors = {
            info: '#e2e8f0',
            error: '#fc8181',
            success: '#48bb78',
            warning: '#f6e05e'
        };
        
        const logEntry = document.createElement('div');
        logEntry.style.color = colors[type] || colors.info;
        logEntry.innerHTML = `[${timestamp}] ${message}`;
        
        logArea.appendChild(logEntry);
        logArea.scrollTop = logArea.scrollHeight;
        
        // Keep only last 100 messages
        while (logArea.children.length > 100) {
            logArea.removeChild(logArea.firstChild);
        }
    }
}

// Global functions for HTML event handlers
function updateParameterDisplay(paramName) {
    const input = document.getElementById(paramName);
    const display = document.getElementById(paramName + 'Display');
    display.textContent = input.value;
}

function generateRandomCities() {
    if (window.tspApp) {
        window.tspApp.generateRandomCities();
    }
}

function runSelectedAlgorithm() {
    if (window.tspApp) {
        window.tspApp.runSelectedAlgorithm();
    }
}

function runComparison() {
    if (window.tspApp) {
        window.tspApp.runComparison();
    }
}

function stopAlgorithm() {
    if (window.tspApp) {
        window.tspApp.stopAlgorithm();
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.tspApp = new TSPApp();
    window.tspApp.logMessage('🐝 Welcome to the Nature-Inspired TSP Solver!');
    window.tspApp.logMessage('👆 Click on cities to add them manually, or use the randomize button');
    window.tspApp.logMessage('🧠 Select an algorithm and click Start to begin optimization');
});