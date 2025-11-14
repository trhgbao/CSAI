import numpy as np
import random
import matplotlib.pyplot as plt

class GeneticAlgorithm_GraphColoring:
    def __init__(self, adjacency, n_colors,
                 n_pop=40, n_generations=300,
                 crossover_rate=0.85, mutation_rate=0.015,
                 n_elite=2, tournament_size=3,
                 seed=42):

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # adjacency là adjacency list
        self.adj = adjacency
        self.n_nodes = len(adjacency)
        self.n_colors = n_colors

        # Convert adjacency list => edge list (u, v)
        edges = []
        for u in range(self.n_nodes):
            for v in adjacency[u]:
                edges.append((u, v))
        self.edges = np.array(edges)  # shape (m, 2)

        # GA params
        self.n_pop = n_pop
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.n_elite = n_elite
        self.tournament_size = tournament_size

        # Initialize population
        self.population = np.random.randint(0, self.n_colors, size=(n_pop, self.n_nodes))
        self.fitness = np.zeros(n_pop)

        self.best_coloring = None
        self.best_used_colors = np.inf
        self.history = []

    # ============= FAST FITNESS =============
    def _calculate_fitness(self, coloring):
        u = self.edges[:, 0]
        v = self.edges[:, 1]
        return np.sum(coloring[u] == coloring[v])

    # ============= VECTORIZED SELECTION (tournament) =============
    def _selection(self):
        idx = np.random.randint(0, self.n_pop, size=(self.n_pop, self.tournament_size))
        best = idx[np.arange(self.n_pop), np.argmin(self.fitness[idx], axis=1)]
        return self.population[best]

    # ============= FAST MUTATION =============
    def _mutation(self, individuals):
        mask = np.random.rand(*individuals.shape) < self.mutation_rate
        random_colors = np.random.randint(0, self.n_colors, size=individuals.shape)
        individuals[mask] = random_colors[mask]
        return individuals

    # ============= CROSSOVER =============
    def _crossover_population(self, parents):
        new_pop = np.empty_like(parents)

        for i in range(0, self.n_pop, 2):
            p1 = parents[i]
            p2 = parents[(i + 1) % self.n_pop]

            if random.random() < self.crossover_rate:
                cp = random.randint(1, self.n_nodes - 1)
                new_pop[i] = np.concatenate([p1[:cp], p2[cp:]])
                new_pop[(i + 1) % self.n_pop] = np.concatenate([p2[:cp], p1[cp:]])
            else:
                new_pop[i] = p1.copy()
                new_pop[(i + 1) % self.n_pop] = p2.copy()

        return new_pop

    # ============= CORE GA LOOP =============
    def run(self, verbose=True):
        for gen in range(self.n_generations):

            # 1. Compute fitness for population
            self.fitness = np.array([self._calculate_fitness(ind) for ind in self.population])

            # Best in current generation
            best_idx = np.argmin(self.fitness)
            best_conflicts = self.fitness[best_idx]

            if best_conflicts == 0:  # valid solution found
                chrom = self.population[best_idx]
                used = len(np.unique(chrom))

                if used < self.best_used_colors:
                    self.best_used_colors = used
                    self.best_coloring = chrom.copy()
                    if verbose:
                        print(f"[Gen {gen}] Found new best valid solution: {used} colors")

            # 2. Elitism
            elite_idx = np.argsort(self.fitness)[:self.n_elite]
            elites = self.population[elite_idx]

            # 3. Selection
            parents = self._selection()

            # 4. Crossover
            offspring = self._crossover_population(parents)

            # 5. Mutation
            offspring = self._mutation(offspring)

            # 6. Build next population
            self.population = offspring
            self.population[:self.n_elite] = elites

            self.history.append(self.best_used_colors)

            if verbose:
                print(f"[Gen {gen}]  conflicts={best_conflicts}  best_colors={self.best_used_colors}")

        # fallback nếu không tìm giải pháp hợp lệ
        if self.best_coloring is None:
            idx = np.argmin(self.fitness)
            self.best_coloring = self.population[idx]
            print("No conflict-free solution found. Returning best-so-far.")

        return self.best_coloring, self.best_used_colors

    # ============= VISUALIZE =============
    def visualize(self, img_path):
        plt.figure(figsize=(6,4))
        plt.plot(self.history)
        plt.xlabel("Generation")
        plt.ylabel("Best Colors")
        plt.grid()
        plt.tight_layout()
        plt.savefig(img_path, dpi=300)
        plt.show()
