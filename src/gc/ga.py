import numpy as np
import time
import random

class GeneticAlgorithm_GraphColoring:
    """
    GA cho Graph Coloring dùng danh sách kề (adjacency list)
    """
    def __init__(self, adjacency, n_colors, n_pop=50, n_generations=100,
                 crossover_rate=0.8, mutation_rate=0.05, n_elite=2,
                 tournament_size=3, seed=None):

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # -----------------------------
        # CHUYỂN SANG DANH SÁCH KỀ
        # -----------------------------
        self.adjacency = adjacency
        self.n_nodes = len(adjacency)
        self.n_colors = n_colors
        
        # Tạo danh sách cạnh (list of edges) để tính nhanh hơn
        self.edges = []
        for u in range(self.n_nodes):
            for v in adjacency[u]:
                if u < v:          # tránh trùng (u,v) và (v,u)
                    self.edges.append((u, v))
        self.edges = np.array(self.edges)

        # -----------------------------
        # GA parameters
        # -----------------------------
        self.n_pop = n_pop
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.n_elite = n_elite
        self.tournament_size = tournament_size

        # Khởi tạo quần thể
        self.population = np.random.randint(0, self.n_colors, size=(n_pop, self.n_nodes))
        self.fitness = np.zeros(n_pop)

        # best
        self.best_coloring = None
        self.best_used_colors = np.inf
        self.history = []

    # -----------------------------
    # FITNESS = SỐ XUNG ĐỘT
    # -----------------------------
    def _calculate_fitness(self, coloring):
        conflicts = 0
        for u, v in self.edges:
            if coloring[u] == coloring[v]:
                conflicts += 1
        return conflicts

    # -----------------------------
    # TOURNAMENT SELECTION
    # -----------------------------
    def _selection(self):
        selected_indices = []
        for _ in range(self.n_pop):
            indices = np.random.randint(0, self.n_pop, self.tournament_size)
            winner = indices[np.argmin(self.fitness[indices])]
            selected_indices.append(winner)
        return self.population[selected_indices]

    # -----------------------------
    # CROSSOVER 1-POINT
    # -----------------------------
    def _crossover(self, p1, p2):
        if random.random() < self.crossover_rate:
            cp = random.randint(1, self.n_nodes - 1)
            c1 = np.concatenate([p1[:cp], p2[cp:]])
            c2 = np.concatenate([p2[:cp], p1[cp:]])
            return c1, c2
        return p1.copy(), p2.copy()

    # -----------------------------
    # MUTATION
    # -----------------------------
    def _mutation(self, ind):
        for i in range(self.n_nodes):
            if random.random() < self.mutation_rate:
                ind[i] = random.randint(0, self.n_colors - 1)
        return ind

    # -----------------------------
    # MAIN LOOP
    # -----------------------------
    def run(self, verbose=True):
        for g in range(self.n_generations):

            # 1. evaluate
            self.fitness = np.array([self._calculate_fitness(ind) for ind in self.population])

            best_idx = np.argmin(self.fitness)
            best_f = self.fitness[best_idx]

            if best_f == 0:
                coloring = self.population[best_idx]
                n_used = len(np.unique(coloring))
                if n_used < self.best_used_colors:
                    self.best_used_colors = n_used
                    self.best_coloring = coloring.copy()
                    if verbose:
                        print(f"Gen {g+1:03d} | New best! Colors = {n_used}")

            # 2. Elitism
            next_gen = []
            elite_idx = np.argsort(self.fitness)[:self.n_elite]
            for idx in elite_idx:
                next_gen.append(self.population[idx])

            # 3. Selection
            parents = self._selection()

            # 4. Crossover + mutation
            for i in range(self.n_elite, self.n_pop, 2):
                p1 = parents[i]
                p2 = parents[i+1 if i+1 < self.n_pop else i]

                c1, c2 = self._crossover(p1, p2)
                next_gen.append(self._mutation(c1))
                if len(next_gen) < self.n_pop:
                    next_gen.append(self._mutation(c2))

            self.population = np.array(next_gen)

            if verbose:
                print(f"Gen {g+1:03d} | Min Conflicts = {best_f} | Best Colors = {self.best_used_colors}")
            
            self.history.append(self.best_used_colors)

        if self.best_coloring is None:
            idx = np.argmin(self.fitness)
            self.best_coloring = self.population[idx]
            print(f"⚠ No valid solution. Returning solution with {self.fitness[idx]} conflicts.")

        return self.best_coloring, self.best_used_colors, self.history
