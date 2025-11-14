import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 10)


class Solution:
    def __init__(self, graph, coloring=None):
        self.graph = graph
        n = self.graph.num_vertices
        if coloring is None:
            # numpy array of int32 colors
            self.coloring = np.random.randint(0, max(1, self.graph.max_colors), size=n, dtype=np.int32)
        else:
            arr = np.array(coloring, dtype=np.int32)
            if arr.shape[0] != n:
                # defensive: pad/truncate to n
                if arr.size < n:
                    new = np.random.randint(0, max(1, self.graph.max_colors), size=n, dtype=np.int32)
                    new[:arr.size] = arr
                    arr = new
                else:
                    arr = arr[:n].copy()
            self.coloring = arr

    def count_conflicts(self):
        c = 0
        A = self.graph.adjacency
        n = self.graph.num_vertices
        # iterate vertices, compare with neighbors via numpy
        for v in range(n):
            nbs = np.nonzero(A[v])[0]
            if nbs.size == 0:
                continue
            c += int(np.sum(self.coloring[nbs] == self.coloring[v]))
        return c // 2

    def count_colors(self):
        return int(np.unique(self.coloring).size)

    def fitness(self):
        c = self.count_conflicts()
        k = self.count_colors()
        return 1000 - k if c == 0 else -10 * c - 0.1 * k

    def copy(self):
        return Solution(self.graph, self.coloring.copy())

    def is_valid(self):
        return self.count_conflicts() == 0


# =========================
# DSATUR (keeps logic identical)
# =========================
class DSATUR:
    def __init__(self, graph):
        self.graph = graph

    def solve(self):
        n = self.graph.num_vertices
        A = self.graph.adjacency

        coloring = [-1] * n
        degrees = {v: int(np.sum(A[v])) for v in range(n)}

        # start from vertex with max degree
        first_vertex = max(degrees, key=degrees.get)
        coloring[first_vertex] = 0

        colored = {first_vertex}
        uncolored = set(range(n)) - colored

        while uncolored:
            sat_degrees = {}
            for v in uncolored:
                nbs = np.nonzero(A[v])[0]
                neighbor_colors = {coloring[nb] for nb in nbs if coloring[nb] != -1}
                sat_degrees[v] = len(neighbor_colors)

            max_sat = max(sat_degrees.values())
            candidates = [v for v, sat in sat_degrees.items() if sat == max_sat]

            if len(candidates) > 1:
                next_vertex = max(candidates, key=lambda v: degrees[v])
            else:
                next_vertex = candidates[0]

            nbs = np.nonzero(A[next_vertex])[0]
            neighbor_colors = {coloring[nb] for nb in nbs if coloring[nb] != -1}

            color = 0
            while color in neighbor_colors:
                color += 1

            coloring[next_vertex] = color
            colored.add(next_vertex)
            uncolored.remove(next_vertex)

        return Solution(self.graph, coloring)


# =========================
# Firefly Algorithm (logic unchanged; uses numpy coloring + adjacency)
# =========================
class FireflyAlgorithmGraphColoring:
    def __init__(self, graph, num_fireflies=40, max_iterations=400, use_dsatur=False):
        self.graph = graph
        self.num_fireflies = num_fireflies
        self.max_iterations = max_iterations
        self.use_dsatur = use_dsatur

        # FA params
        self.gamma = 0.8
        self.alpha_0 = 0.5
        self.alpha_min = 0.01
        self.alpha_decay = 0.97
        self.beta_min, self.beta_max = 0.2, 1.0

        # Local search
        self.local_search_prob = 0.5
        self.local_search_intensity = 15

        # history
        self.history_best_fitness = []
        self.history_avg_fitness = []
        self.history_worst_fitness = []
        self.history_best_colors = []
        self.history_best_conflicts = []

    def solve(self):
        start = time.time()

        # init population
        if self.use_dsatur:
            print("🎯 Using DSATUR for initialization...")
            dsat = DSATUR(self.graph)
            dsat_sol = dsat.solve()
            print(f"   DSATUR Result: {dsat_sol.count_colors()} colors, {dsat_sol.count_conflicts()} conflicts")

            self.population = []
            dsat_ratio = max(1, self.num_fireflies // 3)
            for _ in range(dsat_ratio):
                pert = dsat_sol.copy()
                num_changes = random.randint(max(1, self.graph.num_vertices // 20),
                                             max(1, self.graph.num_vertices // 10))
                idxs = np.random.choice(self.graph.num_vertices, size=num_changes, replace=False)
                for v in idxs:
                    pert.coloring[v] = random.randint(0, self.graph.max_colors - 1)
                self.population.append(pert)
            for _ in range(self.num_fireflies - dsat_ratio):
                self.population.append(Solution(self.graph))
        else:
            print("🎲 Using random initialization...")
            self.population = [Solution(self.graph) for _ in range(self.num_fireflies)]

        self.best_solution = max(self.population, key=lambda s: s.fitness()).copy()
        best_fit = self.best_solution.fitness()

        print("\n" + "=" * 70)
        print("Starting Firefly Algorithm for Graph Coloring")
        print("=" * 70)
        print(f"Fireflies: {self.num_fireflies}, Max Iter: {self.max_iterations}, DSATUR: {self.use_dsatur}")
        print(f"Initial Best: {self.best_solution.count_colors()} colors, {self.best_solution.count_conflicts()} conflicts")
        print("=" * 70 + "\n")

        for it in range(self.max_iterations):
            brightness = [s.fitness() for s in self.population]

            for i in range(self.num_fireflies):
                for j in range(self.num_fireflies):
                    if brightness[j] > brightness[i]:
                        self._move_firefly(i, j, it)

            if it % 10 == 0:
                self._elite_local_search()

            new_best = max(self.population, key=lambda s: s.fitness())
            if new_best.fitness() > best_fit:
                best_fit = new_best.fitness()
                self.best_solution = new_best.copy()

            fitness_values = [s.fitness() for s in self.population]
            self.history_best_fitness.append(max(fitness_values))
            self.history_avg_fitness.append(np.mean(fitness_values))
            self.history_worst_fitness.append(min(fitness_values))
            self.history_best_colors.append(self.best_solution.count_colors())
            self.history_best_conflicts.append(self.best_solution.count_conflicts())

            if it % 50 == 0:
                print(f"Iter {it:4d}: Fitness={best_fit:8.2f}, Colors={self.best_solution.count_colors():3d}, Conflicts={self.best_solution.count_conflicts():4d}")

        elapsed = time.time() - start
        print("\n" + "=" * 70)
        print("✅ Optimization Complete!")
        print("=" * 70)
        print(f"Time: {elapsed:.2f}s, Best Fitness: {best_fit:.2f}, Colors: {self.best_solution.count_colors()}, Conflicts: {self.best_solution.count_conflicts()}")
        print("=" * 70 + "\n")

        return self.best_solution

    def _move_firefly(self, i, j, it):
        sol_i = self.population[i]
        sol_j = self.population[j]

        # hamming distance
        diff_mask = sol_i.coloring != sol_j.coloring
        d = float(diff_mask.sum()) / float(self.graph.num_vertices)

        beta = self.beta_min + (self.beta_max - self.beta_min) * np.exp(-self.gamma * (d ** 2))
        alpha = max(self.alpha_0 * (self.alpha_decay ** it), self.alpha_min)

        new = sol_i.coloring.copy()
        n = self.graph.num_vertices

        rand_beta = np.random.random(n)
        beta_mask = rand_beta < beta
        new[beta_mask] = sol_j.coloring[beta_mask]

        rand_alpha = np.random.random(n)
        alpha_mask = rand_alpha < alpha
        if alpha_mask.any():
            new[alpha_mask] = np.random.randint(0, self.graph.max_colors, size=alpha_mask.sum(), dtype=np.int32)

        self.population[i] = Solution(self.graph, new)

    def _elite_local_search(self):
        elite = sorted(self.population, key=lambda s: s.fitness(), reverse=True)[:max(1, self.num_fireflies // 5)]
        for sol in elite:
            if random.random() < self.local_search_prob:
                self._repair_solution(sol)

    def _repair_solution(self, sol):
        A = self.graph.adjacency
        for _ in range(self.local_search_intensity):
            conflicts = []
            for v in range(self.graph.num_vertices):
                nbs = np.nonzero(A[v])[0]
                if nbs.size == 0:
                    continue
                if np.any(sol.coloring[nbs] == sol.coloring[v]):
                    conflicts.append(v)
            if not conflicts:
                break

            v = random.choice(conflicts)
            nbs = np.nonzero(A[v])[0]
            if nbs.size == 0:
                continue

            # count neighbor colors
            nbcols = sol.coloring[nbs]
            uniq, counts = np.unique(nbcols, return_counts=True)
            counts_arr = np.zeros(self.graph.max_colors, dtype=np.int32)
            counts_arr[uniq] = counts
            best_color = int(np.argmin(counts_arr))
            sol.coloring[v] = best_color

    def plot_convergence(self, save_path='convergence_graph_coloring.png'):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        iterations = range(len(self.history_best_fitness))

        ax1 = axes[0, 0]
        ax1.plot(iterations, self.history_best_fitness, '-', linewidth=2, label='Best')
        ax1.plot(iterations, self.history_avg_fitness, '--', linewidth=1.5, label='Average')
        ax1.plot(iterations, self.history_worst_fitness, ':', linewidth=1, label='Worst')
        ax1.set_xlabel('Iteration'); ax1.set_ylabel('Fitness')
        ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        ax2.plot(iterations, self.history_best_colors, linewidth=2)
        ax2.set_xlabel('Iteration'); ax2.set_ylabel('Number of Colors'); ax2.grid(True, alpha=0.3)

        ax3 = axes[1, 0]
        ax3.plot(iterations, self.history_best_conflicts, linewidth=2)
        ax3.set_xlabel('Iteration'); ax3.set_ylabel('Number of Conflicts'); ax3.grid(True, alpha=0.3)

        ax4 = axes[1, 1]
        ax4_twin = ax4.twinx()
        l1 = ax4.plot(iterations, self.history_best_colors, linewidth=2, label='Colors')
        l2 = ax4_twin.plot(iterations, self.history_best_conflicts, linewidth=2, label='Conflicts')
        ax4.set_xlabel('Iteration'); ax4.set_ylabel('Colors'); ax4_twin.set_ylabel('Conflicts')
        lines = l1 + l2
        labels = [ln.get_label() for ln in lines]
        ax4.legend(lines, labels, fontsize=9, loc='upper right')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Convergence plot saved to: {save_path}")

