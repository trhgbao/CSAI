import numpy as np
import random
from copy import deepcopy

# --- PHIÊN BẢN CHO SPHERE (TỪ ABC_SPHERE.py) ---

class ArtificialBeeColonySphere:
    def __init__(self, objective_func, bounds, dims, n_pop, n_iter, limit):
        self.func = objective_func
        self.bounds = bounds
        self.n_pop = n_pop
        self.dims = dims
        self.limit = limit
        
        min_bound, max_bound = self.bounds
        self.positions = np.random.uniform(min_bound, max_bound, (self.n_pop, self.dims))
        self.fitness_values = np.array([self.func(p) for p in self.positions])
        self.trials = np.zeros(self.n_pop)

        best_idx = np.argmin(self.fitness_values)
        self.gbest_pos = self.positions[best_idx].copy()
        self.gbest_val = self.fitness_values[best_idx]
        self.history = [self.gbest_val]

    def step(self):
        self._employed_phase()
        self._onlooker_phase()
        self._scout_phase()
        self._memorize_best()
        self.history.append(self.gbest_val)
        return self._get_state()

    def _employed_phase(self):
        for i in range(self.n_pop):
            new_pos = self._create_neighbor_sphere(i)
            new_fit = self.func(new_pos)
            if 1.0 / (1.0 + new_fit) > 1.0 / (1.0 + self.fitness_values[i]):
                self.positions[i] = new_pos
                self.fitness_values[i] = new_fit
                self.trials[i] = 0
            else:
                self.trials[i] += 1
    
    def _onlooker_phase(self):
        fitness_for_prob = 1.0 / (1.0 + self.fitness_values)
        total = fitness_for_prob.sum()
        prob = fitness_for_prob / total if total > 0 else np.ones(self.n_pop) / self.n_pop

        for _ in range(self.n_pop):
            i = np.random.choice(self.n_pop, p=prob)
            new_pos = self._create_neighbor_sphere(i)
            new_fit = self.func(new_pos)
            if 1.0 / (1.0 + new_fit) > 1.0 / (1.0 + self.fitness_values[i]):
                self.positions[i] = new_pos
                self.fitness_values[i] = new_fit
                self.trials[i] = 0
            else:
                self.trials[i] += 1
    
    def _scout_phase(self):
        scouts = self.trials > self.limit
        if np.any(scouts):
            min_bound, max_bound = self.bounds
            self.positions[scouts] = np.random.uniform(min_bound, max_bound, (np.sum(scouts), self.dims))
            self.fitness_values[scouts] = self.func(self.positions[scouts])
            self.trials[scouts] = 0

    def _memorize_best(self):
        best_idx = np.argmin(self.fitness_values)
        if self.fitness_values[best_idx] < self.gbest_val:
            self.gbest_val = self.fitness_values[best_idx]
            self.gbest_pos = self.positions[best_idx].copy()

    def _create_neighbor_sphere(self, index):
        pos = self.positions[index].copy()
        partner_idx = random.choice([i for i in range(self.n_pop) if i != index])
        k = np.random.randint(0, self.dims)
        phi = np.random.uniform(-1, 1)
        pos[k] += phi * (self.positions[index, k] - self.positions[partner_idx, k])
        min_bound, max_bound = self.bounds
        pos[k] = np.clip(pos[k], min_bound, max_bound)
        return pos

    def _get_state(self):
        return {"positions": self.positions, "gbest_val": self.gbest_val, "history": self.history}

# --- PHIÊN BẢN CHO GRAPH COLORING (TỪ ABC_GC_NP.py) ---

class ArtificialBeeColonyGraphColoring:
    def __init__(self, graph, n_pop, n_iter, limit, penalty_weight):
        self.n = graph.number_of_nodes()
        self.adj = [list(graph.neighbors(i)) for i in range(self.n)]
        self.SN = n_pop
        self.LIMIT = limit
        self.penalty_weight = penalty_weight # Mặc dù thuật toán gốc không dùng, ta giữ để đồng bộ
        
        self.bees = [{'color': None, 'fitness': 0.0, 'trial': 0} for _ in range(self.SN)]
        self.bestBee = {'color': None, 'fitness': -1.0}
        self.maxColorUsed = 0
        self.basedBee = {'color': None}
        self.history = []
        
        self._initialize(DIVERSITY_RATE=0.5)

    def _dsatur(self):
        color = np.zeros(self.n, dtype=int)
        sat = np.zeros(self.n, dtype=int)
        degree = np.array([len(self.adj[i]) for i in range(self.n)])
        for _ in range(self.n):
            uncolored = np.where(color == 0)[0]
            u = uncolored[0]
            for i in uncolored:
                if sat[i] > sat[u] or (sat[i] == sat[u] and degree[i] > degree[u]):
                    u = i
            used = np.zeros(self.n + 1, dtype=bool)
            used[color[self.adj[u]]] = True
            c = 1
            while used[c]: c += 1
            color[u] = c
            for v in self.adj[u]:
                if color[v] == 0:
                    neigh = set(color[self.adj[v]])
                    neigh.discard(0)
                    sat[v] = len(neigh)
        return color - 1 # Chuyển về 0-based

    def _count_conflicts(self, color):
        conflicts = 0
        for u in range(self.n):
            for v in self.adj[u]:
                if u < v and color[u] == color[v]:
                    conflicts += 1
        return conflicts

    def _evaluate(self, color):
        conflicts = self._count_conflicts(color)
        used_colors = len(np.unique(color))
        return used_colors + self.penalty_weight * conflicts

    def _select_conflict_vertex(self, color):
        conflict_nodes = [u for u in range(self.n) if any(color[u] == color[v] for v in self.adj[u])]
        return random.choice(conflict_nodes) if conflict_nodes else random.randint(0, self.n - 1)

    def _select_color_weighted(self, u, color):
        maxC = self.maxColorUsed + 1
        neighbor_colors = set(color[self.adj[u]])
        valid_colors = [c for c in range(maxC) if c not in neighbor_colors]
        if valid_colors:
            return random.choice(valid_colors)
        return random.randint(0, maxC)

    def _initialize(self, DIVERSITY_RATE):
        self.basedBee['color'] = self._dsatur()
        self.maxColorUsed = int(np.max(self.basedBee['color']))
        for i in range(self.SN):
            b = self.bees[i]
            b['color'] = self.basedBee['color'].copy()
            num_changes = max(1, int(self.n * DIVERSITY_RATE))
            for _ in range(num_changes):
                u = random.randint(0, self.n - 1)
                b['color'][u] = self._select_color_weighted(u, b['color'])
            b['fitness'] = self._evaluate(b['color'])
            b['trial'] = 0
        best = min(self.bees, key=lambda x: x['fitness'])
        self.bestBee = deepcopy(best)
        self.history.append(self.bestBee['fitness'])

    def _employed_phase(self):
        for b in self.bees:
            newColor = b['color'].copy()
            u = self._select_conflict_vertex(newColor)
            newColor[u] = self._select_color_weighted(u, newColor)
            newFit = self._evaluate(newColor)
            if newFit < b['fitness']:
                b['color'] = newColor
                b['fitness'] = newFit
                b['trial'] = 0
            else:
                b['trial'] += 1

    def _onlooker_phase(self):
        fitness_values = np.array([b['fitness'] for b in self.bees])
        max_fit = np.max(fitness_values)
        inverted_fitness = max_fit - fitness_values + 1e-5
        total = inverted_fitness.sum()
        prob = inverted_fitness / total if total > 0 else np.ones(self.SN) / self.SN

        for _ in range(self.SN):
            i = np.random.choice(self.SN, p=prob)
            b = self.bees[i]
            newColor = b['color'].copy()
            u = self._select_conflict_vertex(newColor)
            newColor[u] = self._select_color_weighted(u, newColor)
            newFit = self._evaluate(newColor)
            if newFit < b['fitness']:
                b['color'] = newColor
                b['fitness'] = newFit
                b['trial'] = 0
            else:
                b['trial'] += 1
    
    def _scout_phase(self):
        for b in self.bees:
            if b['trial'] > self.LIMIT:
                b['color'] = self._dsatur() # Tái khởi tạo bằng DSATUR
                b['fitness'] = self._evaluate(b['color'])
                b['trial'] = 0
    
    def _memorize_best(self):
        best_current = min(self.bees, key=lambda x: x['fitness'])
        if best_current['fitness'] < self.bestBee['fitness']:
            self.bestBee = deepcopy(best_current)

    def step(self):
        self._employed_phase()
        self._onlooker_phase()
        self._scout_phase()
        self._memorize_best()
        self.history.append(self.bestBee['fitness'])
        return self._get_state()
        
    def _get_state(self):
        num_colors = len(np.unique(self.bestBee['color']))
        num_conflicts = self._count_conflicts(self.bestBee['color'])
        return {
            "colors": self.bestBee['color'],
            "fitness_tuple": (self.bestBee['fitness'], num_colors, num_conflicts),
            "history": self.history
        }