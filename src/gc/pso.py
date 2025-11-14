import numpy as np
import time

def conflict_count(coloring, graph):
    conflicts = 0
    for u in range(len(graph)):
        for v in graph[u]:
            if coloring[u] == coloring[v]:
                conflicts += 1
    return conflicts // 2  # mỗi cạnh bị đếm 2 lần


class PSO_Coloring_Real:
    def __init__(self, graph, max_color, swarm_size=50, max_iter=2000,
                 w=0.7, c1=1.5, c2=1.5):
        self.graph = graph
        self.n = len(graph)
        self.k = max_color  # số màu tối đa
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.vmax = self.k
        self.positions = np.random.uniform(1, self.k, (swarm_size, self.n))
        self.velocities = np.random.uniform(-self.k, self.k, (swarm_size, self.n))

        self.personal_best = self.positions.copy()
        self.personal_best_fit = np.array([self.evaluate(p) for p in self.personal_best])

        best_idx = np.argmin(self.personal_best_fit)
        self.global_best = self.personal_best[best_idx].copy()
        self.global_best_fit = self.personal_best_fit[best_idx]

    def real_to_color(self, x):
        c = np.rint(x).astype(int)
        return c

    def evaluate(self, x):
        c = self.real_to_color(x)
        conf = conflict_count(c, self.graph)
        used_colors = len(set(c))
        return 100 * conf + used_colors  # fitness cần MIN hóa

    def optimize(self, print_every=50):
        for it in range(self.max_iter):
            for i in range(self.swarm_size):
                r1 = np.random.rand(self.n)
                r2 = np.random.rand(self.n)

                self.velocities[i] = (self.w * self.velocities[i] +
                                      self.c1 * r1 * (self.personal_best[i] - self.positions[i]) +
                                      self.c2 * r2 * (self.global_best - self.positions[i]))

                self.positions[i] += self.velocities[i]

                fit = self.evaluate(self.positions[i])
                if fit < self.personal_best_fit[i]:
                    self.personal_best_fit[i] = fit
                    self.personal_best[i] = self.positions[i].copy()

                if fit < self.global_best_fit:
                    self.global_best_fit = fit
                    self.global_best = self.positions[i].copy()

            if it % print_every == 0:
                print(f"Iter {it}: Best fitness = {self.global_best_fit}")

        best_coloring = self.real_to_color(self.global_best)
        best_conflicts = conflict_count(best_coloring, self.graph)
        used_colors = len(set(best_coloring))
        return self.global_best_fit, best_conflicts, best_coloring, used_colors

