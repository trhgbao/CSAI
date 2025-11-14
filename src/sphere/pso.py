import numpy as np
import time
import matplotlib.pyplot as plt

class Particle:
    def __init__(self, dim, x_min, x_max, v_max):
        self.position = np.random.uniform(x_min, x_max, dim)
        self.velocity = np.random.uniform(-v_max, v_max, dim)
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')

class PSO:
    def __init__(self, dim=1000, swarm_size=50, max_iter=50000,
                 x_min=-5.12, x_max=5.12, w=0.7, c1=1.5, c2=1.5,
                 tol=1e-18):
        self.dim = dim
        self.x_min = x_min
        self.x_max = x_max
        self.v_max = (x_max - x_min)
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.tol = tol
        self.max_iter = max_iter

        if swarm_size is None:
            self.swarm_size = 50
        else:
            self.swarm_size = swarm_size

        self.swarm = [Particle(dim, x_min, x_max, self.v_max) for _ in range(self.swarm_size)]
        self.global_best = np.zeros(dim)
        self.global_best_fitness = float('inf')

        self.history = []   # <= thêm lưu lịch sử

    def sphere(self, x):
        return np.sum(x ** 2)

    def optimize(self):
        for p in self.swarm:
            p.best_fitness = self.sphere(p.position)
            if p.best_fitness < self.global_best_fitness:
                self.global_best_fitness = p.best_fitness
                self.global_best = p.position.copy()

        for it in range(self.max_iter):
            for p in self.swarm:
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                p.velocity = (self.w * p.velocity +
                              self.c1 * r1 * (p.best_position - p.position) +
                              self.c2 * r2 * (self.global_best - p.position))
                p.position += p.velocity
                fitness = self.sphere(p.position)
                if fitness < p.best_fitness:
                    p.best_fitness = fitness
                    p.best_position = p.position.copy()
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best = p.position.copy()

            print(f"Iter {it}: best fitness = {self.global_best_fitness}")

            self.history.append(self.global_best_fitness)   # <= lưu lại

            if self.global_best_fitness < self.tol:
                break

        return self.global_best_fitness, self.global_best


    def visualize(self, img_path):
        if not self.history:
            return
        plt.figure(figsize=(8, 5))
        plt.plot(self.history, linewidth=2)
        plt.title("PSO Convergence Curve")
        plt.xlabel("Iteration")
        plt.ylabel("Best Fitness")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(img_path, dpi=300)
        plt.show()
