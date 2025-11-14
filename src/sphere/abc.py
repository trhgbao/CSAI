import numpy as np
import time
import random


class ABC_Sphere_NP:
    def __init__(self, n, SN, MAX_ITER, LIMIT, lb, ub):
        self.n = n
        self.SN = SN
        self.MAX_ITER = MAX_ITER
        self.LIMIT = LIMIT
        self.lb = lb
        self.ub = ub

        # Population matrices
        self.X = np.zeros((SN, n), dtype=float)
        self.fitness = np.zeros(SN, dtype=float)
        self.trial = np.zeros(SN, dtype=int)

        # store best as a separate copy to avoid shared-reference bugs
        self.bestX = np.zeros(n, dtype=float)
        self.bestFitness = -np.inf

        self.history_best = []
        self.history_fx = []

    def evaluate_population(self, X):
        # Fitness = 1 / (1 + sum(x^2))
        return 1.0 / (1.0 + np.sum(X * X, axis=1))

    def initialize(self):
        self.X = np.random.uniform(self.lb, self.ub, (self.SN, self.n))
        self.fitness = self.evaluate_population(self.X)
        self.trial[:] = 0
        # initialize best as copy
        idx = np.argmax(self.fitness)
        self.bestX = self.X[idx].copy()
        self.bestFitness = float(self.fitness[idx])
    def employedPhase(self):
        for i in range(self.SN):
            j = np.random.randint(0, self.SN)
            while j == i:
                j = np.random.randint(0, self.SN)
            k = np.random.randint(0, self.n)

            phi = np.random.uniform(-1, 1)
            newX = self.X[i].copy()
            newX[k] += phi * (self.X[i, k] - self.X[j, k])
            newX = np.clip(newX, self.lb, self.ub)

            newFit = 1.0 / (1.0 + np.sum(newX * newX))
            if newFit > self.fitness[i]:
                self.X[i] = newX  # cập nhật ngay → asynchronous
                self.fitness[i] = newFit
                self.trial[i] = 0
            else:
                self.trial[i] += 1

    def onlookerPhase(self):
        total = self.fitness.sum()
        if total == 0:
            prob = np.ones(self.SN) / self.SN
        else:
            prob = self.fitness / total

        # simple loop (keeps behavior similar to original)
        for _ in range(self.SN):
            i = np.random.choice(self.SN, p=prob)
            j = np.random.randint(0, self.SN)
            if j == i:
                j = (j + 1) % self.SN
            k = np.random.randint(0, self.n)
            phi = random.uniform(-1, 1)

            newX = self.X[i].copy()
            newX[k] += phi * (self.X[i, k] - self.X[j, k])
            # clip single value
            newX[k] = np.clip(newX[k], self.lb, self.ub)

            newFit = 1.0 / (1.0 + np.sum(newX * newX))
            if newFit > self.fitness[i]:
                self.X[i] = newX
                self.fitness[i] = newFit
                self.trial[i] = 0
            else:
                self.trial[i] += 1

    def scoutPhase(self):
        scouts = self.trial > self.LIMIT
        if np.any(scouts):
            self.X[scouts] = np.random.uniform(self.lb, self.ub, (np.sum(scouts), self.n))
            self.fitness[scouts] = self.evaluate_population(self.X[scouts])
            self.trial[scouts] = 0

    def memorizeBest(self):
        # find best in current population
        new_best_idx = int(np.argmax(self.fitness))
        new_best_fit = float(self.fitness[new_best_idx])
        if new_best_fit > self.bestFitness:
            # copy the row so future changes to self.X won't mutate best
            self.bestX = self.X[new_best_idx].copy()
            self.bestFitness = new_best_fit

    def run(self):
        self.initialize()
        for it in range(self.MAX_ITER):
            self.employedPhase()
            self.onlookerPhase()
            self.scoutPhase()
            self.memorizeBest()

            if it % 50 == 0:
                print(f"Iter {it}: Best fitness = {self.bestFitness}")
            eps = 1e-10
            if abs(1.0 - self.bestFitness) <= eps:
                print(f"Early stop: đạt cực gần tối ưu tại vòng {it}.")
                break

        fx = (1.0 / self.bestFitness) - 1.0  # <-- lấy f(x) từ fitness
        print(f"\nFinal Best Fitness: {self.bestFitness}")
        print(f"Corresponding f(x) = {fx}")  # <-- in ra anh cần
        print("Best Solution:", self.bestX)


if __name__ == "__main__":
    random.seed(int(time.time()))
    np.random.seed(int(time.time()))
    n = 3
    SN = 50
    MAX_ITER = 20000
    LIMIT = 50
    lb, ub = -5.12, 5.12
    start = time.time()  # <---- BẮT ĐẦU ĐO THỜI GIAN
    abc = ABC_Sphere_NP(n, SN, MAX_ITER, LIMIT, lb, ub)
    abc.run()
    end = time.time()  # <---- KẾT THÚC ĐO THỜI GIAN
    print(f"\nTotal runtime: {end - start:.4f} seconds")
