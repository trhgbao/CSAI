import random
import time
import numpy as np
from copy import deepcopy

class Bee:
    def __init__(self):
        self.color = None  # np.array
        self.fitness = 0.0
        self.trial = 0

class ABC_GC:
    def __init__(self, n_, m_, sn_, limit_, max_iter_, adj_):
        self.n = n_
        self.m = m_
        self.SN = sn_
        self.LIMIT = limit_
        self.MAX_ITER = max_iter_
        self.adj = adj_
        self.bees = [Bee() for _ in range(self.SN)]
        self.bestBee = Bee()
        self.bestBee.fitness = -1
        self.maxColorUsed = 0
        self.basedBee = Bee()

    def DSATUR(self):
        n = self.n
        adj = self.adj
        color = np.zeros(n, dtype=int)
        sat = np.zeros(n, dtype=int)
        degree = np.array([len(adj[i]) for i in range(n)])

        for _ in range(n):
            uncolored = np.where(color == 0)[0]
            # chọn đỉnh theo sat, break tie bằng degree
            u = uncolored[0]
            for i in uncolored:
                if sat[i] > sat[u] or (sat[i] == sat[u] and degree[i] > degree[u]):
                    u = i

            used = np.zeros(n + 1, dtype=bool)
            used[color[adj[u]]] = True  # mark used colors
            c = 1
            while used[c]:
                c += 1
            color[u] = c

            for v in adj[u]:
                if color[v] == 0:
                    neigh = set(color[adj[v]])
                    neigh.discard(0)
                    sat[v] = len(neigh)

        return color

    def countConflicts(self, color):
        conflicts = 0
        for u in range(self.n):
            conflicts += np.sum(color[u] == np.array([color[v] for v in self.adj[u]]))
        return conflicts // 2  # mỗi cạnh đếm 2 lần

    def evaluate(self, color):
        conflicts = self.countConflicts(color)
        usedColors = len(np.unique(color))
        if conflicts > 0:
            return 1.0 / (1 + conflicts * 100)
        return 1.0 / usedColors

    def selectConflictVertex(self, color):
        conflict = np.zeros(self.n, dtype=int)
        for u in range(self.n):
            conflict[u] = np.sum(color[u] == np.array([color[v] for v in self.adj[u]]))
        total = np.sum(conflict + 1)
        r = random.random()
        prefix = 0
        for u in range(self.n):
            prefix += (conflict[u] + 1) / total
            if r <= prefix:
                return u
        return random.randint(0, self.n - 1)

    def selectColorWeighted(self, u, color):
        maxC = self.maxColorUsed + 2
        candidates = []
        scores = []

        used_set = set(color[self.adj[u]])
        S = maxC * (maxC + 1) / 2
        for c in range(1, maxC + 1):
            if c not in used_set:
                freq = np.sum(color == c)
                score = 1 + freq + c / S
                candidates.append(c)
                scores.append(score)

        if not candidates:
            K = self.maxColorUsed
            weights = np.array([K - c + 1 for c in range(1, K + 1)])
            probs = weights / np.sum(weights)
            r = random.random()
            prefix = 0
            for c, p in zip(range(1, K + 1), probs):
                prefix += p
                if r <= prefix:
                    return c
            return K

        invSum = np.sum(1.0 / np.array(scores))
        prob = (1.0 / np.array(scores)) / invSum
        r = random.random()
        prefix = 0
        for i in range(len(candidates)):
            prefix += prob[i]
            if r <= prefix:
                return candidates[i]
        return candidates[-1]

    def initialize(self, DIVERSITY_RATE=0.5):
        self.basedBee.color = self.DSATUR()
        self.maxColorUsed = int(np.max(self.basedBee.color))

        for i in range(self.SN):
            b = Bee()
            b.color = self.basedBee.color.copy()
            num_changes = max(1, int(self.n * DIVERSITY_RATE))
            for _ in range(num_changes):
                u = random.randint(0, self.n - 1)
                b.color[u] = self.selectColorWeighted(u, b.color)
            b.fitness = self.evaluate(b.color)
            b.trial = 0
            self.bees[i] = b

        best = max(self.bees, key=lambda x: x.fitness)
        self.bestBee = Bee()
        self.bestBee.color = best.color.copy()
        self.bestBee.fitness = best.fitness
        self.bestBee.trial = best.trial

    def employedPhase(self):
        for b in self.bees:
            newColor = b.color.copy()
            u = self.selectConflictVertex(newColor)
            newColor[u] = self.selectColorWeighted(u, newColor)
            newFit = self.evaluate(newColor)
            if newFit > b.fitness:
                b.color = newColor
                b.fitness = newFit
                b.trial = 0
            else:
                b.trial += 1

    def onlookerPhase(self):
        totalFit = np.sum([b.fitness for b in self.bees])
        prob = [b.fitness / totalFit for b in self.bees]
        for _ in range(self.SN):
            r = random.random()
            s = 0
            i = self.SN - 1
            for j in range(self.SN):
                s += prob[j]
                if r <= s:
                    i = j
                    break
            newColor = self.bees[i].color.copy()
            u = self.selectConflictVertex(newColor)
            newColor[u] = self.selectColorWeighted(u, newColor)
            newFit = self.evaluate(newColor)
            if newFit > self.bees[i].fitness:
                self.bees[i].color = newColor
                self.bees[i].fitness = newFit
                self.bees[i].trial = 0
            else:
                self.bees[i].trial += 1

    def scoutPhase(self):
        for b in self.bees:
            if b.trial > self.LIMIT:
                newColor = self.basedBee.color.copy()
                num_changes = max(1, int(self.n * 0.5))
                for _ in range(num_changes):
                    u = random.randint(0, self.n-1)
                    neighborColors = set(newColor[v] for v in self.adj[u])
                    t = int(np.max(newColor))
                    valid = [c for c in range(1, t+1) if c not in neighborColors]
                    if valid:
                        weights = 1.0 / np.arange(1, len(valid)+1)
                        weights /= np.sum(weights)
                        r = random.random()
                        prefix = 0
                        for i, c in enumerate(valid):
                            prefix += weights[i]
                            if r <= prefix:
                                newColor[u] = c
                                break
                    else:
                        K = self.maxColorUsed + 1
                        S = (K + 1) * K // 2
                        r = random.randint(1, S)
                        prefix = 0
                        for c in range(1, K+1):
                            prefix += K - c + 1
                            if r <= prefix:
                                newColor[u] = c
                                break
                b.color = newColor
                b.fitness = self.evaluate(newColor)
                b.trial = 0

    def memorizeBest(self):
        for b in self.bees:
            if b.fitness > self.bestBee.fitness:
                self.bestBee = deepcopy(b)

    def run(self):
        self.initialize()
        for iter in range(self.MAX_ITER):
            self.employedPhase()
            self.onlookerPhase()
            self.scoutPhase()
            self.memorizeBest()
            if iter % 50 == 0:
                print("[Iter", iter, "] BestFitness=", self.bestBee.fitness,
                      ", Colors=", len(np.unique(self.bestBee.color)),
                      ", Conflicts=", self.countConflicts(self.bestBee.color))
        print("\n==== RESULT ====")
        print("Fitness:", self.bestBee.fitness)
        print("Conflicts:", self.countConflicts(self.bestBee.color))
        print("Colors used:", len(np.unique(self.bestBee.color)))
        print("Solution:", *self.bestBee.color)
