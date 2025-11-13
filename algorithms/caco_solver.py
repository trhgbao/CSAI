import numpy as np
import random

class ContinuousACOSphere:
    def __init__(self, objective_func, bounds, dims, n_ants, n_archive, q, xi, n_iter):
        self.func = objective_func
        self.dims = dims
        self.bounds = bounds
        self.n_ants = n_ants
        self.n_archive = n_archive
        self.q = q
        self.xi = xi

        min_bound, max_bound = self.bounds
        self.archive = np.random.uniform(min_bound, max_bound, (n_archive, dims))
        self.archive_fitness = np.array([self.func(x) for x in self.archive])
        
        sorted_indices = np.argsort(self.archive_fitness)
        self.archive = self.archive[sorted_indices]
        self.archive_fitness = self.archive_fitness[sorted_indices]
        
        self.gbest_val = self.archive_fitness[0]
        self.history = [self.gbest_val]

    def _calculate_weights(self):
        ranks = np.arange(1, self.n_archive + 1)
        weights = (1 / (self.q * self.n_archive * np.sqrt(2 * np.pi))) * \
                  np.exp(-((ranks - 1)**2) / (2 * (self.q**2) * (self.n_archive**2)))
        return weights / np.sum(weights)

    def step(self):
        weights = self._calculate_weights()
        selected_indices = np.random.choice(self.n_archive, size=self.n_ants, p=weights)
        
        new_ants = np.zeros((self.n_ants, self.dims))
        for i in range(self.n_ants):
            mu = self.archive[selected_indices[i]]
            sigma_sum = np.sum([np.abs(self.archive[j] - mu) for j in range(self.n_archive)], axis=0)
            sigma = self.xi * (sigma_sum / (self.n_archive - 1))
            new_ants[i] = np.random.normal(loc=mu, scale=sigma)
        
        min_bound, max_bound = self.bounds
        new_ants = np.clip(new_ants, min_bound, max_bound)

        new_fitness = np.array([self.func(ant) for ant in new_ants])
        combined_solutions = np.vstack([self.archive, new_ants])
        combined_fitness = np.hstack([self.archive_fitness, new_fitness])

        sorted_idx = np.argsort(combined_fitness)
        self.archive = combined_solutions[sorted_idx][:self.n_archive]
        self.archive_fitness = combined_fitness[sorted_idx][:self.n_archive]

        if self.archive_fitness[0] < self.gbest_val:
            self.gbest_val = self.archive_fitness[0]
        
        self.history.append(self.gbest_val)
        return self._get_state()

    def _get_state(self):
        # Để trực quan hóa, ta sẽ coi toàn bộ archive là "bầy đàn"
        return {
            "positions": self.archive,
            "gbest_val": self.gbest_val,
            "history": self.history
        }