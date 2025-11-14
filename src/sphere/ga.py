import numpy as np
import matplotlib.pyplot as plt
import random
import time

def sphere_function(x):
    return np.sum(x ** 2)

class GeneticAlgorithm:
    def __init__(self, dim, bounds, n_pop=50, func=sphere_function,
                 crossover_rate=0.8, mutation_rate=0.05, mutation_strength=0.1,
                 n_elite=2, tournament_size=3,
                 max_iter=100, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        self.func = func
        self.dim = dim
        self.bounds = np.array(bounds)
        self.n_pop = n_pop
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate         # Tỷ lệ đột biến trên mỗi gen
        self.mutation_strength = mutation_strength # Độ mạnh của đột biến
        self.n_elite = n_elite                     # Số lượng cá thể ưu tú
        self.tournament_size = tournament_size     # Kích thước nhóm cho lựa chọn giải đấu
        self.max_iter = max_iter

        # Khởi tạo quần thể ban đầu
        self.population = np.random.uniform(bounds[0], bounds[1], (n_pop, dim))
        self.fitness = np.array([self.func(ind) for ind in self.population])

    def _selection(self):
        """Lựa chọn cha mẹ bằng phương pháp Tournament Selection."""
        selected_indices = []
        for _ in range(self.n_pop):
            # Chọn ngẫu nhiên các đối thủ cho giải đấu
            tournament_indices = np.random.randint(0, self.n_pop, self.tournament_size)
            tournament_fitness = self.fitness[tournament_indices]
            # Chọn người chiến thắng (có fitness nhỏ nhất)
            winner_index = tournament_indices[np.argmin(tournament_fitness)]
            selected_indices.append(winner_index)
        return selected_indices

    def _crossover(self, parent1, parent2):
        """Lai ghép hai cha mẹ để tạo ra hai con (Arithmetic Crossover)."""
        if random.random() < self.crossover_rate:
            alpha = random.random()
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = (1 - alpha) * parent1 + alpha * parent2
            return child1, child2
        return parent1, parent2

    def _mutation(self, individual):
        """Đột biến một cá thể bằng nhiễu Gaussian."""
        for i in range(self.dim):
            if random.random() < self.mutation_rate:
                noise = np.random.normal(loc=0, scale=self.mutation_strength)
                individual[i] += noise
        return individual

    def optimize(self, verbose=False):
        self.history = []
        t0 = time.time()
        # Tìm nghiệm tốt nhất ban đầu
        best_idx = np.argmin(self.fitness)
        global_best_f = self.fitness[best_idx]
        global_best_x = self.population[best_idx].copy()

        for it in range(self.max_iter):
            # Sắp xếp quần thể theo fitness
            sorted_indices = np.argsort(self.fitness)
            self.population = self.population[sorted_indices]
            self.fitness = self.fitness[sorted_indices]
            
            # Elitism: Giữ lại các cá thể tốt nhất
            new_population = list(self.population[:self.n_elite])
            
            # Lựa chọn cha mẹ
            parent_indices = self._selection()

            # Tạo ra thế hệ mới
            for i in range(self.n_elite, self.n_pop, 2):
                # Chọn hai cha mẹ
                p1 = self.population[parent_indices[i]]
                p2 = self.population[parent_indices[i+1 if i+1 < self.n_pop else i]] # Đảm bảo không vượt chỉ số

                # Lai ghép
                c1, c2 = self._crossover(p1, p2)
                
                # Đột biến
                c1 = self._mutation(c1)
                c2 = self._mutation(c2)

                new_population.extend([c1, c2])

            # Cắt bớt quần thể nếu số lượng lẻ
            self.population = np.array(new_population[:self.n_pop])
            
            # Giữ các cá thể trong biên
            self.population = np.clip(self.population, self.bounds[0], self.bounds[1])
            
            # Đánh giá quần thể mới
            self.fitness = np.array([self.func(ind) for ind in self.population])

            # Cập nhật nghiệm tốt nhất toàn cục
            current_best_idx = np.argmin(self.fitness)
            if self.fitness[current_best_idx] < global_best_f:
                global_best_f = self.fitness[current_best_idx]
                global_best_x = self.population[current_best_idx].copy()
            
            self.history.append(global_best_f)
            if verbose:
                print(f"Iter {it+1}/{self.max_iter} | Best = {global_best_f:.6f}")
            
            if global_best_f < float(10 ** (-6)):
                break
        self.global_best_x = global_best_x
        self.global_best_f = global_best_f
        t1 = time.time()
        self.elapsed_time = t1 - t0
        return global_best_x, global_best_f

    def visualize(self, img_path):
        plt.figure(figsize=(8, 10))
        plt.plot(self.history)
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness')
        plt.title(f"Dim: {self.dim} | Best Fitness: {self.global_best_f:.4e} | Time: {self.elapsed_time:.4}s")
        plt.grid(True)
        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.savefig(img_path, dpi=300)
        plt.show()
