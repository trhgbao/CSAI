import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time
import random

class GeneticAlgorithm_GraphColoring:
    def __init__(self, adjacency, n_colors, n_pop=50, n_generations=100,
                 crossover_rate=0.8, mutation_rate=0.05, n_elite=2,
                 tournament_size=3, seed=42):

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.adjacency = adjacency                  # adjacency list
        self.n_nodes = len(adjacency)
        self.n_colors = n_colors

        self.n_pop = n_pop
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.n_elite = n_elite
        self.tournament_size = tournament_size

        # Tạo danh sách cạnh từ adjacency list
        self.edges = []
        for u in range(self.n_nodes):
            for v in adjacency[u]:
                if u < v:
                    self.edges.append((u, v))
        self.edges = np.array(self.edges)

        self.population = np.random.randint(0, self.n_colors, size=(n_pop, self.n_nodes))
        self.fitness = np.zeros(n_pop)

        self.best_coloring = None
        self.best_used_colors = np.inf
        self.history = []

    def _calculate_fitness(self, coloring):
        conflicts = 0
        for u, v in self.edges:
            if coloring[u] == coloring[v]:
                conflicts += 1
        return conflicts

    def _selection(self):
        """Lựa chọn cha mẹ bằng phương pháp Tournament Selection."""
        selected_indices = []
        for _ in range(self.n_pop):
            # Chọn ngẫu nhiên các đối thủ cho giải đấu
            tournament_indices = np.random.randint(0, self.n_pop, self.tournament_size)
            tournament_fitness = self.fitness[tournament_indices]
            # Người chiến thắng là người có fitness nhỏ nhất (ít xung đột nhất)
            winner_index = tournament_indices[np.argmin(tournament_fitness)]
            selected_indices.append(winner_index)
        return self.population[selected_indices]

    def _crossover(self, parent1, parent2):
        """Lai ghép hai cha mẹ bằng Single-Point Crossover."""
        if random.random() < self.crossover_rate:
            # Chọn một điểm cắt ngẫu nhiên
            crossover_point = random.randint(1, self.n_nodes - 1)
            child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
            return child1, child2
        return parent1.copy(), parent2.copy()

    def _mutation(self, individual):
        """Đột biến một cá thể."""
        for i in range(self.n_nodes):
            if random.random() < self.mutation_rate:
                # Thay đổi màu của gen này thành một màu ngẫu nhiên khác
                individual[i] = random.randint(0, self.n_colors - 1)
        return individual

    def run(self, verbose=True):
        for generation in range(self.n_generations):
            # 1. Đánh giá fitness cho toàn bộ quần thể
            self.fitness = np.array([self._calculate_fitness(ind) for ind in self.population])

            # 2. Tìm cá thể tốt nhất trong thế hệ hiện tại
            best_idx_current_gen = np.argmin(self.fitness)
            best_fitness_current_gen = self.fitness[best_idx_current_gen]

            # Nếu tìm thấy một lời giải hợp lệ (0 xung đột)
            if best_fitness_current_gen == 0:
                valid_coloring = self.population[best_idx_current_gen]
                used_colors = len(np.unique(valid_coloring))
                
                # Cập nhật lời giải tốt nhất toàn cục nếu số màu ít hơn
                if used_colors < self.best_used_colors:
                    self.best_used_colors = used_colors
                    self.best_coloring = valid_coloring.copy()
                    if verbose:
                        print(f"Generation {generation+1:03d} | New best solution found! Colors = {self.best_used_colors}")

            # 3. Tạo thế hệ mới
            next_generation = []
            
            # Elitism: Giữ lại các cá thể tốt nhất
            elite_indices = np.argsort(self.fitness)[:self.n_elite]
            for idx in elite_indices:
                next_generation.append(self.population[idx])

            # 4. Lựa chọn, Lai ghép và Đột biến
            selected_parents = self._selection()
            
            # Điền phần còn lại của thế hệ mới
            for i in range(self.n_elite, self.n_pop, 2):
                # Chọn cha mẹ từ pool đã được lựa chọn
                parent1 = selected_parents[i]
                # Đảm bảo không vượt quá chỉ số nếu n_pop là số lẻ
                parent2_idx = i + 1 if (i + 1) < self.n_pop else i
                parent2 = selected_parents[parent2_idx]

                child1, child2 = self._crossover(parent1, parent2)
                
                child1 = self._mutation(child1)
                child2 = self._mutation(child2)
                
                next_generation.append(child1)
                if len(next_generation) < self.n_pop:
                    next_generation.append(child2)
            
            self.population = np.array(next_generation)
            
            # In thông tin tiến trình
            if verbose:
                 print(f"Generation {generation+1:03d} | Min Conflicts = {best_fitness_current_gen} | Best Colors Found = {self.best_used_colors}")
            self.history.append(self.best_used_colors)

        # Nếu không tìm được lời giải hợp lệ, trả về None
        if self.best_coloring is None:
             # Cố gắng trả về lời giải ít xung đột nhất
             best_overall_idx = np.argmin(self.fitness)
             self.best_coloring = self.population[best_overall_idx]
             print(f"Warning: No valid coloring found. Returning solution with {self.fitness[best_overall_idx]} conflicts.")


        return self.best_coloring, self.best_used_colors

    def visuazlie(self, img_path):
        plt.figure(figsize=(6,4))
        plt.plot(self.history, marker='o')
        plt.xlabel("Iteration")
        plt.ylabel("Best #Colors")
        plt.title("ACO Graph Coloring - Convergence")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(img_path, dpi=300)
        plt.show()
