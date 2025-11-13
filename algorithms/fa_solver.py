import numpy as np
import random
import time
import math # Thêm math
import networkx as nx # Thêm networkx

class FireflyAlgorithmSphere:
    def __init__(self, objective_func, bounds, n_pop, n_iter, 
                 alpha0, alpha_min, alpha_decay, beta0, gamma, 
                 p_local, i_local):
        # --- THAM SỐ ĐẦU VÀO ---
        self.func = objective_func
        self.bounds = bounds
        self.n_pop = n_pop
        self.n_dims = len(bounds[0]) if hasattr(bounds[0], '__len__') else 2
        
        # Tham số Alpha động
        self.alpha0, self.alpha_min, self.alpha_decay = alpha0, alpha_min, alpha_decay
        # Các tham số FA kinh điển
        self.beta0, self.gamma = beta0, gamma
        # Tham số Local Search
        self.p_local, self.i_local = p_local, i_local

        # --- BIẾN TRẠNG THÁI ---
        self.current_alpha = self.alpha0 
        min_bound, max_bound = self.bounds
        self.positions = np.random.uniform(min_bound, max_bound, (self.n_pop, self.n_dims))
        self.intensities = np.array([self.func(p) for p in self.positions])

        best_idx = np.argmin(self.intensities)
        self.gbest_pos = self.positions[best_idx].copy()
        self.gbest_val = self.intensities[best_idx]
        self.history = [self.gbest_val]

    def _perform_local_search(self, starting_point):
        """Thực hiện tìm kiếm cục bộ (Hill Climbing đơn giản)"""
        current_solution = starting_point.copy()
        current_fitness = self.func(current_solution)
        
        for _ in range(self.i_local):
            # Tạo một hàng xóm bằng cách di chuyển một chút ngẫu nhiên
            step = (self.bounds[1] - self.bounds[0]) * 0.01 # Bước nhảy 1%
            neighbor = current_solution + np.random.uniform(-step, step, self.n_dims)
            neighbor = np.clip(neighbor, self.bounds[0], self.bounds[1])
            
            neighbor_fitness = self.func(neighbor)
            
            # Nếu hàng xóm tốt hơn, chấp nhận nó
            if neighbor_fitness < current_fitness:
                current_solution = neighbor
                current_fitness = neighbor_fitness
                
        return current_solution, current_fitness

    def step(self):
        # 1. Giai đoạn di chuyển toàn cục của Firefly
        for i in range(self.n_pop):
            for j in range(self.n_pop):
                if self.intensities[j] < self.intensities[i]:
                    distance_sq = np.sum((self.positions[i] - self.positions[j]) ** 2)
                    beta = self.beta0 * np.exp(-self.gamma * distance_sq)
                    random_term = self.current_alpha * (np.random.rand(self.n_dims) - 0.5)
                    self.positions[i] += beta * (self.positions[j] - self.positions[i]) + random_term
                    self.positions[i] = np.clip(self.positions[i], self.bounds[0], self.bounds[1])
                    self.intensities[i] = self.func(self.positions[i])

        # 2. Cập nhật lời giải tốt nhất từ giai đoạn toàn cục
        best_idx_after_global = np.argmin(self.intensities)
        if self.intensities[best_idx_after_global] < self.gbest_val:
            self.gbest_val = self.intensities[best_idx_after_global]
            self.gbest_pos = self.positions[best_idx_after_global].copy()

        # 3. Giai đoạn Tìm kiếm Cục bộ (Memetic step)
        if random.random() < self.p_local:
            # Thực hiện tìm kiếm cục bộ trên lời giải tốt nhất hiện tại
            refined_solution, refined_fitness = self._perform_local_search(self.gbest_pos)
            
            # Nếu tìm kiếm cục bộ tìm được lời giải còn tốt hơn nữa
            if refined_fitness < self.gbest_val:
                self.gbest_val = refined_fitness
                self.gbest_pos = refined_solution
                
                # Cập nhật lại con đom đóm tốt nhất trong quần thể
                self.positions[best_idx_after_global] = refined_solution
                self.intensities[best_idx_after_global] = refined_fitness

        # 4. Cập nhật Alpha cho thế hệ tiếp theo
        new_alpha = self.current_alpha * self.alpha_decay
        self.current_alpha = max(new_alpha, self.alpha_min)
            
        self.history.append(self.gbest_val)
        return self._get_state()

    def _get_state(self):
        return {"positions": self.positions, "gbest_val": self.gbest_val, "history": self.history}
    
from algorithms.traditional_solvers import SAGraphSolution as FA_GraphColoringSolution

# ============================================================================
# FIREFLY ALGORITHM FOR GRAPH COLORING (NEW)
# ============================================================================
class FireflyAlgorithmGraphColoring:
    def __init__(self, graph, penalty_weight, n_pop, n_iter, use_dsatur=False):
        # `graph` ở đây là một đối tượng networkx.Graph
        self.graph_nx = graph
        self.adj_list = [list(graph.neighbors(i)) for i in range(graph.number_of_nodes())]
        self.num_vertices = graph.number_of_nodes()
        
        degrees = [d for n, d in self.graph_nx.degree()]
        self.max_colors = max(degrees) + 1 if degrees else 1

        self.penalty_weight = penalty_weight # SA sẽ sử dụng energy riêng, nhưng giữ lại cho tương thích

        self.n_pop = n_pop
        self.n_iter = n_iter # Số vòng lặp tối đa
        self.use_dsatur = use_dsatur

        # Tham số FA từ FireflyAlgorithmGraphColoring của bạn
        self.gamma = 0.8
        self.alpha_0 = 0.5
        self.alpha_min = 0.01
        self.alpha_decay = 0.97
        self.beta_min, self.beta_max = 0.2, 1.0

        # Tham số Local Search
        self.local_search_prob = 0.5
        self.local_search_intensity = 15

        self.population = []
        self._initialize_population()

        self.best_solution = max(self.population, key=lambda s: s.fitness()).copy()
        self.best_fitness = self.best_solution.fitness()

        self.history = [self.best_fitness] # Lịch sử sẽ lưu best_fitness

    def _initialize_population(self):
        """Khởi tạo quần thể firefly."""
        if self.use_dsatur:
            # Import DSATUR (nếu cần, hoặc copy logic vào đây)
            # Giả định DSATUR là một hàm/class riêng biệt có thể tạo ra solution
            # Để đơn giản, tôi sẽ tái tạo một phần logic DSATUR ở đây hoặc bạn có thể import từ đâu đó
            try:
                from algorithms.cs_mcoa_solver import dsatur_coloring
                dsatur_coloring_arr = dsatur_coloring(self.num_vertices, self.adj_list)
                dsatur_solution = FA_GraphColoringSolution(self.adj_list, self.num_vertices, self.max_colors, dsatur_coloring_arr)
            except ImportError:
                print("Cảnh báo: Không tìm thấy dsatur_coloring trong cs_mcoa_solver. Sử dụng khởi tạo ngẫu nhiên.")
                dsatur_solution = FA_GraphColoringSolution(self.adj_list, self.num_vertices, self.max_colors)
            
            # 30% quần thể: DSATUR solution với nhiễu nhỏ
            dsatur_ratio_count = max(1, self.n_pop // 3)
            for _ in range(dsatur_ratio_count):
                perturbed = dsatur_solution.copy()
                num_changes = random.randint(self.num_vertices // 20, self.num_vertices // 10)
                for _ in range(num_changes):
                    v = random.randint(0, self.num_vertices - 1)
                    perturbed.coloring[v] = random.randint(0, self.max_colors - 1)
                self.population.append(perturbed)

            # Phần còn lại: random
            for _ in range(self.n_pop - dsatur_ratio_count):
                self.population.append(FA_GraphColoringSolution(self.adj_list, self.num_vertices, self.max_colors))
        else:
            self.population = [FA_GraphColoringSolution(self.adj_list, self.num_vertices, self.max_colors) for _ in range(self.n_pop)]


    def step(self):
        """Thực hiện một bước của thuật toán Firefly."""
        # Tính độ sáng (fitness) của tất cả firefly
        brightness = [s.fitness() for s in self.population]

        current_alpha = max(self.alpha_0 * (self.alpha_decay ** len(self.history)), self.alpha_min)

        # Di chuyển firefly tối hơn về phía firefly sáng hơn
        for i in range(self.n_pop):
            for j in range(self.n_pop):
                if brightness[j] > brightness[i]: # Firefly j sáng hơn (fitness cao hơn) firefly i
                    self._move_firefly(i, j, current_alpha)

        # Áp dụng local search cho các firefly tốt nhất
        # Có thể thêm điều kiện `if len(self.history) % 10 == 0:` nếu muốn chạy định kỳ
        self._elite_local_search()

        # Cập nhật best solution
        new_best_solution_obj = max(self.population, key=lambda s: s.fitness())
        if new_best_solution_obj.fitness() > self.best_fitness: # Tìm max fitness
            self.best_fitness = new_best_solution_obj.fitness()
            self.best_solution = new_best_solution_obj.copy()

        # Lưu lịch sử
        self.history.append(self.best_fitness)
        return self._get_state()

    def _move_firefly(self, i, j, alpha):
        sol_i, sol_j = self.population[i], self.population[j]

        # Tính khoảng cách Hamming (tỷ lệ đỉnh khác màu)
        d = sum(1 for a, b in zip(sol_i.coloring, sol_j.coloring) if a != b) / self.num_vertices

        # Độ hấp dẫn giảm theo khoảng cách
        beta = self.beta_min + (self.beta_max - self.beta_min) * np.exp(-self.gamma * d ** 2)

        new = sol_i.coloring.copy()
        for v in range(self.num_vertices):
            if random.random() < beta: # Hút về phía firefly j với xác suất beta
                new[v] = sol_j.coloring[v]
            if random.random() < alpha: # Random walk với xác suất alpha
                new[v] = random.randint(0, self.max_colors - 1)

        self.population[i] = FA_GraphColoringSolution(self.adj_list, self.num_vertices, self.max_colors, new)


    def _elite_local_search(self):
        elite_size = max(1, self.n_pop // 5)
        # Sắp xếp theo fitness giảm dần (tốt nhất trước)
        elite = sorted(self.population, key=lambda s: s.fitness(), reverse=True)[:elite_size]
        for sol in elite:
            if random.random() < self.local_search_prob:
                self._repair_solution(sol)

    def _repair_solution(self, sol):
        for _ in range(self.local_search_intensity):
            conflicts = [v for v in range(self.num_vertices) if
                         any(sol.coloring[v] == sol.coloring[nb] for nb in self.adj_list[v])]
            if not conflicts:
                break

            v = random.choice(conflicts)
            
            # Đếm số conflict của mỗi màu với các đỉnh kề
            color_conf = {c: sum(sol.coloring[nb] == c for nb in self.adj_list[v]) for c in
                          range(self.max_colors)}

            sol.coloring[v] = min(color_conf, key=color_conf.get)

    def _get_state(self):
        return {
            "colors": self.best_solution.coloring,
            "fitness_tuple": (self.best_fitness, self.best_solution.count_colors(), self.best_solution.count_conflicts()),
            "history": self.history
        }