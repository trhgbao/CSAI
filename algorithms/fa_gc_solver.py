import numpy as np
import random

def calculate_graph_fitness(solution, adj_list, penalty_weight):
    # ... (Hàm này không đổi)
    num_colors = len(np.unique(solution))
    num_conflicts = 0
    for u, neighbors in enumerate(adj_list):
        for v in neighbors:
            if u < v and solution[u] == solution[v]:
                num_conflicts += 1
    return num_colors + penalty_weight * num_conflicts, num_colors, num_conflicts

class FireflyAlgorithmGraphColoring:
    def __init__(self, graph, n_pop, n_iter, penalty_weight,
                 alpha0, alpha_min, alpha_decay, beta_max, beta_min, gamma,
                 p_local, i_local):
        # --- THAM SỐ ĐẦU VÀO ---
        self.graph = graph
        self.num_vertices = graph.number_of_nodes()
        self.adj_list = [list(graph.neighbors(i)) for i in range(self.num_vertices)]
        self.n_pop = n_pop
        self.n_iter = n_iter # Cần biết tổng số lần lặp để tính beta
        self.penalty_weight = penalty_weight
        
        # Tham số Alpha động
        self.alpha0, self.alpha_min, self.alpha_decay = alpha0, alpha_min, alpha_decay
        # Tham số Beta động
        self.beta_max, self.beta_min = beta_max, beta_min
        # Tham số FA kinh điển
        self.gamma = gamma
        # Tham số Local Search
        self.p_local, self.i_local = p_local, i_local

        # --- BIẾN TRẠNG THÁI ---
        self.current_iteration = 0 # Thêm biến đếm lần lặp
        self.current_alpha = self.alpha0
        self.positions = np.random.randint(0, self.num_vertices, (self.n_pop, self.num_vertices))
        self.intensities_tuples = [calculate_graph_fitness(p, self.adj_list, self.penalty_weight) for p in self.positions]
        self.intensities = np.array([t[0] for t in self.intensities_tuples])

        best_idx = np.argmin(self.intensities)
        self.gbest_pos = self.positions[best_idx].copy()
        self.gbest_val_tuple = self.intensities_tuples[best_idx]
        self.history = [self.gbest_val_tuple[0]]

    # --- Các hàm _hamming_distance, _move_firefly, _random_move, _perform_local_search không đổi ---
    def _hamming_distance(self, sol1, sol2):
        return np.sum(sol1 != sol2)

    def _move_firefly(self, firefly_i, firefly_j, beta):
        moved_firefly = firefly_i.copy()
        diff_indices = np.where(firefly_i != firefly_j)[0]
        if len(diff_indices) == 0: return moved_firefly
        num_to_move = int(np.ceil(beta * len(diff_indices)))
        if num_to_move > 0:
            indices_to_change = np.random.choice(diff_indices, size=num_to_move, replace=False)
            moved_firefly[indices_to_change] = firefly_j[indices_to_change]
        return moved_firefly

    def _random_move(self, firefly):
        mutated_firefly = firefly.copy()
        num_to_mutate = int(np.ceil(self.current_alpha * self.num_vertices))
        if num_to_mutate > 0:
            indices_to_mutate = np.random.choice(self.num_vertices, size=num_to_mutate, replace=False)
            max_color = len(np.unique(mutated_firefly)) + 1
            new_colors = np.random.randint(0, max_color, size=num_to_mutate)
            mutated_firefly[indices_to_mutate] = new_colors
        return mutated_firefly

    def _perform_local_search(self, starting_point):
        current_solution = starting_point.copy()
        current_fitness_tuple = calculate_graph_fitness(current_solution, self.adj_list, self.penalty_weight)
        for _ in range(self.i_local):
            conflicted_vertices = [u for u in range(self.num_vertices) if any(current_solution[u] == current_solution[v] for v in self.adj_list[u])]
            if not conflicted_vertices: break
            vertex_to_change = random.choice(conflicted_vertices)
            neighbor = current_solution.copy()
            max_color = len(np.unique(neighbor)) + 1
            neighbor[vertex_to_change] = random.randint(0, max_color)
            neighbor_fitness_tuple = calculate_graph_fitness(neighbor, self.adj_list, self.penalty_weight)
            if neighbor_fitness_tuple[0] < current_fitness_tuple[0]:
                current_solution = neighbor
                current_fitness_tuple = neighbor_fitness_tuple
        return current_solution, current_fitness_tuple
    
    def step(self):
        # --- CẬP NHẬT BETA ĐỘNG CHO LẦN LẶP NÀY ---
        # Beta giảm tuyến tính từ beta_max xuống beta_min
        current_beta = self.beta_max - (self.beta_max - self.beta_min) * (self.current_iteration / self.n_iter)
        
        # 1. Giai đoạn di chuyển toàn cục
        for i in range(self.n_pop):
            for j in range(self.n_pop):
                if self.intensities[j] < self.intensities[i]:
                    distance = self._hamming_distance(self.positions[i], self.positions[j])
                    # SỬ DỤNG current_beta thay vì self.beta0
                    beta_attractiveness = current_beta * np.exp(-self.gamma * (distance**2))
                    
                    new_pos = self._move_firefly(self.positions[i], self.positions[j], beta_attractiveness)
                    new_pos = self._random_move(new_pos)
                    
                    new_intensity_tuple = calculate_graph_fitness(new_pos, self.adj_list, self.penalty_weight)
                    if new_intensity_tuple[0] < self.intensities[i]:
                        self.positions[i] = new_pos
                        self.intensities_tuples[i] = new_intensity_tuple
                        self.intensities[i] = new_intensity_tuple[0]

        # 2. Các giai đoạn cập nhật và local search không đổi...
        best_idx_after_global = np.argmin(self.intensities)
        if self.intensities[best_idx_after_global] < self.gbest_val_tuple[0]:
            self.gbest_val_tuple = self.intensities_tuples[best_idx_after_global]
            self.gbest_pos = self.positions[best_idx_after_global].copy()

        if random.random() < self.p_local:
            refined_solution, refined_fitness_tuple = self._perform_local_search(self.gbest_pos)
            if refined_fitness_tuple[0] < self.gbest_val_tuple[0]:
                self.gbest_val_tuple = refined_fitness_tuple
                self.gbest_pos = refined_solution
                self.positions[best_idx_after_global] = refined_solution
                self.intensities_tuples[best_idx_after_global] = refined_fitness_tuple
                self.intensities[best_idx_after_global] = refined_fitness_tuple[0]

        # 3. Cập nhật Alpha VÀ TĂNG BIẾN ĐẾM LẦN LẶP
        new_alpha = self.current_alpha * self.alpha_decay
        self.current_alpha = max(new_alpha, self.alpha_min)
        self.current_iteration += 1
            
        self.history.append(self.gbest_val_tuple[0])
        return self._get_state()

    def _get_state(self):
        return {
            "colors": self.gbest_pos,
            "fitness_tuple": self.gbest_val_tuple,
            "history": self.history
        }