import numpy as np

class PSOGraphColoring:
    def __init__(self, graph, max_color, n_pop, n_iter, w, c1, c2, penalty_weight):
        # --- Ánh xạ từ cấu trúc của ứng dụng sang cấu trúc của thuật toán gốc ---
        self.graph_adj_list = [list(graph.neighbors(i)) for i in range(graph.number_of_nodes())]
        self.n = len(self.graph_adj_list)
        self.k = max_color
        self.swarm_size = n_pop
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.penalty_weight = penalty_weight

        # --- Logic khởi tạo từ PSO_Coloring_Real ---
        self.positions = np.random.uniform(0, self.k - 1, (self.swarm_size, self.n))
        self.velocities = np.random.uniform(-(self.k-1), (self.k-1), (self.swarm_size, self.n))
        
        self.personal_best = self.positions.copy()
        self.personal_best_fit = np.array([self._evaluate(p) for p in self.personal_best])

        best_idx = np.argmin(self.personal_best_fit)
        self.global_best = self.personal_best[best_idx].copy()
        self.global_best_fit = self.personal_best_fit[best_idx]
        
        # Biến cho giao diện
        self.history = [self.global_best_fit]

    # --- Các hàm cốt lõi từ PSO_Coloring_Real ---
    def _conflict_count(self, coloring):
        conflicts = 0
        for u in range(len(self.graph_adj_list)):
            for v in self.graph_adj_list[u]:
                if u < v and coloring[u] == coloring[v]:
                    conflicts += 1
        return conflicts

    def _real_to_color(self, x):
        return np.rint(x).astype(int)

    def _evaluate(self, x):
        c = self._real_to_color(x)
        c = np.clip(c, 0, self.k - 1) # Đảm bảo màu không vượt quá giới hạn
        conf = self._conflict_count(c)
        used_colors = len(np.unique(c))
        return self.penalty_weight * conf + used_colors

    # --- Hàm step() cho ứng dụng ---
    def step(self):
        # Chạy một vòng lặp của logic tối ưu hóa
        for i in range(self.swarm_size):
            r1 = np.random.rand(self.n)
            r2 = np.random.rand(self.n)
            
            self.velocities[i] = (self.w * self.velocities[i] +
                                  self.c1 * r1 * (self.personal_best[i] - self.positions[i]) +
                                  self.c2 * r2 * (self.global_best - self.positions[i]))
            
            self.positions[i] += self.velocities[i]
            # Giữ vị trí trong khoảng màu hợp lệ [0, k-1]
            self.positions[i] = np.clip(self.positions[i], 0, self.k - 1)
            
            fit = self._evaluate(self.positions[i])
            if fit < self.personal_best_fit[i]:
                self.personal_best_fit[i] = fit
                self.personal_best[i] = self.positions[i].copy()
                if fit < self.global_best_fit:
                    self.global_best_fit = fit
                    self.global_best = self.positions[i].copy()

        self.history.append(self.global_best_fit)
        return self._get_state()

    # --- Hàm trả về trạng thái cho giao diện ---
    def _get_state(self):
        best_coloring = self._real_to_color(self.global_best)
        best_coloring = np.clip(best_coloring, 0, self.k - 1)
        
        num_colors = len(np.unique(best_coloring))
        num_conflicts = self._conflict_count(best_coloring)
        
        return {
            "colors": best_coloring,
            "fitness_tuple": (self.global_best_fit, num_colors, num_conflicts),
            "history": self.history
        }