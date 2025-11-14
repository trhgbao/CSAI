import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

class Solution:
    """
    Biểu diễn một phương án tô màu đồ thị
    coloring[i] = màu của đỉnh i
    """

    def __init__(self, graph, coloring=None):
        self.graph = graph
        self.coloring = coloring or [random.randint(0, graph.max_colors - 1) for _ in range(graph.num_vertices)]

    def count_conflicts(self):
        """Đếm số cạnh vi phạm (2 đỉnh kề cùng màu)"""
        c = 0
        for v in range(self.graph.num_vertices):
            for nb in self.graph.adjacency[v]:
                if self.coloring[v] == self.coloring[nb]:
                    c += 1
        return c // 2

    def count_colors(self):
        """Đếm số màu được sử dụng"""
        return len(set(self.coloring))

    def fitness(self):
        """
        Hàm đánh giá chất lượng lời giải:
        - Nếu hợp lệ (0 conflict): fitness = 1000 - số màu (càng ít màu càng tốt)
        - Nếu không hợp lệ: phạt nặng theo số conflict
        """
        c = self.count_conflicts()
        k = self.count_colors()
        return 1000 - k if c == 0 else -10 * c - 0.1 * k

    def copy(self):
        """Tạo bản sao của solution"""
        return Solution(self.graph, self.coloring.copy())

    def is_valid(self):
        """Kiểm tra xem lời giải có hợp lệ không"""
        return self.count_conflicts() == 0

class DSATUR:
    """
    Thuật toán DSATUR (Degree of Saturation)
    Tô màu đồ thị theo thứ tự: đỉnh có saturation degree cao nhất

    Saturation degree: số màu khác nhau đã được sử dụng bởi các đỉnh kề
    """

    def __init__(self, graph):
        self.graph = graph

    def solve(self):
        """
        Chạy thuật toán DSATUR
        Trả về một solution hợp lệ (không có conflict)
        """
        n = self.graph.num_vertices
        coloring = [-1] * n  # -1 = chưa tô màu

        # Bắt đầu với đỉnh có bậc cao nhất
        degrees = {v: len(self.graph.adjacency[v]) for v in range(n)}
        first_vertex = max(degrees, key=degrees.get)
        coloring[first_vertex] = 0

        # Tô màu các đỉnh còn lại
        colored = {first_vertex}
        uncolored = set(range(n)) - colored

        while uncolored:
            # Tính saturation degree cho mỗi đỉnh chưa tô
            sat_degrees = {}
            for v in uncolored:
                # Tìm các màu của các đỉnh kề đã được tô
                neighbor_colors = {coloring[nb] for nb in self.graph.adjacency[v]
                                   if coloring[nb] != -1}
                sat_degrees[v] = len(neighbor_colors)

            # Chọn đỉnh có saturation degree cao nhất
            # Nếu bằng nhau, chọn đỉnh có degree cao nhất
            max_sat = max(sat_degrees.values())
            candidates = [v for v, sat in sat_degrees.items() if sat == max_sat]

            if len(candidates) > 1:
                # Tie-breaking: chọn đỉnh có degree cao nhất
                next_vertex = max(candidates, key=lambda v: degrees[v])
            else:
                next_vertex = candidates[0]

            # Tìm màu nhỏ nhất có thể dùng cho đỉnh này
            neighbor_colors = {coloring[nb] for nb in self.graph.adjacency[next_vertex]
                               if coloring[nb] != -1}

            color = 0
            while color in neighbor_colors:
                color += 1

            coloring[next_vertex] = color
            colored.add(next_vertex)
            uncolored.remove(next_vertex)

        return Solution(self.graph, coloring)


# ============================================================================
# FIREFLY ALGORITHM FOR GRAPH COLORING
# ============================================================================
class FireflyAlgorithmGraphColoring:
    """
    Thuật toán Firefly cho bài toán tô màu đồ thị

    Tham số:
    - num_fireflies: số lượng firefly trong quần thể
    - max_iterations: số vòng lặp tối đa
    - gamma: hệ số hấp thụ ánh sáng
    - alpha_0, alpha_min, alpha_decay: tham số điều khiển độ ngẫu nhiên
    - beta_min, beta_max: giới hạn độ hấp dẫn
    - local_search_prob: xác suất thực hiện local search
    - local_search_intensity: số bước local search
    - use_dsatur: sử dụng DSATUR để khởi tạo quần thể
    """

    def __init__(self, graph, num_fireflies=40, max_iterations=400, use_dsatur=False):
        self.graph = graph
        self.num_fireflies = num_fireflies
        self.max_iterations = max_iterations
        self.use_dsatur = use_dsatur

        # Tham số FA
        self.gamma = 0.8
        self.alpha_0 = 0.5
        self.alpha_min = 0.01
        self.alpha_decay = 0.97
        self.beta_min, self.beta_max = 0.2, 1.0

        # Tham số Local Search
        self.local_search_prob = 0.5
        self.local_search_intensity = 15

        # Lưu lịch sử để vẽ đồ thị
        self.history_best_fitness = []
        self.history_avg_fitness = []
        self.history_worst_fitness = []
        self.history_best_colors = []
        self.history_best_conflicts = []

    def solve(self):
        """Vòng lặp chính của thuật toán"""
        start = time.time()

        # Khởi tạo quần thể firefly
        if self.use_dsatur:
            print("🎯 Using DSATUR for initialization...")
            dsatur = DSATUR(self.graph)
            dsatur_solution = dsatur.solve()

            print(f"   DSATUR Result: {dsatur_solution.count_colors()} colors, "
                  f"{dsatur_solution.count_conflicts()} conflicts")

            # Khởi tạo quần thể: một phần từ DSATUR, phần còn lại random
            self.population = []

            # 30% quần thể: DSATUR solution với nhiễu nhỏ
            dsatur_ratio = max(1, self.num_fireflies // 3)
            for _ in range(dsatur_ratio):
                perturbed = dsatur_solution.copy()
                # Thêm nhiễu nhỏ: đổi màu ngẫu nhiên 5-10% đỉnh
                num_changes = random.randint(self.graph.num_vertices // 20,
                                             self.graph.num_vertices // 10)
                for _ in range(num_changes):
                    v = random.randint(0, self.graph.num_vertices - 1)
                    perturbed.coloring[v] = random.randint(0, self.graph.max_colors - 1)
                self.population.append(perturbed)

            # 70% còn lại: random
            for _ in range(self.num_fireflies - dsatur_ratio):
                self.population.append(Solution(self.graph))
        else:
            print("🎲 Using random initialization...")
            self.population = [Solution(self.graph) for _ in range(self.num_fireflies)]

        self.best_solution = max(self.population, key=lambda s: s.fitness()).copy()
        best_fit = self.best_solution.fitness()

        print(f"\n{'=' * 70}")
        print(f"Starting Firefly Algorithm for Graph Coloring")
        print(f"{'=' * 70}")
        print(f"Fireflies: {self.num_fireflies}")
        print(f"Max Iterations: {self.max_iterations}")
        print(f"DSATUR Init: {'Enabled' if self.use_dsatur else 'Disabled'}")
        print(f"Initial Best: {self.best_solution.count_colors()} colors, "
              f"{self.best_solution.count_conflicts()} conflicts")
        print(f"{'=' * 70}\n")

        for it in range(self.max_iterations):
            # Tính độ sáng (fitness) của tất cả firefly
            brightness = [s.fitness() for s in self.population]

            # Di chuyển firefly tối hơn về phía firefly sáng hơn
            for i in range(self.num_fireflies):
                for j in range(self.num_fireflies):
                    if brightness[j] > brightness[i]:
                        self._move_firefly(i, j, it)

            # Áp dụng local search cho các firefly tốt nhất
            if it % 10 == 0:
                self._elite_local_search()

            # Cập nhật best solution
            new_best = max(self.population, key=lambda s: s.fitness())
            if new_best.fitness() > best_fit:
                best_fit = new_best.fitness()
                self.best_solution = new_best.copy()

            # Lưu lịch sử
            fitness_values = [s.fitness() for s in self.population]
            self.history_best_fitness.append(max(fitness_values))
            self.history_avg_fitness.append(np.mean(fitness_values))
            self.history_worst_fitness.append(min(fitness_values))
            self.history_best_colors.append(self.best_solution.count_colors())
            self.history_best_conflicts.append(self.best_solution.count_conflicts())

            # In kết quả
            if it % 50 == 0:
                print(f"Iter {it:4d}: Fitness={best_fit:8.2f}, "
                      f"Colors={self.best_solution.count_colors():3d}, "
                      f"Conflicts={self.best_solution.count_conflicts():4d}")

        elapsed = time.time() - start

        print(f"\n{'=' * 70}")
        print(f"Optimization Complete!")
        print(f"{'=' * 70}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Best Fitness: {best_fit:.2f}")
        print(f"Colors Used: {self.best_solution.count_colors()}")
        print(f"Conflicts: {self.best_solution.count_conflicts()}")
        print(f"Valid Solution: {'Yes' if self.best_solution.is_valid() else 'No'}")
        print(f"{'=' * 70}\n")

        return self.best_solution

    def _move_firefly(self, i, j, it):
        """
        Di chuyển firefly i về phía firefly j (sáng hơn)

        Sử dụng:
        - Hamming distance để đo khoảng cách giữa 2 solution
        - Beta decay theo khoảng cách
        - Alpha decay theo iteration
        """
        sol_i, sol_j = self.population[i], self.population[j]

        # Tính khoảng cách Hamming (tỷ lệ đỉnh khác màu)
        d = sum(1 for a, b in zip(sol_i.coloring, sol_j.coloring) if a != b) / self.graph.num_vertices

        # Độ hấp dẫn giảm theo khoảng cách
        beta = self.beta_min + (self.beta_max - self.beta_min) * np.exp(-self.gamma * d ** 2)

        # Độ ngẫu nhiên giảm theo iteration
        alpha = max(self.alpha_0 * (self.alpha_decay ** it), self.alpha_min)

        # Tạo solution mới
        new = sol_i.coloring.copy()
        for v in range(self.graph.num_vertices):
            # Hút về phía firefly j với xác suất beta
            if random.random() < beta:
                new[v] = sol_j.coloring[v]
            # Random walk với xác suất alpha
            if random.random() < alpha:
                new[v] = random.randint(0, self.graph.max_colors - 1)

        self.population[i] = Solution(self.graph, new)

    def _elite_local_search(self):
        """
        Áp dụng local search cho 20% firefly tốt nhất
        Giúp khai thác tốt hơn vùng lân cận của solution tốt
        """
        elite = sorted(self.population, key=lambda s: s.fitness(), reverse=True)[:self.num_fireflies // 5]
        for sol in elite:
            if random.random() < self.local_search_prob:
                self._repair_solution(sol)

    def _repair_solution(self, sol):
        """
        Sửa chữa solution bằng cách đổi màu các đỉnh conflict
        về màu ít gây conflict nhất với các đỉnh kề
        """
        for _ in range(self.local_search_intensity):
            # Tìm các đỉnh bị conflict
            conflicts = [v for v in range(self.graph.num_vertices) if
                         any(sol.coloring[v] == sol.coloring[nb] for nb in self.graph.adjacency[v])]
            if not conflicts:
                break

            # Chọn một đỉnh conflict ngẫu nhiên
            v = random.choice(conflicts)

            # Đếm số conflict của mỗi màu với các đỉnh kề
            color_conf = {c: sum(sol.coloring[nb] == c for nb in self.graph.adjacency[v]) for c in
                          range(self.graph.max_colors)}

            # Đổi sang màu ít conflict nhất
            sol.coloring[v] = min(color_conf, key=color_conf.get)

    def plot_convergence(self, save_path='convergence_graph_coloring.png'):
        """Vẽ đồ thị hội tụ của thuật toán"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        iterations = range(len(self.history_best_fitness))

        # Subplot 1: Fitness convergence
        ax1 = axes[0, 0]
        ax1.plot(iterations, self.history_best_fitness, 'b-', linewidth=2, label='Best')
        ax1.plot(iterations, self.history_avg_fitness, 'g--', linewidth=1.5, label='Average')
        ax1.plot(iterations, self.history_worst_fitness, 'r:', linewidth=1, label='Worst')
        ax1.set_xlabel('Iteration', fontsize=11)
        ax1.set_ylabel('Fitness', fontsize=11)
        ax1.set_title('Fitness Convergence', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Subplot 2: Number of colors
        ax2 = axes[0, 1]
        ax2.plot(iterations, self.history_best_colors, 'purple', linewidth=2)
        ax2.set_xlabel('Iteration', fontsize=11)
        ax2.set_ylabel('Number of Colors', fontsize=11)
        ax2.set_title('Colors Used (Best Solution)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=0)

        # Subplot 3: Number of conflicts
        ax3 = axes[1, 0]
        ax3.plot(iterations, self.history_best_conflicts, 'orange', linewidth=2)
        ax3.set_xlabel('Iteration', fontsize=11)
        ax3.set_ylabel('Number of Conflicts', fontsize=11)
        ax3.set_title('Conflicts (Best Solution)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(bottom=0)

        # Subplot 4: Combined view
        ax4 = axes[1, 1]
        ax4_twin = ax4.twinx()

        line1 = ax4.plot(iterations, self.history_best_colors, 'purple', linewidth=2, label='Colors')
        line2 = ax4_twin.plot(iterations, self.history_best_conflicts, 'orange', linewidth=2, label='Conflicts')

        ax4.set_xlabel('Iteration', fontsize=11)
        ax4.set_ylabel('Number of Colors', fontsize=11, color='purple')
        ax4_twin.set_ylabel('Number of Conflicts', fontsize=11, color='orange')
        ax4.set_title('Colors vs Conflicts', fontsize=12, fontweight='bold')
        ax4.tick_params(axis='y', labelcolor='purple')
        ax4_twin.tick_params(axis='y', labelcolor='orange')
        ax4.grid(True, alpha=0.3)

        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, fontsize=9, loc='upper right')

        # Add DSATUR annotation if used
        if self.use_dsatur:
            fig.text(0.5, 0.02, 'Initialized with DSATUR', ha='center',
                     fontsize=10, style='italic', color='blue')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Convergence plot saved to: {save_path}")
