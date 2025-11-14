import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import math

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 10)

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

    def energy(self):
        """
        Hàm năng lượng cho SA (càng thấp càng tốt)
        E = số_conflicts * 1000 + số_màu
        """
        return self.count_conflicts() * 1000 + self.count_colors()

    def copy(self):
        """Tạo bản sao của solution"""
        return Solution(self.graph, self.coloring.copy())

    def is_valid(self):
        """Kiểm tra xem lời giải có hợp lệ không"""
        return self.count_conflicts() == 0


# ============================================================================
# SIMULATED ANNEALING FOR GRAPH COLORING
# ============================================================================
class SimulatedAnnealingGraphColoring:
    """
    Thuật toán Simulated Annealing cho bài toán tô màu đồ thị

    Tham số:
    - T0: nhiệt độ ban đầu
    - T_min: nhiệt độ tối thiểu
    - alpha: hệ số làm lạnh (cooling rate)
    - max_iterations: số iteration tối đa ở mỗi nhiệt độ
    """

    def __init__(self, graph, T0=1000, T_min=0.1, alpha=0.95, max_iterations=100):
        self.graph = graph
        self.T0 = T0
        self.T_min = T_min
        self.alpha = alpha
        self.max_iterations = max_iterations

        # Lưu lịch sử
        self.history_best_energy = []
        self.history_current_energy = []
        self.history_temperature = []
        self.history_best_colors = []
        self.history_best_conflicts = []

    def solve(self):
        """
        Vòng lặp chính của thuật toán Simulated Annealing
        """
        start = time.time()

        # Khởi tạo solution ngẫu nhiên
        current = Solution(self.graph)
        best = current.copy()

        T = self.T0
        iteration_count = 0

        print(f"\n{'=' * 70}")
        print(f"Simulated Annealing - Graph Coloring")
        print(f"{'=' * 70}")
        print(f"Initial Temperature (T0): {self.T0}")
        print(f"Final Temperature (T_min): {self.T_min}")
        print(f"Cooling Rate (alpha): {self.alpha}")
        print(f"Iterations per Temperature: {self.max_iterations}")
        print(f"Initial Energy: {current.energy()}")
        print(f"{'=' * 70}\n")

        # Vòng lặp chính
        while T > self.T_min:
            for _ in range(self.max_iterations):
                iteration_count += 1

                # Tạo solution láng giềng (neighbor)
                neighbor = self._get_neighbor(current)

                # Tính delta năng lượng
                delta_E = neighbor.energy() - current.energy()

                # Chấp nhận solution mới nếu:
                # 1. Tốt hơn (delta_E < 0)
                # 2. Hoặc theo xác suất Boltzmann
                if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                    current = neighbor

                    # Cập nhật best solution
                    if current.energy() < best.energy():
                        best = current.copy()

                # Lưu lịch sử
                self.history_best_energy.append(best.energy())
                self.history_current_energy.append(current.energy())
                self.history_temperature.append(T)
                self.history_best_colors.append(best.count_colors())
                self.history_best_conflicts.append(best.count_conflicts())

                # In kết quả mỗi 1000 iterations
                if iteration_count % 1000 == 0:
                    print(f"Iter {iteration_count:6d}: T={T:8.2f}, "
                          f"Best Energy={best.energy():6d}, "
                          f"Colors={best.count_colors():3d}, "
                          f"Conflicts={best.count_conflicts():4d}")

            # Làm lạnh (cooling)
            T *= self.alpha

        elapsed = time.time() - start

        print(f"\n{'=' * 70}")
        print(f"✅ Optimization Complete!")
        print(f"{'=' * 70}")
        print(f"Total Iterations: {iteration_count}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Final Temperature: {T:.4f}")
        print(f"Best Energy: {best.energy()}")
        print(f"Colors Used: {best.count_colors()}")
        print(f"Conflicts: {best.count_conflicts()}")
        print(f"Valid Solution: {'Yes' if best.is_valid() else 'No'}")
        print(f"{'=' * 70}\n")

        return best

    def _get_neighbor(self, solution):
        """
        Tạo solution láng giềng bằng cách đổi màu của 1 đỉnh ngẫu nhiên
        """
        neighbor = solution.copy()

        # Chọn ngẫu nhiên 1 đỉnh
        vertex = random.randint(0, self.graph.num_vertices - 1)

        # Đổi sang màu ngẫu nhiên khác
        new_color = random.randint(0, self.graph.max_colors - 1)
        neighbor.coloring[vertex] = new_color

        return neighbor

    def plot_convergence(self, save_path='convergence_sa_graph_coloring.png'):
        """Vẽ đồ thị hội tụ"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        iterations = range(len(self.history_best_energy))

        # Subplot 1: Energy convergence
        ax1 = axes[0, 0]
        ax1.plot(iterations, self.history_best_energy, 'b-', linewidth=2, label='Best Energy')
        ax1.plot(iterations, self.history_current_energy, 'r:', linewidth=1, alpha=0.5, label='Current Energy')
        ax1.set_xlabel('Iteration', fontsize=11)
        ax1.set_ylabel('Energy', fontsize=11)
        ax1.set_title('Energy Convergence', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # Subplot 2: Temperature
        ax2 = axes[0, 1]
        ax2.plot(iterations, self.history_temperature, 'orange', linewidth=2)
        ax2.set_xlabel('Iteration', fontsize=11)
        ax2.set_ylabel('Temperature', fontsize=11)
        ax2.set_title('Temperature Cooling', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')

        # Subplot 3: Number of colors
        ax3 = axes[1, 0]
        ax3.plot(iterations, self.history_best_colors, 'purple', linewidth=2)
        ax3.set_xlabel('Iteration', fontsize=11)
        ax3.set_ylabel('Number of Colors', fontsize=11)
        ax3.set_title('Colors Used (Best Solution)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(bottom=0)

        # Subplot 4: Number of conflicts
        ax4 = axes[1, 1]
        ax4.plot(iterations, self.history_best_conflicts, 'red', linewidth=2)
        ax4.set_xlabel('Iteration', fontsize=11)
        ax4.set_ylabel('Number of Conflicts', fontsize=11)
        ax4.set_title('Conflicts (Best Solution)', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(bottom=0)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"📊 Convergence plot saved to: {save_path}")