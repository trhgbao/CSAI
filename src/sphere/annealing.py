import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import math
import random

sns.set_style("whitegrid")


# ============================================================================
# SIMULATED ANNEALING FOR SPHERE FUNCTION
# ============================================================================
class SimulatedAnnealingSphereFunction:
    """
    Thuật toán Simulated Annealing cho bài toán tối ưu liên tục - Sphere Function

    Sphere Function: f(x) = sum(x_i^2)
    Optimum: f(0, 0, ..., 0) = 0

    Tham số:
    - dim: số chiều của không gian
    - bounds: giới hạn không gian tìm kiếm [min, max]
    - T0: nhiệt độ ban đầu
    - T_min: nhiệt độ tối thiểu
    - alpha: hệ số làm lạnh (cooling rate)
    - max_iterations: số iteration tối đa ở mỗi nhiệt độ
    - step_size: độ lớn bước di chuyển
    """

    def __init__(self, dim=30, bounds=(-5.12, 5.12), T0=100, T_min=0.01,
                 alpha=0.95, max_iterations=100, step_size=1.0):
        self.dim = dim
        self.bounds = bounds
        self.T0 = T0
        self.T_min = T_min
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.step_size = step_size

        # Lưu lịch sử
        self.history_best = []
        self.history_current = []
        self.history_temperature = []

    def sphere_function(self, x):
        """
        Hàm Sphere: f(x) = sum(x_i^2)
        Global minimum: f(0, 0, ..., 0) = 0
        """
        return np.sum(x ** 2)

    def solve(self):
        """
        Vòng lặp chính của thuật toán Simulated Annealing
        """
        start = time.time()

        # Khởi tạo solution ngẫu nhiên
        low, high = self.bounds
        current = np.random.uniform(low, high, self.dim)
        best = current.copy()

        current_energy = self.sphere_function(current)
        best_energy = current_energy

        T = self.T0
        iteration_count = 0

        print(f"\n{'=' * 70}")
        print(f"Simulated Annealing - Sphere Function Optimization")
        print(f"{'=' * 70}")
        print(f"Dimension: {self.dim}")
        print(f"Bounds: {self.bounds}")
        print(f"Initial Temperature (T0): {self.T0}")
        print(f"Final Temperature (T_min): {self.T_min}")
        print(f"Cooling Rate (alpha): {self.alpha}")
        print(f"Iterations per Temperature: {self.max_iterations}")
        print(f"Step Size: {self.step_size}")
        print(f"Initial f(x): {current_energy:.10f}")
        print(f"{'=' * 70}\n")

        # Vòng lặp chính
        while T > self.T_min:
            for _ in range(self.max_iterations):
                iteration_count += 1

                # Tạo solution láng giềng
                neighbor = self._get_neighbor(current, T)
                neighbor_energy = self.sphere_function(neighbor)

                # Tính delta năng lượng
                delta_E = neighbor_energy - current_energy

                # Chấp nhận solution mới nếu:
                # 1. Tốt hơn (delta_E < 0)
                # 2. Hoặc theo xác suất Boltzmann
                if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                    current = neighbor
                    current_energy = neighbor_energy

                    # Cập nhật best solution
                    if current_energy < best_energy:
                        best = current.copy()
                        best_energy = current_energy

                # Lưu lịch sử
                self.history_best.append(best_energy)
                self.history_current.append(current_energy)
                self.history_temperature.append(T)

                # In kết quả
                if iteration_count % 1000 == 0:
                    print(f"Iter {iteration_count:6d}: T={T:8.4f}, "
                          f"Best f(x)={best_energy:.10f}, "
                          f"Current f(x)={current_energy:.10f}")

            # Làm lạnh (cooling)
            T *= self.alpha

        elapsed = time.time() - start

        print(f"\n{'=' * 70}")
        print(f"✅ Optimization Complete!")
        print(f"{'=' * 70}")
        print(f"Total Iterations: {iteration_count}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Final Temperature: {T:.6f}")
        print(f"Best f(x): {best_energy:.15f}")
        print(f"Distance to Optimum (√f(x)): {np.sqrt(best_energy):.15f}")
        print(f"\nBest Solution (first 10 values):")
        print(f"  {best[:min(10, self.dim)]}")
        print(f"{'=' * 70}\n")

        return best, best_energy

    def _get_neighbor(self, current, T):
        """
        Tạo solution láng giềng bằng cách thêm nhiễu Gaussian
        Độ lớn nhiễu giảm theo nhiệt độ (adaptive step size)
        """
        # Step size thích nghi: giảm khi nhiệt độ giảm
        adaptive_step = self.step_size * (T / self.T0)

        # Thêm nhiễu Gaussian
        neighbor = current + np.random.normal(0, adaptive_step, self.dim)

        # Đảm bảo neighbor nằm trong bounds
        neighbor = np.clip(neighbor, self.bounds[0], self.bounds[1])

        return neighbor

    def plot_convergence(self, save_path='convergence_sa_sphere.png'):
        """Vẽ đồ thị hội tụ"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        iterations = range(len(self.history_best))

        # Subplot 1: f(x) convergence
        ax1 = axes[0]
        ax1.plot(iterations, self.history_best, 'b-', linewidth=2, label='Best f(x)')
        ax1.plot(iterations, self.history_current, 'r:', linewidth=1, alpha=0.5, label='Current f(x)')
        ax1.set_xlabel('Iteration', fontsize=11)
        ax1.set_ylabel('f(x)', fontsize=11)
        ax1.set_title('Function Value Convergence', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # Subplot 2: Temperature
        ax2 = axes[1]
        ax2.plot(iterations, self.history_temperature, 'orange', linewidth=2)
        ax2.set_xlabel('Iteration', fontsize=11)
        ax2.set_ylabel('Temperature', fontsize=11)
        ax2.set_title('Temperature Cooling', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"📊 Convergence plot saved to: {save_path}")

