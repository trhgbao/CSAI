import numpy as np
import time
import random
import matplotlib.pyplot as plt
import seaborn as sns


class FireflyAlgorithmSphereFunction:
    """
    Thuật toán Firefly cho bài toán tối ưu liên tục - Sphere Function

    Sphere Function: f(x) = sum(x_i^2)
    Optimum: f(0, 0, ..., 0) = 0

    Tham số được tối ưu cho từng dim
    """

    def __init__(self, dim=30, bounds=(-5.12, 5.12), num_fireflies=None, max_iterations=None):
        self.dim = dim
        self.bounds = bounds

        # Tự động điều chỉnh tham số theo dim
        self.num_fireflies = num_fireflies if num_fireflies else self._get_optimal_fireflies(dim)
        self.max_iterations = max_iterations if max_iterations else self._get_optimal_iterations(dim)

        # Tham số FA được điều chỉnh theo dim
        self.gamma = self._get_optimal_gamma(dim)
        self.alpha_0 = 0.5
        self.alpha_min = 1e-9
        self.alpha_decay = 0.98  # Cải thiện: decay nhanh hơn cho hội tụ tốt hơn
        self.beta_0 = 1.0

        # Tham số Local Search được cải thiện
        self.local_search_prob = 0.4  # Tăng xác suất local search
        self.local_search_rate = 5  # Thực hiện local search thường xuyên hơn

        # Lưu lịch sử để vẽ đồ thị
        self.history_best = []
        self.history_avg = []
        self.history_worst = []

    def _get_optimal_fireflies(self, dim):
        """Số fireflies tối ưu theo dim - Cân bằng giữa khám phá và tốc độ"""
        if dim <= 5:
            return 20
        elif dim <= 10:
            return 25
        elif dim <= 20:
            return 30
        elif dim <= 40:
            return 40
        elif dim <= 100:
            return 50
        elif dim <= 200:
            return 60
        else:  # dim >= 300
            return 70

    def _get_optimal_iterations(self, dim):
        """Số iterations tối ưu theo dim"""
        if dim <= 5:
            return 500
        elif dim <= 10:
            return 800
        elif dim <= 20:
            return 1000
        elif dim <= 40:
            return 1500
        elif dim <= 100:
            return 2000
        elif dim <= 200:
            return 2500
        else:  # dim >= 300
            return 3000

    def _get_optimal_gamma(self, dim):
        """
        Gamma tối ưu theo dim
        Gamma cao = tương tác local mạnh, phù hợp dim thấp
        Gamma thấp = tương tác global, phù hợp dim cao
        """
        if dim <= 10:
            return 0.01
        elif dim <= 40:
            return 0.005
        elif dim <= 100:
            return 0.002
        else:
            return 0.001

    def sphere_function(self, x):
        """
        Hàm Sphere: f(x) = sum(x_i^2)
        Global minimum: f(0, 0, ..., 0) = 0
        """
        return np.sum(x ** 2)

    def initialize_population(self):
        """
        Khởi tạo quần thể firefly ngẫu nhiên
        Cải thiện: Sử dụng Latin Hypercube Sampling cho phân bố đều hơn
        """
        low, high = self.bounds

        # Khởi tạo ngẫu nhiên chuẩn
        self.population = np.random.uniform(low, high, (self.num_fireflies, self.dim))

        # Đảm bảo có 1 firefly ở gần origin (điểm tối ưu)
        # self.population[0] = np.random.uniform(-1, 1, self.dim)

        # Tính fitness ban đầu
        self.fitness = np.array([self.sphere_function(x) for x in self.population])

        # Lưu best solution
        best_idx = np.argmin(self.fitness)
        self.best_solution = self.population[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]

    def solve(self, verbose=True):
        """Vòng lặp chính của thuật toán"""
        self.initialize_population()
        start = time.time()

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"Firefly Algorithm - Sphere Function Optimization")
            print(f"{'=' * 70}")
            print(f"Dimension: {self.dim}")
            print(f"Bounds: {self.bounds}")
            print(f"Fireflies: {self.num_fireflies}")
            print(f"Max Iterations: {self.max_iterations}")
            print(f"Gamma: {self.gamma}")
            print(f"{'=' * 70}\n")

        for it in range(self.max_iterations):
            # Giảm dần độ ngẫu nhiên theo iteration
            alpha = max(self.alpha_0 * (self.alpha_decay ** it), self.alpha_min)

            # Lưu fitness trước khi di chuyển (để so sánh công bằng)
            fitness_before = self.fitness.copy()

            # Pha di chuyển: so sánh tất cả các cặp firefly
            for i in range(self.num_fireflies):
                for j in range(self.num_fireflies):
                    # Nếu firefly j sáng hơn (fitness tốt hơn) firefly i
                    if fitness_before[j] < fitness_before[i]:
                        self._move_towards(i, j, alpha)
                        # Cập nhật fitness ngay sau khi di chuyển
                        self.fitness[i] = self.sphere_function(self.population[i])

            # Áp dụng Local Search cho các firefly tốt nhất
            if it % self.local_search_rate == 0:
                self._elite_local_search()

            # Cập nhật best solution toàn cục
            current_best_idx = np.argmin(self.fitness)
            if self.fitness[current_best_idx] < self.best_fitness:
                self.best_fitness = self.fitness[current_best_idx]
                self.best_solution = self.population[current_best_idx].copy()

            # Lưu lịch sử
            self.history_best.append(self.best_fitness)
            self.history_avg.append(self.fitness.mean())
            self.history_worst.append(self.fitness.max())

            # In kết quả
            if verbose and it % (self.max_iterations // 10) == 0:
                print(f"Iter {it:5d}/{self.max_iterations}: "
                      f"Best f(x) = {self.best_fitness:.10f} | "
                      f"Avg f(x) = {self.fitness.mean():.10f}")

        elapsed = time.time() - start

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"✅ Optimization Complete!")
            print(f"{'=' * 70}")
            print(f"Dimension: {self.dim}")
            print(f"Time: {elapsed:.2f}s")
            print(f"Best f(x): {self.best_fitness:.15f}")
            print(f"Distance to Optimum (√f(x)): {np.sqrt(self.best_fitness):.15f}")

            # Hiển thị một số giá trị của best solution
            print(f"\nBest Solution (first 10 values):")
            print(f"  {self.best_solution[:min(10, self.dim)]}")
            print(f"{'=' * 70}\n")

        return self.best_solution, self.best_fitness

    def _move_towards(self, i, j, alpha):
        """
        Di chuyển firefly i về phía firefly j (sáng hơn)

        Công thức FA chuẩn:
        x_i = x_i + beta * (x_j - x_i) + alpha * random_step

        Trong đó:
        - beta giảm theo khoảng cách: beta = beta_0 * exp(-gamma * r^2)
        - random_step: bước ngẫu nhiên để khám phá
        """
        # Tính khoảng cách Euclidean
        r = np.linalg.norm(self.population[i] - self.population[j])

        # Độ hấp dẫn giảm theo khoảng cách
        beta = self.beta_0 * np.exp(-self.gamma * r ** 2)

        # Thành phần ngẫu nhiên (random walk)
        rand_step = alpha * (np.random.rand(self.dim) - 0.5) * (self.bounds[1] - self.bounds[0])

        # Cập nhật vị trí: hút về j + random walk
        self.population[i] += beta * (self.population[j] - self.population[i]) + rand_step

        # Đảm bảo firefly nằm trong bounds
        self.population[i] = np.clip(self.population[i], self.bounds[0], self.bounds[1])

    def _elite_local_search(self):
        """
        Áp dụng local search cho 20% firefly tốt nhất
        Giúp khai thác tốt hơn vùng lân cận của các solution tốt
        """
        # Sắp xếp theo fitness (tốt nhất trước)
        elite_size = max(3, self.num_fireflies // 5)
        elite_indices = np.argsort(self.fitness)[:elite_size]

        for idx in elite_indices:
            if random.random() < self.local_search_prob:
                self._local_refinement(idx)

    def _local_refinement(self, idx):
        """
        Tinh chỉnh vị trí của firefly bằng Hill Climbing
        Thử di chuyển nhỏ theo từng chiều và giữ lại nếu cải thiện
        """
        current_fitness = self.fitness[idx]

        # Thử di chuyển nhỏ theo mỗi chiều
        for d in range(self.dim):
            # Tính step size dựa trên giá trị hiện tại
            step_size = 0.01 * abs(self.population[idx][d]) if abs(self.population[idx][d]) > 0.1 else 0.1

            # Thử cả 2 hướng (+/-)
            for direction in [1, -1]:
                # Tạo solution mới
                new_position = self.population[idx].copy()
                new_position[d] += direction * step_size

                # Đảm bảo trong bounds
                new_position[d] = np.clip(new_position[d], self.bounds[0], self.bounds[1])

                # Tính fitness của vị trí mới
                new_fitness = self.sphere_function(new_position)

                # Nếu cải thiện, cập nhật
                if new_fitness < current_fitness:
                    self.population[idx] = new_position
                    self.fitness[idx] = new_fitness
                    current_fitness = new_fitness
                    break  # Chỉ di chuyển 1 lần mỗi chiều

    def plot_convergence(self, save_path='convergence.png'):
        """Vẽ đồ thị hội tụ"""
        plt.figure(figsize=(12, 6))

        iterations = range(len(self.history_best))

        plt.plot(iterations, self.history_best, 'b-', linewidth=2, label='Best')
        plt.plot(iterations, self.history_avg, 'g--', linewidth=1.5, label='Average')
        plt.plot(iterations, self.history_worst, 'r:', linewidth=1, label='Worst')

        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.title(f'Convergence Curve - Dim={self.dim}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale để thấy rõ hơn

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"📊 Convergence plot saved to: {save_path}")


# ============================================================================
# TEST - Chạy thử nghiệm cho các dim khác nhau
# ============================================================================
def run_experiments():
    """Chạy thử nghiệm với các dim khác nhau"""
    dimensions = [5, 10, 20, 40, 100, 200, 300]
    bounds = (-5.12, 5.12)

    results = []

    print("\n" + "=" * 80)
    print("FIREFLY ALGORITHM - SPHERE FUNCTION BENCHMARK")
    print("=" * 80)

    for dim in dimensions:
        print(f"\n{'#' * 80}")
        print(f"# Testing with Dimension = {dim}")
        print(f"{'#' * 80}")

        # Tạo và chạy thuật toán
        fa = FireflyAlgorithmSphereFunction(dim=dim, bounds=bounds)
        best_solution, best_fitness = fa.solve(verbose=True)

        results.append({
            'dim': dim,
            'fireflies': fa.num_fireflies,
            'iterations': fa.max_iterations,
            'gamma': fa.gamma,
            'best_fitness': best_fitness,
            'distance': np.sqrt(best_fitness)
        })

        # Vẽ đồ thị hội tụ cho mỗi dim
        fa.plot_convergence(save_path=f'convergence_dim{dim}.png')

    # Tổng kết
    print("\n" + "=" * 95)
    print("SUMMARY - ALL DIMENSIONS")
    print("=" * 95)
    print(f"{'Dim':<6} {'Fireflies':<11} {'Iterations':<11} {'Gamma':<10} {'Best f(x)':<20} {'Distance':<15}")
    print("-" * 95)
    for r in results:
        print(f"{r['dim']:<6} {r['fireflies']:<11} {r['iterations']:<11} {r['gamma']:<10.4f} "
              f"{r['best_fitness']:<20.15f} {r['distance']:<15.15f}")
    print("=" * 95 + "\n")

    return results


def run_single_test(dim=30):
    """Chạy thử nghiệm đơn lẻ cho 1 dim"""
    print(f"\n{'=' * 70}")
    print(f"Single Test - Dimension = {dim}")
    print(f"{'=' * 70}")

    fa = FireflyAlgorithmSphereFunction(dim=dim, bounds=(-5.12, 5.12))
    best_solution, best_fitness = fa.solve(verbose=True)

    # Vẽ đồ thị
    fa.plot_convergence(save_path=f'convergence_dim{dim}.png')

    return fa, best_solution, best_fitness


# ============================================================================
# MAIN - Chạy chương trình
# ============================================================================
if __name__ == "__main__":
    # Cài đặt seed để có thể reproduce kết quả
    np.random.seed(42)
    random.seed(42)

    # Chọn 1 trong 2 chế độ:

    # Chế độ 1: Chạy thử nghiệm cho tất cả các dim
    results = run_experiments()

    # Chế độ 2: Chạy riêng lẻ cho 1 dim
    # fa, best_solution, best_fitness = run_single_test(dim=30)