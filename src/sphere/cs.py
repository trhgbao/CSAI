import numpy as np
import time
import math
import matplotlib.pyplot as plt


class CuckooSearchHybrid:
    def __init__(self, pop_size, max_gen, p_abandon, levy_beta,
                 initial_alpha, diversity_rate, search_range):
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.p_abandon = p_abandon
        self.levy_beta = levy_beta
        self.initial_alpha = initial_alpha
        self.diversity_rate = diversity_rate
        self.low, self.high = search_range

        # --- THÊM MỚI: Khởi tạo các thuộc tính để lưu lịch sử ---
        self.history = None
        self.dimensions = None

    @staticmethod
    def sphere(x):
        """
        TỐI ƯU HÓA: Thêm axis=-1 để hàm có thể tính fitness cho
        toàn bộ quần thể (ma trận) trong một lần gọi.
        """
        return np.sum(x ** 2, axis=-1)

    def get_levy_flight_step(self, beta, dims):
        # ... (phương thức này giữ nguyên)
        gamma_beta = math.gamma(1 + beta)
        sin_pi_beta = math.sin(math.pi * beta / 2)
        gamma_half = math.gamma((1 + beta) / 2)
        pow_two = 2 ** ((beta - 1) / 2)
        sigma_u = (gamma_beta * sin_pi_beta / (gamma_half * beta * pow_two)) ** (1 / beta)
        u = np.random.normal(0, sigma_u, size=dims)
        v = np.random.normal(0, 1, size=dims)
        step = u / (np.abs(v) ** (1 / beta))
        return step

    def run(self, dims):
        """
        SỬA ĐỔI: Phương thức này giờ đây sẽ ghi lại lịch sử hội tụ.
        """
        self.dimensions = dims 
        
        nests = np.random.uniform(self.low, self.high, size=(self.pop_size, dims))
        fitness = self.sphere(nests)

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_nest = nests[best_idx].copy()
        history_data = {'best': [], 'average': [], 'worst': []}

        for gen in range(self.max_gen):
            history_data['best'].append(best_fitness)
            history_data['average'].append(np.mean(fitness))
            history_data['worst'].append(np.max(fitness))

            current_alpha = self.initial_alpha * (1 - (gen + 1) / self.max_gen)

            # --- Lévy flight ---
            i = np.random.randint(0, self.pop_size)
            levy_step = self.get_levy_flight_step(self.levy_beta, dims)
            new_nest = nests[i] + current_alpha * levy_step
            new_nest = np.clip(new_nest, self.low, self.high)
            new_fitness = self.sphere(new_nest)

            j = np.random.randint(0, self.pop_size)
            if new_fitness < fitness[j]:
                nests[j], fitness[j] = new_nest, new_fitness

            # --- Abandon worst nests ---
            sorted_indices = np.argsort(fitness)
            num_abandon = int(self.p_abandon * self.pop_size)

            for idx in sorted_indices[-num_abandon:]:
                if np.random.rand() < self.diversity_rate:
                    new_nest_abandoned = np.random.uniform(self.low, self.high, size=dims)
                else:
                    j_rand, k_rand = np.random.choice(self.pop_size, 2, replace=False)
                    random_walk = np.random.rand() * (nests[j_rand] - nests[k_rand])
                    new_nest_abandoned = best_nest + random_walk

                new_nest_abandoned = np.clip(new_nest_abandoned, self.low, self.high)
                nests[idx] = new_nest_abandoned
                fitness[idx] = self.sphere(new_nest_abandoned)

            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_fitness = fitness[current_best_idx]
                best_nest = nests[current_best_idx].copy()
        
        self.history = history_data

        return best_nest, best_fitness

    def visualize(self, img_path):
        """
        HÀM MỚI: Vẽ biểu đồ hội tụ từ lịch sử đã được ghi lại.
        """
        if not self.history:
            print("Chưa có lịch sử để vẽ. Vui lòng chạy thuật toán bằng phương thức .run() trước.")
            return

        plt.figure(figsize=(12, 7))
        
        plt.plot(self.history['best'], color='blue', linestyle='-', label='Best')
        plt.plot(self.history['average'], color='green', linestyle='--', label='Average')
        plt.plot(self.history['worst'], color='red', linestyle=':', label='Worst')
        
        plt.yscale('log')
        plt.title(f"Cuckoo Search Convergence Curve - Dim={self.dimensions}", fontsize=16)
        plt.xlabel("Iteration (Generation)", fontsize=12)
        plt.ylabel("Fitness f(x)", fontsize=12)
        plt.grid(True, which="both", ls="--", linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(img_path, dpi=300)
        print(f"Đã lưu biểu đồ vào: {img_path}")
        plt.show()
