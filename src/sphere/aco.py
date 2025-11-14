import numpy as np
import matplotlib.pyplot as plt
import random
import time

def sphere_function(x):
    return np.sum(x ** 2)

class ContinuousACO:
    def __init__(self, dim, bounds, func=sphere_function, n_ants=20, n_archive=30,
                 q=0.1, xi=0.85,  # Các tham số mới thay cho alpha và evaporation_rate
                 max_iter=100, use_roulette=True, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        self.func = func
        self.dim = dim
        self.bounds = np.array(bounds)
        self.n_ants = n_ants
        self.n_archive = n_archive
        self.q = q  # Tương đương với độ lệch chuẩn ban đầu
        self.xi = xi  # Tương đương với tốc độ bay hơi
        self.max_iter = max_iter
        self.use_roulette = use_roulette

        # Archive lưu các nghiệm, được sắp xếp theo fitness
        self.archive = np.random.uniform(bounds[0], bounds[1], (n_archive, dim))
        self.archive_fitness = np.array([func(x) for x in self.archive])
        
        # Sắp xếp archive ban đầu
        sorted_indices = np.argsort(self.archive_fitness)
        self.archive = self.archive[sorted_indices]
        self.archive_fitness = self.archive_fitness[sorted_indices]
        self.history = []

    def _calculate_weights(self):
        """Tính trọng số cho các nghiệm trong archive dựa trên thứ hạng."""
        # Trọng số được tính bằng hàm Gaussian dựa trên thứ hạng (rank)
        ranks = np.arange(1, self.n_archive + 1)
        weights = (1 / (self.q * self.n_archive * np.sqrt(2 * np.pi))) * \
                  np.exp(-((ranks - 1)**2) / (2 * (self.q**2) * (self.n_archive**2)))
        return weights / np.sum(weights)

    def optimize(self, verbose=False):
        t0 = time.time()
        self.history = []
        global_best_f = self.archive_fitness[0]

        for it in range(self.max_iter):
            # 1. Tính trọng số cho các nghiệm trong archive
            weights = self._calculate_weights()

            # 2. Chọn nghiệm từ archive dựa trên trọng số (Roulette Wheel)
            if self.use_roulette:
                selected_indices = np.random.choice(self.n_archive, size=self.n_ants, p=weights)
            else:
                selected_indices = np.random.randint(0, self.n_archive, size=self.n_ants)
            # 3. Tạo ra các nghiệm mới
            new_ants = np.zeros((self.n_ants, self.dim))
            for i in range(self.n_ants):
                # Chọn một nghiệm làm trung tâm
                mu = self.archive[selected_indices[i]]
                
                # Tính độ lệch chuẩn cho việc lấy mẫu
                # Dựa trên khoảng cách trung bình từ nghiệm được chọn đến các nghiệm khác trong archive
                sigma_sum = 0
                for j in range(self.n_archive):
                    sigma_sum += np.abs(self.archive[j] - mu)
                sigma = self.xi * (sigma_sum / (self.n_archive - 1))
                
                # Tạo nghiệm mới bằng cách lấy mẫu Gaussian
                new_ants[i] = np.random.normal(loc=mu, scale=sigma)
            
            # Giữ các nghiệm trong biên
            new_ants = np.clip(new_ants, self.bounds[0], self.bounds[1])

            # 4. Đánh giá và cập nhật archive
            new_fitness = np.array([self.func(ant) for ant in new_ants])

            combined_solutions = np.vstack([self.archive, new_ants])
            combined_fitness = np.hstack([self.archive_fitness, new_fitness])

            # Sắp xếp và giữ lại các nghiệm tốt nhất
            sorted_idx = np.argsort(combined_fitness)
            self.archive = combined_solutions[sorted_idx][:self.n_archive]
            self.archive_fitness = combined_fitness[sorted_idx][:self.n_archive]

            # Cập nhật nghiệm tốt nhất toàn cục
            if self.archive_fitness[0] < global_best_f:
                global_best_f = self.archive_fitness[0]
            
            self.history.append(global_best_f)
            if verbose:
                print(f"Iter {it+1}/{self.max_iter} | Best = {global_best_f:.6f}")
        
        t1 = time.time()
        self.elapsed_time = t1 - t0
        return self.archive[0], self.archive_fitness[0]
    
    def visualize(self, img_path): 
        best_x = self.archive[0].copy()
        best_f = self.archive_fitness[0]
        # Vẽ đường hội tụ
        plt.figure(figsize=(8, 10))
        plt.plot(self.history)
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness')
        plt.title(f"Dim: {self.dim} | Best Fitness: {best_f:.4e} | Time: {self.elapsed_time:.4}s")
        # plt.legend()
        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.grid(True)
        # plt.savefig(f"caco_{dim}.png", dpi=300)
        plt.savefig(img_path, dpi=300)
        plt.show()

