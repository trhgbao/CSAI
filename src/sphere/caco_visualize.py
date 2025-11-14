import numpy as np
import matplotlib.pyplot as plt
import random
import time

# =========================
#  Lớp Continuous ACO (Đã chỉnh sửa để dễ dàng thử nghiệm)
# =========================

class ContinuousACO:
    def __init__(self, func, dim, bounds, n_ants=20, n_archive=30,
                 q=0.1, xi=0.85, max_iter=100, seed=None):
        self.func = func
        self.dim = dim
        self.bounds = np.array(bounds)
        self.n_ants = n_ants
        self.n_archive = n_archive
        self.q = q
        self.xi = xi
        self.max_iter = max_iter
        self.seed = seed

        # Đặt seed để có thể tái tạo kết quả
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        # Archive lưu các nghiệm, được sắp xếp theo fitness
        self.archive = np.random.uniform(bounds[0], bounds[1], (n_archive, dim))
        self.archive_fitness = np.array([func(x) for x in self.archive])
        
        sorted_indices = np.argsort(self.archive_fitness)
        self.archive = self.archive[sorted_indices]
        self.archive_fitness = self.archive_fitness[sorted_indices]

    def _calculate_weights(self):
        ranks = np.arange(1, self.n_archive + 1)
        weights = (1 / (self.q * self.n_archive * np.sqrt(2 * np.pi))) * \
                  np.exp(-((ranks - 1)**2) / (2 * (self.q**2) * (self.n_archive**2)))
        return weights / np.sum(weights)

    def optimize(self, verbose=False):
        best_values = []
        global_best_f = self.archive_fitness[0]

        for it in range(self.max_iter):
            weights = self._calculate_weights()
            selected_indices = np.random.choice(self.n_archive, size=self.n_ants, p=weights)

            new_ants = np.zeros((self.n_ants, self.dim))
            for i in range(self.n_ants):
                mu = self.archive[selected_indices[i]]
                sigma_sum = np.sum(np.abs(self.archive - mu), axis=0) # Tối ưu hóa tính toán sigma
                sigma = self.xi * (sigma_sum / (self.n_archive - 1))
                
                new_ants[i] = np.random.normal(loc=mu, scale=sigma)
            
            new_ants = np.clip(new_ants, self.bounds[0], self.bounds[1])
            new_fitness = np.array([self.func(ant) for ant in new_ants])

            combined_solutions = np.vstack([self.archive, new_ants])
            combined_fitness = np.hstack([self.archive_fitness, new_fitness])

            sorted_idx = np.argsort(combined_fitness)
            self.archive = combined_solutions[sorted_idx][:self.n_archive]
            self.archive_fitness = combined_fitness[sorted_idx][:self.n_archive]

            if self.archive_fitness[0] < global_best_f:
                global_best_f = self.archive_fitness[0]
            
            best_values.append(global_best_f)
            if verbose and (it + 1) % 100 == 0:
                print(f"Iter {it+1}/{self.max_iter} | Best = {global_best_f:.6f}")

        return self.archive[0], self.archive_fitness[0], best_values

# =========================
#  Hàm mục tiêu và các thông số chung
# =========================
def sphere(x):
    return np.sum(x ** 2)

dim = 70
bounds = [-5.12, 5.12]
max_iter = 500 # Tăng số lần lặp để thấy rõ sự khác biệt
seed = 42

# Cấu hình gốc
baseline_params = {
    'n_ants': 30, 'n_archive': 30, 'q': 0.1, 'xi': 0.85
}

# =========================
#  Chạy Thử nghiệm
# =========================

results = {}

# 1. Thử nghiệm với tham số q
print("--- Tuning 'q' (Selection Pressure) ---")
q_values = [0.05, 0.1, 0.3]
results['q'] = {}
for q in q_values:
    params = baseline_params.copy()
    params['q'] = q
    aco = ContinuousACO(sphere, dim, bounds, **params, max_iter=max_iter, seed=seed)
    _, _, history = aco.optimize()
    results['q'][f"q={q}"] = history
    print(f"Finished q={q}, Final Fitness: {history[-1]:.4f}")

# 2. Thử nghiệm với n_archive
print("\n--- Tuning 'n_archive' (Archive Size) ---")
archive_values = [15, 30, 45]
results['archive'] = {}
for n_archive in archive_values:
    params = baseline_params.copy()
    params['n_archive'] = n_archive
    aco = ContinuousACO(sphere, dim, bounds, **params, max_iter=max_iter, seed=seed)
    _, _, history = aco.optimize()
    results['archive'][f"n_archive={n_archive}"] = history
    print(f"Finished n_archive={n_archive}, Final Fitness: {history[-1]:.4f}")

# 3. Thử nghiệm với xi
print("\n--- Tuning 'xi' (Exploration Factor) ---")
xi_values = [0.6, 0.85, 1.1]
results['xi'] = {}
for xi in xi_values:
    params = baseline_params.copy()
    params['xi'] = xi
    aco = ContinuousACO(sphere, dim, bounds, **params, max_iter=max_iter, seed=seed)
    _, _, history = aco.optimize()
    results['xi'][f"xi={xi}"] = history
    print(f"Finished xi={xi}, Final Fitness: {history[-1]:.4f}")

# 4. So sánh cấu hình Gốc và Tinh chỉnh
print("\n--- Final Comparison ---")
# Dựa trên kết quả trên, ta chọn ra bộ tham số tốt nhất
tuned_params = {
    'n_ants': 30, 'n_archive': 30, 'q': 0.05, 'xi': 0.85
}
# Chạy lại với cấu hình gốc
aco_base = ContinuousACO(sphere, dim, bounds, **baseline_params, max_iter=max_iter, seed=seed)
_, _, history_base = aco_base.optimize()
print(f"Finished Baseline, Final Fitness: {history_base[-1]:.4f}")

# Chạy với cấu hình đã tinh chỉnh
aco_tuned = ContinuousACO(sphere, dim, bounds, **tuned_params, max_iter=max_iter, seed=seed)
_, _, history_tuned = aco_tuned.optimize()
print(f"Finished Tuned, Final Fitness: {history_tuned[-1]:.4f}")


# =========================
#  Vẽ Biểu đồ
# =========================
fig, axs = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('Hyperparameter Tuning for Continuous ACO', fontsize=18)

# Biểu đồ cho q
ax = axs[0, 0]
for label, history in results['q'].items():
    ax.plot(history, label=label)
ax.set_title("Tuning 'q' (Selection Pressure)")
ax.set_xlabel('Iteration')
ax.set_ylabel('Best Fitness (log scale)')
ax.set_yscale('log')
ax.legend()
ax.grid(True, which="both", ls="--")

# Biểu đồ cho n_archive
ax = axs[0, 1]
for label, history in results['archive'].items():
    ax.plot(history, label=label)
ax.set_title("Tuning 'n_archive' (Archive Size)")
ax.set_xlabel('Iteration')
ax.set_ylabel('Best Fitness (log scale)')
ax.set_yscale('log')
ax.legend()
ax.grid(True, which="both", ls="--")

# Biểu đồ cho xi
ax = axs[1, 0]
for label, history in results['xi'].items():
    ax.plot(history, label=label)
ax.set_title("Tuning 'xi' (Exploration Factor)")
ax.set_xlabel('Iteration')
ax.set_ylabel('Best Fitness (log scale)')
ax.set_yscale('log')
ax.legend()
ax.grid(True, which="both", ls="--")

# Biểu đồ so sánh cuối cùng
ax = axs[1, 1]
ax.plot(history_base, label=f'Baseline (Final: {history_base[-1]:.2f})', linewidth=2, linestyle='--')
ax.plot(history_tuned, label=f'Tuned (Final: {history_tuned[-1]:.2f})', linewidth=2.5, color='red')
ax.set_title("Baseline vs. Tuned Configuration")
ax.set_xlabel('Iteration')
ax.set_ylabel('Best Fitness (log scale)')
ax.set_yscale('log')
ax.legend()
ax.grid(True, which="both", ls="--")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("aco_tuning_results.png", dpi=300)
plt.show()