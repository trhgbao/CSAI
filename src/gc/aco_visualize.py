import numpy as np
import matplotlib.pyplot as plt
import time
import networkx as nx
from aco import * 

gammas = [2, 10, 1000, 1e6]
results = {}
convergence_histories = {}

for gamma in gammas:
    print(f"\n=== Running ACO with gamma = {gamma} ===")

    aco = ACO_GraphColoring(
        adjacency=adjacency,
        n_colors=initial_colors,
        n_ants=40,
        n_iterations=10,
        alpha=1.66,
        beta=0.8,
        rho=0.35,
        q=100.0,
        seed=42,
        use_dsatur=False,  # cố định để so sánh ảnh hưởng gamma
        gamma=gamma
    )

    t0 = time.time()
    best_coloring, best_used_colors, history = aco.run(verbose=False)
    t1 = time.time()

    # Lưu kết quả
    results[gamma] = {
        "best_colors": best_used_colors,
        "time": t1 - t0,
        "coloring": best_coloring
    }
    convergence_histories[gamma] = history

    print(f"Gamma={gamma:>7} | Best colors: {best_used_colors:>3} | Time: {t1 - t0:.2f}s")


plt.figure(figsize=(8, 6))
for gamma, hist in convergence_histories.items():
    plt.plot(hist, marker='o', label=f'γ={gamma}')

plt.xlabel('Số vòng lặp (Iteration)', fontsize=12)
plt.ylabel('Số màu tốt nhất', fontsize=12)
plt.title('Đường hội tụ của ACO theo hệ số γ', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("convergence_gamma.png", dpi=300)
plt.show()


# === Running ACO with gamma = 2 ===
# Gamma=      2 | Best colors:  50 | Time: 13.73s

# === Running ACO with gamma = 10 ===
# Gamma=     10 | Best colors:  38 | Time: 12.78s

# === Running ACO with gamma = 1000 ===
# Gamma=   1000 | Best colors:  18 | Time: 12.89s

# === Running ACO with gamma = 1000000.0 ===
# Gamma=1000000.0 | Best colors:  16 | Time: 12.45s


# === Running ACO with gamma = 2 ===
# Gamma=      2 | Best colors:  50 | Time: 45.65s

# === Running ACO with gamma = 10 ===
# Gamma=     10 | Best colors:  41 | Time: 44.21s

# === Running ACO with gamma = 1000 ===
# Gamma=   1000 | Best colors:  16 | Time: 44.19s

# === Running ACO with gamma = 1000000.0 ===
# Gamma=1000000.0 | Best colors:  13 | Time: 44.43s