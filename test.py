# import numpy as np
# import matplotlib.pyplot as plt
# import networkx as nx
# import time
# import random

# class GeneticAlgorithm_GraphColoring:
#     """
#     Giải bài toán tô màu đồ thị bằng thuật toán Di truyền (Genetic Algorithm).
#     Mục tiêu là tìm một cách tô màu hợp lệ (không có xung đột) với số màu ít nhất.
#     """
#     def __init__(self, adjacency, n_colors, n_pop=50, n_generations=100,
#                  crossover_rate=0.8, mutation_rate=0.05, n_elite=2,
#                  tournament_size=3, seed=None):
#         # Nếu người dùng truyền seed: đặt seed cho reproducibility
#         if seed is not None:
#             np.random.seed(seed)
#             random.seed(seed)

#         # Lưu ma trận kề và thông tin đồ thị
#         self.adjacency = adjacency
#         self.n_nodes = adjacency.shape[0]
#         self.n_colors = n_colors  # Số màu tối đa được phép sử dụng
        
#         # Tham số của thuật toán GA
#         self.n_pop = n_pop  # Kích thước quần thể
#         self.n_generations = n_generations # Số thế hệ
#         self.crossover_rate = crossover_rate # Tỷ lệ lai ghép
#         self.mutation_rate = mutation_rate   # Tỷ lệ đột biến trên mỗi gen
#         self.n_elite = n_elite # Số cá thể ưu tú được giữ lại
#         self.tournament_size = tournament_size # Kích thước giải đấu cho lựa chọn

#         # Lấy danh sách các cạnh để tính fitness nhanh hơn
#         self.edges = np.argwhere(np.triu(adjacency) == 1)

#         # Khởi tạo quần thể ban đầu: mỗi cá thể là một mảng màu ngẫu nhiên
#         self.population = np.random.randint(0, self.n_colors, size=(n_pop, self.n_nodes))
#         # Fitness của mỗi cá thể (số xung đột, cần tối thiểu hóa)
#         self.fitness = np.zeros(n_pop)

#         # Biến lưu kết quả tốt nhất
#         self.best_coloring = None
#         self.best_used_colors = np.inf # Số màu sử dụng ít nhất (cho lời giải hợp lệ)
#         self.history = []


#     def _calculate_fitness(self, coloring):
#         """Tính fitness cho một cá thể (coloring). Fitness = số xung đột."""
#         conflicts = 0
#         # Duyệt qua danh sách cạnh đã tính trước
#         for u, v in self.edges:
#             if coloring[u] == coloring[v]:
#                 conflicts += 1
#         return conflicts

#     def _selection(self):
#         """Lựa chọn cha mẹ bằng phương pháp Tournament Selection."""
#         selected_indices = []
#         for _ in range(self.n_pop):
#             # Chọn ngẫu nhiên các đối thủ cho giải đấu
#             tournament_indices = np.random.randint(0, self.n_pop, self.tournament_size)
#             tournament_fitness = self.fitness[tournament_indices]
#             # Người chiến thắng là người có fitness nhỏ nhất (ít xung đột nhất)
#             winner_index = tournament_indices[np.argmin(tournament_fitness)]
#             selected_indices.append(winner_index)
#         return self.population[selected_indices]

#     def _crossover(self, parent1, parent2):
#         """Lai ghép hai cha mẹ bằng Single-Point Crossover."""
#         if random.random() < self.crossover_rate:
#             # Chọn một điểm cắt ngẫu nhiên
#             crossover_point = random.randint(1, self.n_nodes - 1)
#             child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
#             child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
#             return child1, child2
#         return parent1.copy(), parent2.copy()

#     def _mutation(self, individual):
#         """Đột biến một cá thể."""
#         for i in range(self.n_nodes):
#             if random.random() < self.mutation_rate:
#                 # Thay đổi màu của gen này thành một màu ngẫu nhiên khác
#                 individual[i] = random.randint(0, self.n_colors - 1)
#         return individual

#     def run(self, verbose=True):
#         for generation in range(self.n_generations):
#             # 1. Đánh giá fitness cho toàn bộ quần thể
#             self.fitness = np.array([self._calculate_fitness(ind) for ind in self.population])

#             # 2. Tìm cá thể tốt nhất trong thế hệ hiện tại
#             best_idx_current_gen = np.argmin(self.fitness)
#             best_fitness_current_gen = self.fitness[best_idx_current_gen]

#             # Nếu tìm thấy một lời giải hợp lệ (0 xung đột)
#             if best_fitness_current_gen == 0:
#                 valid_coloring = self.population[best_idx_current_gen]
#                 used_colors = len(np.unique(valid_coloring))
                
#                 # Cập nhật lời giải tốt nhất toàn cục nếu số màu ít hơn
#                 if used_colors < self.best_used_colors:
#                     self.best_used_colors = used_colors
#                     self.best_coloring = valid_coloring.copy()
#                     if verbose:
#                         print(f"Generation {generation+1:03d} | New best solution found! Colors = {self.best_used_colors}")

#             # 3. Tạo thế hệ mới
#             next_generation = []
            
#             # Elitism: Giữ lại các cá thể tốt nhất
#             elite_indices = np.argsort(self.fitness)[:self.n_elite]
#             for idx in elite_indices:
#                 next_generation.append(self.population[idx])

#             # 4. Lựa chọn, Lai ghép và Đột biến
#             selected_parents = self._selection()
            
#             # Điền phần còn lại của thế hệ mới
#             for i in range(self.n_elite, self.n_pop, 2):
#                 # Chọn cha mẹ từ pool đã được lựa chọn
#                 parent1 = selected_parents[i]
#                 # Đảm bảo không vượt quá chỉ số nếu n_pop là số lẻ
#                 parent2_idx = i + 1 if (i + 1) < self.n_pop else i
#                 parent2 = selected_parents[parent2_idx]

#                 child1, child2 = self._crossover(parent1, parent2)
                
#                 child1 = self._mutation(child1)
#                 child2 = self._mutation(child2)
                
#                 next_generation.append(child1)
#                 if len(next_generation) < self.n_pop:
#                     next_generation.append(child2)
            
#             self.population = np.array(next_generation)
            
#             # In thông tin tiến trình
#             if verbose:
#                  print(f"Generation {generation+1:03d} | Min Conflicts = {best_fitness_current_gen} | Best Colors Found = {self.best_used_colors}")
#             self.history.append(self.best_used_colors)

#         # Nếu không tìm được lời giải hợp lệ, trả về None
#         if self.best_coloring is None:
#              # Cố gắng trả về lời giải ít xung đột nhất
#              best_overall_idx = np.argmin(self.fitness)
#              self.best_coloring = self.population[best_overall_idx]
#              print(f"Warning: No valid coloring found. Returning solution with {self.fitness[best_overall_idx]} conflicts.")


#         return self.best_coloring, self.best_used_colors, self.history

# def read_graph_from_file(filepath):
#     # Đọc đồ thị từ file testcase theo định dạng:
#     # Dòng 1: n_nodes n_edges
#     # Các dòng sau: u v (1-based indices)
#     # Trả về số đỉnh, số cạnh, danh sách cạnh (0-based), và ma trận kề adjacency
#     with open(filepath, "r") as f:
#         lines = f.readlines()
#         n_nodes, n_edges = map(int, lines[0].split())
#         edges = [(int(u) - 1, int(v) - 1) for u, v in (line.split() for line in lines[1:])]
    
#     adjacency = np.zeros((n_nodes, n_nodes), dtype=int)
#     for u, v in edges:
#         adjacency[u, v] = 1
#         adjacency[v, u] = 1
        
#     return n_nodes, n_edges, edges, adjacency

# # ==========================
# #        MAIN SCRIPT
# # ==========================
# from pathlib import Path

# root = Path("./data")
# for _file in sorted(root.glob("*.txt")):
#     if "my" not in _file.name: continue
#     print("\ndata: ", _file.name)
#     n_nodes, n_edges, edges, adjacency = read_graph_from_file(_file)

#     degrees = np.sum(adjacency, axis=1)
#     max_degree = np.max(degrees)
#     initial_colors = max_degree + 1

#     # print(f"Number of nodes: {n_nodes}")
#     # print(f"Number of edges: {n_edges}")
#     # print(f"Max degree: {max_degree}")
#     # print(f"Initial number of colors (k): {initial_colors}")

#     t0 = time.time()

#     # Khởi tạo và chạy Genetic Algorithm
#     ga = GeneticAlgorithm_GraphColoring(
#         adjacency=adjacency,
#         n_colors=initial_colors,
#         n_pop=100,
#         n_generations=500,
#         crossover_rate=0.65,
#         mutation_rate=0.01,
#         n_elite=5,
#         tournament_size=3,
#         seed=42
#     )

#     best_coloring, best_used_colors, history = ga.run(verbose=False)
#     t1 = time.time()
    
#     # In kết quả và vẽ đồ thị
#     if best_coloring is not None:
#         print("\n--Results:")
#         print(f"Execution time: {t1 - t0:.4f} seconds")
#         print(f"Number of colors used: {best_used_colors}")
        
#         # Kiểm tra lại tính hợp lệ của lời giải cuối cùng
#         final_conflicts = ga._calculate_fitness(best_coloring)
#         print(f"Final solution conflicts: {final_conflicts}")


import numpy as np
import matplotlib.pyplot as plt

# ---- Particle and PSO ----
class Particle:
    def __init__(self, dim, x_min, x_max, v_max):
        self.position = np.random.uniform(x_min, x_max, dim)
        self.velocity = np.random.uniform(-v_max, v_max, dim)
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')

class PSO:
    def __init__(self, dim=3, swarm_size=40, max_iter=200,
                 x_min=-5.12, x_max=5.12, w=0.7, c1=1.5, c2=1.5):
        self.dim = dim
        self.x_min = x_min
        self.x_max = x_max
        self.v_max = (x_max - x_min)
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter

        self.swarm = [Particle(dim, x_min, x_max, self.v_max) for _ in range(swarm_size)]
        self.global_best = np.zeros(dim)
        self.global_best_fitness = float('inf')

    def sphere(self, x):
        return np.sum(x ** 2)

    def optimize(self, record_iters=[0, 25, 50, 75, 100]):
        history = {}

        # Evaluate initial
        for p in self.swarm:
            fitness = self.sphere(p.position)
            p.best_fitness = fitness
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best = p.position.copy()

        # Record initial swarm
        history[0] = np.array([p.position.copy() for p in self.swarm])

        for it in range(1, self.max_iter + 1):
            for p in self.swarm:
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                p.velocity = (self.w * p.velocity +
                              self.c1 * r1 * (p.best_position - p.position) +
                              self.c2 * r2 * (self.global_best - p.position))

                p.position += p.velocity

                fitness = self.sphere(p.position)
                if fitness < p.best_fitness:
                    p.best_fitness = fitness
                    p.best_position = p.position.copy()
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best = p.position.copy()

            if it in record_iters:
                history[it] = np.array([p.position.copy() for p in self.swarm])

        return history


# ---- Plotting ----
def plot_swarm_3D(positions, title):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')

    xs = positions[:,0]
    ys = positions[:,1]
    zs = positions[:,2]

    ax.scatter(xs, ys, zs, c='red', s=40)

    ax.set_xlim(-5.12, 5.12)
    ax.set_ylim(-5.12, 5.12)
    ax.set_zlim(-5.12, 5.12)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.show()


# ---- Run and Plot ----
pso = PSO(dim=3, swarm_size=50, max_iter=20000)
history = pso.optimize()

for it, pos in history.items():
    plot_swarm_3D(pos, f"Swarm at Iteration {it}")