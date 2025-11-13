import numpy as np
import random
import math
from problems.sphere_function import sphere

# --- HÀM TIỆN ÍCH DÙNG CHUNG ---
# (Bạn có thể chuyển các hàm calculate_fitness từ file cs_mcoa_solver.py sang một file tiện ích chung)
def calculate_graph_fitness(solution, adj_list, penalty_weight):
    num_colors = len(np.unique(solution))
    num_conflicts = 0
    for u, neighbors in enumerate(adj_list):
        for v in neighbors:
            if u < v and solution[u] == solution[v]:
                num_conflicts += 1
    return num_colors + penalty_weight * num_conflicts, num_colors, num_conflicts

# ====================================================================
# 1. HILL CLIMBING (STEEPEST ASCENT)
# ====================================================================

class HillClimbingBase:
    def __init__(self, initial_solution, fitness_func):
        self.current_solution = initial_solution
        self.fitness_func = fitness_func
        self.current_fitness_tuple = self.fitness_func(self.current_solution)
        self.history = [self.current_fitness_tuple[0]]
        self.is_stuck = False

    def _get_neighbors(self):
        """PHẢI ĐƯỢC GHI ĐÈ BỞI LỚP CON. Trả về một danh sách các lời giải hàng xóm."""
        raise NotImplementedError

    def step(self):
        if self.is_stuck:
            return self._get_state()

        neighbors = self._get_neighbors()
        if not neighbors:
            self.is_stuck = True
            return self._get_state()

        neighbor_fitness_tuples = [self.fitness_func(n) for n in neighbors]
        best_neighbor_idx = np.argmin([f[0] for f in neighbor_fitness_tuples])
        
        best_neighbor_solution = neighbors[best_neighbor_idx]
        best_neighbor_fitness_tuple = neighbor_fitness_tuples[best_neighbor_idx]

        if best_neighbor_fitness_tuple[0] < self.current_fitness_tuple[0]:
            self.current_solution = best_neighbor_solution
            self.current_fitness_tuple = best_neighbor_fitness_tuple
        else:
            self.is_stuck = True # Đã đạt tới đỉnh cục bộ

        self.history.append(self.current_fitness_tuple[0])
        return self._get_state()

    def _get_state(self):
        raise NotImplementedError # Lớp con phải định nghĩa cách trả về state


class HillClimbingGraphColoring(HillClimbingBase):
    def __init__(self, graph, penalty_weight):
        self.graph = graph
        self.num_vertices = graph.number_of_nodes()
        self.adj_list = [list(graph.neighbors(i)) for i in range(self.num_vertices)]
        self.penalty_weight = penalty_weight
        
        initial_solution = np.random.randint(0, self.num_vertices, size=self.num_vertices)
        fitness_func = lambda sol: calculate_graph_fitness(sol, self.adj_list, self.penalty_weight)
        super().__init__(initial_solution, fitness_func)

    def _get_neighbors(self):
        neighbors = []
        # Tạo hàng xóm bằng cách đổi màu của 1 đỉnh
        for i in range(self.num_vertices):
            original_color = self.current_solution[i]
            # Thử các màu khác
            num_colors = len(np.unique(self.current_solution))
            for c in range(num_colors + 1):
                if c != original_color:
                    neighbor = self.current_solution.copy()
                    neighbor[i] = c
                    neighbors.append(neighbor)
        return neighbors

    def _get_state(self):
        return {
            "colors": self.current_solution,
            "fitness_tuple": self.current_fitness_tuple,
            "history": self.history
        }

# ====================================================================
# 2. SIMULATED ANNEALING (SA)
# ====================================================================

class SimulatedAnnealingBase:
    def __init__(self, initial_solution, fitness_func, initial_temp, cooling_rate):
        self.current_solution = initial_solution
        self.fitness_func = fitness_func
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        
        self.current_temp = self.initial_temp

        # --- MỚI: Tính toán fitness ban đầu một cách an toàn ---
        # Hàm fitness_func có thể trả về scalar (Sphere) hoặc tuple (Graph Coloring)
        initial_fitness_val = self.fitness_func(self.current_solution)
        if not isinstance(initial_fitness_val, tuple):
            self.current_fitness_tuple = (initial_fitness_val,)
        else:
            self.current_fitness_tuple = initial_fitness_val
        
        self.best_solution = self.current_solution.copy()
        self.best_fitness_tuple = self.current_fitness_tuple # Best fitness ban đầu giống current
        self.history = [self.best_fitness_tuple[0]] # Bây giờ history sẽ được khởi tạo đúng

    def _get_random_neighbor(self):
        raise NotImplementedError

    def step(self):
        if self.current_temp < 0.01: 
            return self._get_state()

        neighbor = self._get_random_neighbor(self.current_solution, self.current_temp) # Truyền current_solution và current_temp
        neighbor_fitness_val = self.fitness_func(neighbor) # Chỉ là một giá trị đơn (energy)
        
        # Đảm bảo neighbor_fitness_tuple là một tuple
        if not isinstance(neighbor_fitness_val, tuple):
            neighbor_fitness_tuple = (neighbor_fitness_val,)
        else:
            neighbor_fitness_tuple = neighbor_fitness_val

        delta_E = neighbor_fitness_tuple[0] - self.current_fitness_tuple[0] 

        if delta_E < 0 or random.uniform(0, 1) < math.exp(-delta_E / self.current_temp):
            self.current_solution = neighbor.copy() if hasattr(neighbor, 'copy') else neighbor # Đảm bảo copy nếu là object
            self.current_fitness_tuple = neighbor_fitness_tuple

            if self.current_fitness_tuple[0] < self.best_fitness_tuple[0]:
                self.best_solution = self.current_solution.copy() if hasattr(self.current_solution, 'copy') else self.current_solution
                self.best_fitness_tuple = self.current_fitness_tuple

        self.current_temp *= self.cooling_rate
        self.history.append(self.best_fitness_tuple[0])
        return self._get_state()

    def _get_state(self):
        raise NotImplementedError


class SimulatedAnnealingGraphColoring(SimulatedAnnealingBase):
    def __init__(self, graph, penalty_weight, initial_temp, cooling_rate, n_iter):
        # `graph` ở đây là một đối tượng networkx.Graph
        self.graph_nx = graph
        self.num_vertices = graph.number_of_nodes()
        self.adj_list = [list(graph.neighbors(i)) for i in range(self.num_vertices)]
        self.penalty_weight = penalty_weight # SA sẽ sử dụng energy riêng, nhưng giữ lại cho tương thích
        
        # max_colors có thể ước tính từ bậc của đồ thị, hoặc đơn giản là num_vertices
        degrees = [d for n, d in self.graph_nx.degree()]
        self.max_colors = max(degrees) + 1 if degrees else 1 # Ước tính ban đầu

        # Khởi tạo solution ngẫu nhiên với lớp SAGraphSolution
        initial_solution = SAGraphSolution(self.adj_list, self.num_vertices, self.max_colors)
        
        # Hàm fitness ở đây là hàm energy của solution
        super().__init__(initial_solution, lambda s: s.energy(), initial_temp, cooling_rate)
        
        # Cập nhật best_solution và best_fitness_tuple ban đầu
        # self.best_solution = initial_solution.copy()
        # self.best_fitness_tuple = (initial_solution.energy(), initial_solution.count_colors(), initial_solution.count_conflicts())
        # self.history = [self.best_fitness_tuple[0]] # Lịch sử sẽ lưu best_energy

    def _get_random_neighbor(self, current_solution_obj, current_T):
        """
        Tạo solution láng giềng bằng cách đổi màu của 1 đỉnh ngẫu nhiên
        """
        neighbor = current_solution_obj.copy()
        vertex = random.randint(0, self.num_vertices - 1)
        new_color = random.randint(0, self.max_colors - 1)
        neighbor.coloring[vertex] = new_color
        return neighbor

    def step(self):
        # Ghi đè phương thức step() của SimulatedAnnealingBase để xử lý object Solution
        if self.current_temp < 0.01: # T_min
            return self._get_state()

        neighbor_obj = self._get_random_neighbor(self.current_solution, self.current_temp)
        neighbor_energy = neighbor_obj.energy()

        delta_E = neighbor_energy - self.current_fitness_tuple[0]

        if delta_E < 0 or random.uniform(0, 1) < math.exp(-delta_E / self.current_temp):
            self.current_solution = neighbor_obj.copy() # Cập nhật solution object
            self.current_fitness_tuple = (neighbor_energy, neighbor_obj.count_colors(), neighbor_obj.count_conflicts())

            if self.current_fitness_tuple[0] < self.best_fitness_tuple[0]:
                self.best_solution = self.current_solution.copy()
                self.best_fitness_tuple = self.current_fitness_tuple

        self.current_temp *= self.cooling_rate
        self.history.append(self.best_fitness_tuple[0])
        return self._get_state()

    def _get_state(self):
        # Trả về các thông tin cần thiết để visualize Graph Coloring
        return {
            "colors": self.best_solution.coloring,
            "fitness_tuple": self.best_fitness_tuple, # (energy, num_colors, num_conflicts)
            "history": self.history
        }

# ====================================================================
# 3. GENETIC ALGORITHM (GA)
# ====================================================================

class GeneticAlgorithmBase:
    def __init__(self, n_pop, mutation_rate, crossover_rate):
        self.n_pop = n_pop
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = self._initial_population()
        self.fitness_tuples = [self._calculate_fitness(ind) for ind in self.population]
        
        best_idx = np.argmin([f[0] for f in self.fitness_tuples])
        self.best_solution = self.population[best_idx]
        self.best_fitness_tuple = self.fitness_tuples[best_idx]
        self.history = [self.best_fitness_tuple[0]]
        
    def _initial_population(self): raise NotImplementedError
    def _calculate_fitness(self, individual): raise NotImplementedError
    def _crossover(self, parent1, parent2): raise NotImplementedError
    def _mutate(self, individual): raise NotImplementedError
    
    def _tournament_selection(self, k=3):
        selection_ix = random.randint(0, self.n_pop - 1)
        for _ in range(k - 1):
            ix = random.randint(0, self.n_pop - 1)
            if self.fitness_tuples[ix][0] < self.fitness_tuples[selection_ix][0]:
                selection_ix = ix
        return self.population[selection_ix]

    def step(self):
        new_population = []
        # Giữ lại cá thể tốt nhất (Elitism)
        best_idx = np.argmin([f[0] for f in self.fitness_tuples])
        new_population.append(self.population[best_idx])

        while len(new_population) < self.n_pop:
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            new_population.append(self._mutate(child1))
            if len(new_population) < self.n_pop:
                new_population.append(self._mutate(child2))

        self.population = new_population
        self.fitness_tuples = [self._calculate_fitness(ind) for ind in self.population]

        best_idx = np.argmin([f[0] for f in self.fitness_tuples])
        if self.fitness_tuples[best_idx][0] < self.best_fitness_tuple[0]:
            self.best_solution = self.population[best_idx]
            self.best_fitness_tuple = self.fitness_tuples[best_idx]
        
        self.history.append(self.best_fitness_tuple[0])
        return self._get_state()

    def _get_state(self): raise NotImplementedError

class GeneticAlgorithmGraphColoring(GeneticAlgorithmBase):
    def __init__(self, graph, penalty_weight, n_pop, mutation_rate, crossover_rate):
        self.graph = graph
        self.num_vertices = graph.number_of_nodes()
        self.adj_list = [list(graph.neighbors(i)) for i in range(self.num_vertices)]
        self.penalty_weight = penalty_weight
        self.max_colors = self.num_vertices
        super().__init__(n_pop, mutation_rate, crossover_rate)
        
    def _initial_population(self):
        return [np.random.randint(0, self.max_colors, size=self.num_vertices) for _ in range(self.n_pop)]
        
    def _calculate_fitness(self, individual):
        return calculate_graph_fitness(individual, self.adj_list, self.penalty_weight)
        
    def _crossover(self, parent1, parent2):
        # Uniform crossover
        crossover_point = random.randint(1, self.num_vertices - 1)
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return child1, child2
        
    def _mutate(self, individual):
        for i in range(self.num_vertices):
            if random.random() < self.mutation_rate:
                individual[i] = random.randint(0, self.max_colors - 1)
        return individual
        
    def _get_state(self):
        return {
            "colors": self.best_solution,
            "fitness_tuple": self.best_fitness_tuple,
            "history": self.history
        }
    

class HillClimbingSphere(HillClimbingBase):
    def __init__(self, objective_func, bounds, dims, n_iter, step_size=0.05):
        self.bounds = bounds
        self.dims = dims
        self.step_size = step_size
        
        min_bound, max_bound = self.bounds
        initial_solution = np.random.uniform(min_bound, max_bound, self.dims)
        
        # Hàm fitness cho Sphere chỉ có 1 giá trị, ta tạo tuple để đồng bộ
        fitness_func = lambda sol: (objective_func(sol),) 
        
        super().__init__(initial_solution, fitness_func)

    def _get_neighbors(self):
        neighbors = []
        # Tạo hàng xóm bằng cách di chuyển một chút theo mỗi chiều
        for i in range(self.dims):
            # Hàng xóm dương
            neighbor_pos = self.current_solution.copy()
            neighbor_pos[i] += self.step_size
            neighbors.append(np.clip(neighbor_pos, self.bounds[0], self.bounds[1]))
            # Hàng xóm âm
            neighbor_neg = self.current_solution.copy()
            neighbor_neg[i] -= self.step_size
            neighbors.append(np.clip(neighbor_neg, self.bounds[0], self.bounds[1]))
        return neighbors

    def _get_state(self):
        # Trực quan hóa chỉ cần 1 điểm, ta tạo mảng 2D
        return {
            "positions": self.current_solution.reshape(1, -1), 
            "gbest_val": self.current_fitness_tuple[0],
            "history": self.history
        }

# --- SIMULATED ANNEALING CHO SPHERE ---

class SimulatedAnnealingSphere(SimulatedAnnealingBase):
    def __init__(self, objective_func, bounds, dims, n_iter, initial_temp, cooling_rate, step_size=1.0):
        self.dims = dims
        self.bounds = bounds
        self.objective_func = objective_func
        self.step_size = step_size # Thêm step_size từ triển khai của bạn
        
        # Khởi tạo giải pháp ngẫu nhiên ban đầu
        min_bound, max_bound = self.bounds
        initial_solution = np.random.uniform(min_bound, max_bound, self.dims)
        
        # Hàm fitness ở đây là hàm mục tiêu trực tiếp
        super().__init__(initial_solution, self.objective_func, initial_temp, cooling_rate)
        
        # Cập nhật best_solution và best_fitness_tuple ban đầu cho Sphere
        # self.best_solution = initial_solution.copy()
        # self.best_fitness_tuple = (self.objective_func(initial_solution),)
        # self.history = [self.best_fitness_tuple[0]] # Lịch sử sẽ lưu best_energy (f(x))

    def _get_random_neighbor(self, current_pos, current_T):
        """
        Tạo solution láng giềng bằng cách thêm nhiễu Gaussian
        Độ lớn nhiễu giảm theo nhiệt độ (adaptive step size)
        """
        # Step size thích nghi: giảm khi nhiệt độ giảm
        # T0 trong SABase là initial_temp
        adaptive_step = self.step_size * (current_T / self.initial_temp)
        neighbor = current_pos + np.random.normal(0, adaptive_step, self.dims)
        return np.clip(neighbor, self.bounds[0], self.bounds[1])

    def _get_state(self):
        return {
            "positions": self.best_solution.reshape(1, -1), # Luôn trả về best_solution
            "gbest_val": self.best_fitness_tuple[0],
            "history": self.history
        }

# --- GENETIC ALGORITHM CHO SPHERE ---

class GeneticAlgorithmSphere(GeneticAlgorithmBase):
    def __init__(self, objective_func, bounds, dims, n_pop, n_iter, mutation_rate, crossover_rate):
        self.func = objective_func
        self.bounds = bounds
        self.dims = dims
        super().__init__(n_pop, mutation_rate, crossover_rate)
        
    def _initial_population(self):
        min_bound, max_bound = self.bounds
        return [np.random.uniform(min_bound, max_bound, self.dims) for _ in range(self.n_pop)]
        
    def _calculate_fitness(self, individual):
        return (self.func(individual),)
        
    def _crossover(self, parent1, parent2):
        # Arithmetic crossover
        alpha = random.random()
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
        return child1, child2
        
    def _mutate(self, individual):
        # Gaussian mutation
        if random.random() < self.mutation_rate:
            mutation_value = np.random.normal(0, 0.1, self.dims)
            individual += mutation_value
            individual = np.clip(individual, self.bounds[0], self.bounds[1])
        return individual
        
    def _get_state(self):
        return {
            "positions": np.array(self.population),
            "gbest_val": self.best_fitness_tuple[0],
            "history": self.history
        }
    

class SAGraphSolution:
    """
    Biểu diễn một phương án tô màu đồ thị cho SA.
    Dựa trên graph của NetworkX (adj_list)
    """
    def __init__(self, adj_list, num_vertices, max_colors, coloring=None):
        self.adj_list = adj_list
        self.num_vertices = num_vertices
        self.max_colors = max_colors
        self.coloring = coloring if coloring is not None else np.random.randint(0, self.max_colors, self.num_vertices)

    def count_conflicts(self):
        c = 0
        for v in range(self.num_vertices):
            for nb in self.adj_list[v]:
                if self.coloring[v] == self.coloring[nb]:
                    c += 1
        return c // 2

    def count_colors(self):
        # Chỉ đếm các màu thực sự được sử dụng (không phải các giá trị màu tiềm năng chưa được gán)
        unique_colors = np.unique(self.coloring)
        return len(unique_colors)

    def energy(self):
        """
        Hàm năng lượng cho SA (càng thấp càng tốt)
        E = số_conflicts * 1000 + số_màu
        """
        return self.count_conflicts() * 1000 + self.count_colors()

    def copy(self):
        return SAGraphSolution(self.adj_list, self.num_vertices, self.max_colors, self.coloring.copy())

    def is_valid(self):
        return self.count_conflicts() == 0