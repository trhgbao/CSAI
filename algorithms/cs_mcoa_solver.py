
# algorithms/cs_mcoa_solver.py
import numpy as np
import math

# --- CÁC HÀM TIỆN ÍCH CHUNG ---

def dsatur_coloring(num_vertices, adj_list):
    """
    Thuật toán tô màu tham lam DSATUR.
    Trả về một lời giải (mảng màu).
    """
    degrees = np.array([len(adj) for adj in adj_list])
    colors = np.full(num_vertices, -1, dtype=int)
    uncolored_vertices = set(range(num_vertices))

    while uncolored_vertices:
        max_sat, max_deg, next_vertex = -1, -1, -1
        
        # Sắp xếp để có thứ tự nhất quán, giúp kết quả ổn định hơn
        sub_list_to_check = sorted(list(uncolored_vertices)) 

        for v_idx in sub_list_to_check:
            neighbor_colors = {colors[neighbor] for neighbor in adj_list[v_idx] if colors[neighbor] != -1}
            current_sat = len(neighbor_colors)

            if current_sat > max_sat:
                max_sat, max_deg, next_vertex = current_sat, degrees[v_idx], v_idx
            elif current_sat == max_sat and degrees[v_idx] > max_deg:
                max_deg, next_vertex = degrees[v_idx], v_idx
        
        if next_vertex == -1: # Nếu không tìm thấy đỉnh nào (trường hợp đồ thị rỗng)
            next_vertex = sub_list_to_check[0]

        used_neighbor_colors = {colors[neighbor] for neighbor in adj_list[next_vertex] if colors[neighbor] != -1}
        c = 0
        while True:
            if c not in used_neighbor_colors:
                colors[next_vertex] = c
                break
            c += 1
        uncolored_vertices.remove(next_vertex)
        
    return colors

def calculate_fitness(solution, adj_list, penalty_weight):
    """Tính điểm chất lượng: số màu + phạt cho xung đột."""
    num_colors = len(np.unique(solution))
    num_conflicts = 0
    for u, neighbors in enumerate(adj_list):
        for v in neighbors:
            if u < v and solution[u] == solution[v]:
                num_conflicts += 1
    return num_colors + penalty_weight * num_conflicts, num_colors, num_conflicts

def get_levy_step_size(beta, num_vertices):
    """Tạo ra một bước nhảy theo phân phối Lévy."""
    sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
               (math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
    sigma_v = 1
    u = np.random.normal(0, sigma_u)
    v = np.random.normal(0, sigma_v)
    step = u / (abs(v)**(1 / beta))
    step_size = int(np.ceil(0.1 * abs(step)))
    if step_size < 1: step_size = 1
    if step_size > num_vertices // 2: step_size = num_vertices // 2
    return step_size

def discrete_levy_flight(solution, max_colors):
    """Thực hiện bước nhảy Lévy trong không gian rời rạc."""
    new_solution = solution.copy()
    num_to_change = get_levy_step_size(1.5, len(solution)) # LEVY_BETA = 1.5
    indices_to_change = np.random.choice(len(solution), size=num_to_change, replace=False)
    new_colors = np.random.randint(0, max_colors, size=num_to_change)
    new_solution[indices_to_change] = new_colors
    return new_solution

def local_search_intensification(solution, adj_list, max_steps):
    """Tìm kiếm cục bộ để tinh chỉnh lời giải (chỉ dùng cho MCOA)."""
    improved_solution = solution.copy()
    _, _, current_conflicts = calculate_fitness(improved_solution, adj_list, 1)
    for _ in range(max_steps):
        if current_conflicts == 0: break
        conflicted_vertices = set()
        for u, neighbors in enumerate(adj_list):
            for v in neighbors:
                if u < v and improved_solution[u] == improved_solution[v]:
                    conflicted_vertices.add(u); conflicted_vertices.add(v)
        if not conflicted_vertices: break
        best_move, max_reduction = None, 0
        for v_idx in conflicted_vertices:
            original_color = improved_solution[v_idx]
            possible_colors = np.unique(improved_solution)
            for color in possible_colors:
                if color == original_color: continue
                temp_solution = improved_solution.copy()
                temp_solution[v_idx] = color
                _, _, new_conflicts = calculate_fitness(temp_solution, adj_list, 1)
                reduction = current_conflicts - new_conflicts
                if reduction > max_reduction:
                    max_reduction, best_move = reduction, (v_idx, color)
        if best_move:
            v_idx, new_color = best_move
            improved_solution[v_idx] = new_color
            current_conflicts -= max_reduction
        else: break
    return improved_solution

# --- LỚP CƠ SỞ CHUNG ---

class CuckooSearchBase:
    """Lớp cơ sở chứa các logic khởi tạo và cập nhật chung."""
    def __init__(self, graph, n_pop, p_abandon, penalty_weight):
        self.graph = graph
        self.adj_list = [list(graph.neighbors(i)) for i in range(graph.number_of_nodes())]
        self.num_vertices = graph.number_of_nodes()
        self.n_pop = n_pop
        self.p_abandon = p_abandon
        self.penalty_weight = penalty_weight
        self.max_colors = self.num_vertices # Bắt đầu với số màu tối đa

        # Khởi tạo quần thể
        self.population = [np.random.randint(0, self.max_colors, size=self.num_vertices) for _ in range(self.n_pop)]
        self.fitness_tuples = [calculate_fitness(sol, self.adj_list, self.penalty_weight) for sol in self.population]
        
        # Theo dõi lời giải tốt nhất
        best_idx = np.argmin([f[0] for f in self.fitness_tuples])
        self.gbest_fitness = self.fitness_tuples[best_idx][0]
        self.gbest_solution = self.population[best_idx].copy()
        
        self.history = [self.gbest_fitness]

    def _update_global_best(self):
        """Cập nhật lời giải tốt nhất toàn cục sau mỗi thế hệ."""
        current_best_idx = np.argmin([f[0] for f in self.fitness_tuples])
        if self.fitness_tuples[current_best_idx][0] < self.gbest_fitness:
            self.gbest_fitness = self.fitness_tuples[current_best_idx][0]
            self.gbest_solution = self.population[current_best_idx].copy()
        self.history.append(self.gbest_fitness)

    def _get_state(self): 
        """Trả về trạng thái hiện tại của thuật toán để trực quan hóa."""
        best_current_idx = np.argmin([f[0] for f in self.fitness_tuples])
        return {
            "colors": self.population[best_current_idx], # Trả về màu của lời giải tốt nhất hiện tại trong quần thể
            "fitness_tuple": self.fitness_tuples[best_current_idx],
            "history": self.history
        }

# --- PHIÊN BẢN CUCKOO SEARCH (CS) TIÊU CHUẨN ---

class StandardCS(CuckooSearchBase):
    def step(self):
        # 1. KHÁM PHÁ: Tạo cúc cu mới bằng Lévy flight
        i = np.random.randint(0, self.n_pop)
        cuckoo_solution = discrete_levy_flight(self.population[i], self.max_colors)
        cuckoo_fitness_tuple = calculate_fitness(cuckoo_solution, self.adj_list, self.penalty_weight)

        # 2. So sánh và thay thế
        j = np.random.randint(0, self.n_pop)
        if cuckoo_fitness_tuple[0] < self.fitness_tuples[j][0]:
            self.population[j] = cuckoo_solution
            self.fitness_tuples[j] = cuckoo_fitness_tuple

        # 3. LOẠI BỎ TỔ XẤU VÀ THAY BẰNG TỔ MỚI NGẪU NHIÊN
        num_abandon = int(self.p_abandon * self.n_pop)
        if num_abandon > 0:
            worst_indices = np.argsort([f[0] for f in self.fitness_tuples])[-num_abandon:]
            for idx in worst_indices:
                self.population[idx] = np.random.randint(0, self.max_colors, size=self.num_vertices)
                self.fitness_tuples[idx] = calculate_fitness(self.population[idx], self.adj_list, self.penalty_weight)
        
        self._update_global_best()
        
        # Trả về trạng thái để trực quan hóa
        best_current_idx = np.argmin([f[0] for f in self.fitness_tuples])
        return {
            "colors": self.population[best_current_idx],
            "fitness_tuple": self.fitness_tuples[best_current_idx],
            "history": self.history
        }
        return self._get_state() 

# --- PHIÊN BẢN MODIFIED CUCKOO SEARCH (MCOA) ---

class ModifiedCS(CuckooSearchBase):
    def __init__(self, graph, n_pop, p_abandon, penalty_weight, local_search_steps):
        super().__init__(graph, n_pop, p_abandon, penalty_weight)
        self.local_search_steps = local_search_steps

    def step(self):
        # 1. KHÁM PHÁ: Tạo cúc cu mới bằng Lévy flight (giống CS tiêu chuẩn)
        i = np.random.randint(0, self.n_pop)
        cuckoo_solution = discrete_levy_flight(self.population[i], self.max_colors)
        cuckoo_fitness_tuple = calculate_fitness(cuckoo_solution, self.adj_list, self.penalty_weight)

        # 2. So sánh và thay thế (giống CS tiêu chuẩn)
        j = np.random.randint(0, self.n_pop)
        if cuckoo_fitness_tuple[0] < self.fitness_tuples[j][0]:
            self.population[j] = cuckoo_solution
            self.fitness_tuples[j] = cuckoo_fitness_tuple

        # 3. KHAI THÁC: LOẠI BỎ TỔ XẤU VÀ THAY BẰNG LỜI GIẢI TỐT NHẤT ĐÃ TINH CHỈNH
        num_abandon = int(self.p_abandon * self.n_pop)
        if num_abandon > 0:
            sorted_indices = np.argsort([f[0] for f in self.fitness_tuples])
            best_current_solution = self.population[sorted_indices[0]]
            
            # Tinh chỉnh lời giải tốt nhất bằng tìm kiếm cục bộ
            refined_solution = local_search_intensification(best_current_solution, self.adj_list, self.local_search_steps)
            refined_fitness_tuple = calculate_fitness(refined_solution, self.adj_list, self.penalty_weight)
            
            # Thay thế các tổ xấu nhất bằng phiên bản đã tinh chỉnh này
            for idx in sorted_indices[-num_abandon:]:
                self.population[idx] = refined_solution
                self.fitness_tuples[idx] = refined_fitness_tuple

        self._update_global_best()
        
        # Trả về trạng thái để trực quan hóa
        best_current_idx = np.argmin([f[0] for f in self.fitness_tuples])
        return {
            "colors": self.population[best_current_idx],
            "fitness_tuple": self.fitness_tuples[best_current_idx],
            "history": self.history
        }
    
class CuckooSearchDSATUR_Base(CuckooSearchBase):
    """Lớp cơ sở mới kế thừa từ CuckooSearchBase, nhưng ghi đè phương thức khởi tạo."""
    def __init__(self, graph, n_pop, p_abandon, penalty_weight, dsatur_ratio, **kwargs):
        # Gọi __init__ của lớp cha, nhưng không khởi tạo quần thể ngay
        super().__init__(graph, n_pop, p_abandon, penalty_weight)

        # --- GHI ĐÈ PHẦN KHỞI TẠO QUẦN THỂ ---
        self.dsatur_ratio = dsatur_ratio
        
        population = []
        num_dsatur_solutions = int(self.n_pop * self.dsatur_ratio)

        # 1. Tạo các lời giải chất lượng cao bằng DSATUR
        for _ in range(num_dsatur_solutions):
            solution = dsatur_coloring(self.num_vertices, self.adj_list)
            population.append(solution)

        # 2. Tạo các lời giải ngẫu nhiên để duy trì sự đa dạng
        # Xác định max_colors từ các lời giải DSATUR để quần thể ngẫu nhiên không quá tệ
        if num_dsatur_solutions > 0:
            # Lấy số màu trung bình từ các lời giải tốt và cộng thêm một chút
            avg_colors = int(np.mean([len(np.unique(s)) for s in population]))
            self.max_colors = avg_colors + 5 
        else:
            self.max_colors = self.num_vertices

        for _ in range(self.n_pop - num_dsatur_solutions):
            population.append(np.random.randint(0, self.max_colors, size=self.num_vertices))
        
        # Cập nhật lại quần thể và fitness
        self.population = population
        self.fitness_tuples = [calculate_fitness(sol, self.adj_list, self.penalty_weight) for sol in self.population]
        
        # Cập nhật lại lời giải tốt nhất ban đầu
        best_idx = np.argmin([f[0] for f in self.fitness_tuples])
        self.gbest_fitness = self.fitness_tuples[best_idx][0]
        self.gbest_solution = self.population[best_idx].copy()
        self.history = [self.gbest_fitness]

class StandardCS_DSATUR(CuckooSearchDSATUR_Base, StandardCS):
    """
    Kế thừa từ cả hai:
    - CuckooSearchDSATUR_Base: Để có phương thức __init__ đã được cải tiến.
    - StandardCS: Để có phương thức step() tiêu chuẩn.
    """
    def __init__(self, graph, n_pop, p_abandon, penalty_weight, dsatur_ratio):
        # Gọi __init__ của lớp cơ sở DSATUR
        super().__init__(graph, n_pop, p_abandon, penalty_weight, dsatur_ratio)


# --- PHIÊN BẢN MODIFIED CUCKOO SEARCH (MCOA) + DSATUR ---

class ModifiedCS_DSATUR(CuckooSearchDSATUR_Base, ModifiedCS):
    """
    Kế thừa từ cả hai:
    - CuckooSearchDSATUR_Base: Để có phương thức __init__ đã được cải tiến.
    - ModifiedCS: Để có phương thức step() đã được sửa đổi.
    """
    def __init__(self, graph, n_pop, p_abandon, penalty_weight, local_search_steps, dsatur_ratio):
        # Phải gọi __init__ của lớp cơ sở DSATUR
        super().__init__(graph, n_pop, p_abandon, penalty_weight, dsatur_ratio)
        # Đồng thời, gán thêm thuộc tính mà ModifiedCS cần
        self.local_search_steps = local_search_steps