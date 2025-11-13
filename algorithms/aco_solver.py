import numpy as np
import random
import networkx as nx

def calculate_graph_fitness(solution, adj_list, penalty_weight):
    """Tính điểm chất lượng: số màu + phạt cho xung đột."""
    # Đảm bảo chỉ tính các màu thực tế được sử dụng (không phải -1)
    num_colors = len(np.unique(solution[solution != -1]))
    num_conflicts = 0
    for u, neighbors in enumerate(adj_list):
        for v in neighbors:
            # Chỉ kiểm tra xung đột nếu cả hai đỉnh đều đã được tô màu và có cùng màu
            if solution[u] != -1 and solution[v] != -1 and u < v and solution[u] == solution[v]:
                num_conflicts += 1
    # Nếu có đỉnh chưa được tô màu, đó cũng là một dạng "xung đột" hoặc lời giải chưa hoàn chỉnh.
    # Tuy nhiên, trong ACO này, các đỉnh luôn được tô màu.
    num_uncolored_nodes = np.sum(solution == -1)
    if num_uncolored_nodes > 0:
        # Có thể thêm phạt nếu lời giải chưa hoàn chỉnh
        # Ví dụ: num_colors + penalty_weight * (num_conflicts + num_uncolored_nodes)
        pass # Với ACO, giải pháp luôn hoàn chỉnh

    return num_colors + penalty_weight * num_conflicts, num_colors, num_conflicts

class AntColonyOptimizationGraphColoring:
    def __init__(self, graph, n_colors, n_ants, n_iter, alpha, beta, rho, q, use_dsatur, gamma, penalty_weight):
        self.adjacency = nx.to_numpy_array(graph)
        self.graph = graph # Giữ lại để lấy adj_list
        self.n_nodes = self.adjacency.shape[0]
        self.n_colors = n_colors # Số màu tối đa mà kiến có thể chọn
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.use_dsatur = use_dsatur
        self.gamma = gamma
        self.penalty_weight = penalty_weight 

        # Pheromone matrix: n_nodes x n_colors
        self.pheromone = np.ones((self.n_nodes, self.n_colors))
        self.neighbors = [np.where(self.adjacency[i])[0] for i in range(self.n_nodes)]
        
        degrees = np.sum(self.adjacency, axis=1)
        # Sắp xếp theo thứ tự giảm dần của bậc nếu không dùng DSATUR
        self.node_order = np.argsort(-degrees) 
        self.degrees = degrees

        self.best_solution_global = None # Khởi tạo là None
        self.best_fitness_global_tuple = (float('inf'), float('inf'), float('inf'))
        self.history = [float('inf')] # Lịch sử fitness

    def heuristic(self, coloring, node):
        """Tính giá trị heuristic cho việc gán màu cho một đỉnh."""
        neighbor_colors = coloring[self.neighbors[node]]
        # Chỉ xét các màu đã được gán cho hàng xóm
        used_neighbor_colors = neighbor_colors[neighbor_colors != -1]
        
        counts = np.bincount(used_neighbor_colors, minlength=self.n_colors)
        
        # Giá trị heuristic: ưu tiên các màu ít được dùng bởi hàng xóm
        heuristic_values = 1.0 / (1.0 + counts)

        # Giảm giá trị heuristic cho các màu đã được sử dụng bởi các đỉnh khác trong đồ thị
        # để khuyến khích sử dụng lại màu đã có.
        # Nhưng ở đây, logic gamma đã làm điều ngược lại: phạt màu mới.
        # Cân nhắc lại: Gamma (New Color Penalty)
        # Nếu muốn ưu tiên màu đã có, thì màu mới phải có heuristic thấp hơn.
        # current_solution_unique_colors = np.unique(coloring[coloring != -1])
        # mask_new_colors = np.ones(self.n_colors, dtype=bool)
        # if current_solution_unique_colors.size > 0:
        #     mask_new_colors[current_solution_unique_colors] = False
        # heuristic_values[mask_new_colors] /= self.gamma # Gamma > 1 sẽ phạt màu mới

        return heuristic_values

    def select_next_node_dsatur(self, uncolored_indices, saturation):
        """Chọn đỉnh tiếp theo để tô màu bằng chiến lược DSATUR."""
        if not uncolored_indices.size:
            return -1 # Không còn đỉnh nào để tô

        max_sat = np.max(saturation[uncolored_indices])
        candidates = uncolored_indices[saturation[uncolored_indices] == max_sat]
        
        if len(candidates) > 1:
            # Phá vỡ hòa bằng cách chọn đỉnh có bậc cao nhất
            return candidates[np.argmax(self.degrees[candidates])]
        return candidates[0]

    def _construct_solution(self):
        """Mỗi con kiến xây dựng một lời giải tô màu."""
        coloring = np.full(self.n_nodes, -1, dtype=int) # -1 nghĩa là chưa tô màu
        uncolored_mask = np.ones(self.n_nodes, dtype=bool)
        saturation = np.zeros(self.n_nodes, dtype=int) # Chỉ dùng cho DSATUR
        order_ptr = 0 # Chỉ dùng nếu không DSATUR

        for _ in range(self.n_nodes):
            node_to_color = -1
            if self.use_dsatur:
                uncolored_idx = np.where(uncolored_mask)[0]
                node_to_color = self.select_next_node_dsatur(uncolored_idx, saturation)
            else:
                while order_ptr < self.n_nodes and not uncolored_mask[self.node_order[order_ptr]]:
                    order_ptr += 1
                if order_ptr < self.n_nodes:
                    node_to_color = self.node_order[order_ptr]

            if node_to_color == -1: # Không tìm được đỉnh nào để tô màu
                break

            # Tính xác suất chọn màu cho node_to_color
            eta = self.heuristic(coloring, node_to_color) # Heuristic
            tau = self.pheromone[node_to_color] # Pheromone

            # Xác suất tổng hợp
            probs = (tau ** self.alpha) * (eta ** self.beta)

            # Loại bỏ các màu bị cấm bởi hàng xóm
            forbidden_colors = np.unique(coloring[self.neighbors[node_to_color]][coloring[self.neighbors[node_to_color]] != -1])
            if forbidden_colors.size > 0:
                probs[forbidden_colors] = 0.0

            # Xử lý trường hợp tất cả xác suất đều bằng 0 hoặc NaN
            if np.sum(probs) == 0 or np.isnan(np.sum(probs)):
                # Nếu không có màu hợp lệ, gán màu ngẫu nhiên trong số các màu có thể
                # Hoặc, chỉ cho phép các màu không bị cấm có xác suất bằng nhau
                allowed_colors_mask = np.ones(self.n_colors, dtype=bool)
                if forbidden_colors.size > 0:
                    allowed_colors_mask[forbidden_colors] = False
                
                # Nếu tất cả các màu đều bị cấm, chọn một màu ngẫu nhiên (sẽ tạo xung đột)
                if not allowed_colors_mask.any():
                    chosen_color = np.random.randint(0, self.n_colors)
                else:
                    # Chọn ngẫu nhiên từ các màu được phép
                    allowed_indices = np.where(allowed_colors_mask)[0]
                    chosen_color = np.random.choice(allowed_indices)

            else:
                probs /= np.sum(probs) # Chuẩn hóa xác suất
                chosen_color = np.random.choice(self.n_colors, p=probs)
            
            coloring[node_to_color] = chosen_color

            # Cập nhật độ bão hòa cho các hàng xóm nếu dùng DSATUR
            if self.use_dsatur:
                for nb in self.neighbors[node_to_color]:
                    if coloring[nb] == -1: # Chỉ cập nhật cho các đỉnh chưa tô màu
                        neighbor_colors_of_nb = coloring[self.neighbors[nb]]
                        saturation[nb] = len(np.unique(neighbor_colors_of_nb[neighbor_colors_of_nb != -1]))
            
            uncolored_mask[node_to_color] = False
            if not uncolored_mask.any(): # Nếu tất cả đỉnh đã được tô màu
                break
        return coloring

    def step(self):
        """Thực hiện một bước (thế hệ) của thuật toán ACO."""
        all_colorings = []
        all_fitness_tuples = []

        # Mỗi con kiến xây dựng một lời giải
        for _ in range(self.n_ants):
            c = self._construct_solution()
            all_colorings.append(c)
            all_fitness_tuples.append(calculate_graph_fitness(c, self.neighbors, self.penalty_weight))
        
        self.pheromone *= (1 - self.rho) # Bốc hơi pheromone

        # Cập nhật pheromone dựa trên các lời giải của kiến
        for ant_idx in range(self.n_ants):
            cost = all_fitness_tuples[ant_idx][0]
            # Tránh chia cho 0 hoặc giá trị quá nhỏ nếu cost = -1 (không thể xảy ra với fitness)
            delta_pheromone = self.q / (cost + 1e-6) # Thêm một số nhỏ để ổn định
            coloring = all_colorings[ant_idx]
            
            # Cập nhật pheromone cho các cặp (đỉnh, màu) trong lời giải
            # Đảm bảo chỉ cập nhật cho các đỉnh có màu hợp lệ
            valid_nodes_mask = coloring != -1
            self.pheromone[np.arange(self.n_nodes)[valid_nodes_mask], coloring[valid_nodes_mask]] += delta_pheromone

        # Cập nhật lời giải tốt nhất toàn cục
        current_best_ant_idx = np.argmin([f[0] for f in all_fitness_tuples])
        if all_fitness_tuples[current_best_ant_idx][0] < self.best_fitness_global_tuple[0]:
            self.best_fitness_global_tuple = all_fitness_tuples[current_best_ant_idx]
            self.best_solution_global = all_colorings[current_best_ant_idx]

        self.history.append(self.best_fitness_global_tuple[0])
        return self._get_state()

    def _get_state(self):
        """Trả về trạng thái hiện tại của thuật toán để trực quan hóa."""
        return {
            "colors": self.best_solution_global, # Có thể là None nếu chưa tìm được
            "fitness_tuple": self.best_fitness_global_tuple,
            "history": self.history
        }