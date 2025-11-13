import numpy as np
import random
import networkx as nx

class ACO_GraphColoring:
    def __init__(self, graph, n_colors, n_ants=10, n_iter=100,
                 alpha=1.0, beta=2.0, rho=0.1, q=100.0,
                 seed=None, use_dsatur=True, gamma=1e6):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.adjacency = nx.to_numpy_array(graph)
        # Lưu ma trận kề (adjacency matrix). Đây là ma trận vuông kích thước n_nodes x n_nodes
        # self.adjacency = adjacency
        # Số đỉnh
        self.n_nodes = self.adjacency.shape[0]
        # Số màu tối đa cho thuật toán (k)
        self.n_colors = n_colors
        # Số kiến (số lời giải tạo mỗi iteration)
        self.n_ants = n_ants
        # Số vòng lặp của thuật toán
        self.n_iterations = n_iter
        # alpha: trọng số pheromone
        self.alpha = alpha
        # beta: trọng số heuristic (eta)
        self.beta = beta
        # rho: tỉ lệ bay hơi pheromone
        self.rho = rho
        # q: hằng số lắng đọng pheromone (ảnh hưởng lượng pheromone thêm vào)
        self.q = q
        # Chọn cơ chế chọn đỉnh: DSATUR (động theo độ bão hòa) nếu True, ngược lại dùng thứ tự cố định
        self.use_dsatur = use_dsatur
        # gamma: hệ số phạt màu mới trong heuristic (giá trị lớn => phạt mạnh màu chưa dùng)
        self.gamma = gamma
        # Khởi tạo ma trận pheromone: kích thước (n_nodes, n_colors)
        # pheromone[i, c] biểu diễn lượng pheromone cho việc gán màu c cho đỉnh i
        self.pheromone = np.ones((self.n_nodes, self.n_colors))

        # Tính trước danh sách hàng xóm cho từng đỉnh để truy xuất nhanh
        # neighbors[i] là mảng chỉ số các đỉnh kề với i
        self.neighbors = [np.where(self.adjacency[i])[0] for i in range(self.n_nodes)]

        # Nếu không dùng DSATUR, ta dùng thứ tự cố định theo degree giảm dần:
        # node_order là mảng các chỉ số đỉnh được sắp theo bậc giảm dần
        self.degrees = np.sum(self.adjacency, axis=1)
        self.node_order = np.argsort(-self.degrees)

        # Biến lưu kết quả tốt nhất tới hiện tại
        self.best_coloring = None
        # best_used_colors lưu số lượng màu tốt nhất (khởi tạo inf để dễ so sánh)
        self.best_used_colors = np.inf
        
        self.history = []


    def heuristic(self, coloring, node):
        """
        Tính heuristic η_{i,c} cho một đỉnh node với tất cả màu c = 0..n_colors-1.
        - coloring: mảng kích thước n_nodes chứa màu đã gán (-1 nếu chưa gán)
        - Ý tưởng: ưu tiên các màu đã có (đã dùng trên đồ thị) và tránh màu đang xuất hiện nhiều
        - Ta tính counts = số lượng hàng xóm đã dùng mỗi màu, sau đó eta = 1/(1+counts)
        - Để khuyến khích dùng lại màu cũ (thay vì mở màu mới), ta "phạt" màu chưa dùng bằng gamma
        """
        # Mảng màu các hàng xóm (có thể chứa -1)
        neighbor_colors = coloring[self.neighbors[node]]
        # used: boolean đánh dấu các đỉnh đã được tô màu trong toàn đồ thị
        used = (coloring != -1)
        # used_colors: danh sách chỉ số màu đã xuất hiện ở bất kỳ vị trí nào (global used)
        # np.bincount trả về tần suất từng màu; flatnonzero lấy chỉ mục có count > 0
        used_colors = np.flatnonzero(np.bincount(coloring[used], minlength=self.n_colors))
        # counts: số lượng hàng xóm có mỗi màu (chỉ đếm hàng xóm đã được tô)
        counts = np.bincount(neighbor_colors[neighbor_colors != -1], minlength=self.n_colors)

        # heuristic cơ bản: 1/(1+counts) => màu ít cạnh tranh (ít hàng xóm dùng) được ưu tiên
        heuristic_values = 1.0 / (1.0 + counts)
        # mask_new_colors đánh dấu các màu chưa từng xuất hiện ở bất kỳ đỉnh nào (global)
        mask_new_colors = np.ones(self.n_colors, dtype=bool)
        mask_new_colors[used_colors] = False
        # Phạt mạnh màu mới: chia cho gamma lớn để giảm xác suất chọn màu mới
        heuristic_values[mask_new_colors] /= self.gamma  # phạt mạnh màu mới

        return heuristic_values


    def select_next_node_dsatur(self, uncolored, saturation):
        """Chọn đỉnh tiếp theo theo DSATUR.
        - uncolored: mảng chỉ số các đỉnh chưa tô (ví dụ np.where(uncolored_mask)[0])
        - saturation: mảng saturation cho từng đỉnh (số màu khác nhau xuất hiện trong hàng xóm)
        - degrees: mảng bậc (degree) của từng đỉnh, dùng để phá vây khi tie
        DSATUR: chọn đỉnh có saturation cao nhất; nếu tie, chọn đỉnh có degree lớn hơn.
        """
        # saturation[uncolored] là các mức bão hòa chỉ của những đỉnh chưa tô
        max_sat = np.max(saturation[uncolored])
        # candidates: các đỉnh chưa tô có saturation = max_sat
        candidates = uncolored[saturation[uncolored] == max_sat]
        if len(candidates) > 1:
            # nếu có nhiều ứng viên, chọn đỉnh có degree lớn nhất trong số đó
            node = candidates[np.argmax(self.degrees[candidates])]
        else:
            node = candidates[0]
        return node

    def step(self):
        # Lưu tất cả lời giải của các kiến trong iteration hiện tại:
        # all_colorings[ant] là mảng màu của ant-th kiến
        all_colorings = np.full((self.n_ants, self.n_nodes), -1, dtype=int)

        # --- VÒNG SINH LỜI GIẢI: mỗi kiến sinh một lời giải độc lập ---
        for ant in range(self.n_ants):
            # coloring: tham chiếu tới 1 hàng trong all_colorings để thao tác nhanh
            coloring = all_colorings[ant]
            # uncolored_mask: boolean mask, True nếu đỉnh chưa tô
            # Dùng mask (numpy) thay vì set() giúp truy xuất nhanh hơn (O(1) vector ops)
            uncolored_mask = np.ones(self.n_nodes, dtype=bool)
            # saturation: số màu khác nhau xuất hiện trong hàng xóm (bắt đầu 0)
            saturation = np.zeros(self.n_nodes, dtype=int)
            # order_ptr: con trỏ dùng khi chọn theo node_order (degree order cố định)
            order_ptr = 0

            # Duyệt tối đa n_nodes lần để tô hết đỉnh
            for _ in range(self.n_nodes):
                # --- Chọn đỉnh cần tô tiếp theo ---
                if self.use_dsatur:
                    # Nếu dùng DSATUR: lấy chỉ số các đỉnh chưa tô (np.where nhanh hơn list)
                    uncolored_idx = np.where(uncolored_mask)[0]
                    # select_next_node_dsatur trả về 1 chỉ số đỉnh
                    node = self.select_next_node_dsatur(uncolored_idx, saturation)
                else:
                    # Nếu không dùng DSATUR: dùng thứ tự cố định self.node_order
                    # order_ptr lần lượt đi qua node_order và dừng ở đỉnh chưa tô
                    while order_ptr < self.n_nodes and not uncolored_mask[self.node_order[order_ptr]]:
                        order_ptr += 1
                    node = self.node_order[order_ptr]

                # Lấy danh sách hàng xóm của node để xử lý ràng buộc màu
                neighbors = self.neighbors[node]

                # --- Tính xác suất chọn màu: kết hợp pheromone (tau) và heuristic (eta) ---
                # eta: vector heuristic cho node (với trạng thái coloring hiện tại)
                eta = self.heuristic(coloring, node) ** self.beta
                # tau: pheromone ứng với node cho mỗi màu, nâng mũ alpha để điều chỉnh ảnh hưởng
                tau = self.pheromone[node] ** self.alpha
                # probs: tích giữa tau và eta (theo công thức ACO cổ điển)
                probs = tau * eta

                # --- Áp ràng buộc: cấm các màu đã xuất hiện trên hàng xóm ---
                neighbor_colors = coloring[neighbors]
                # forbidden: các màu bị cấm (unique để tránh lặp)
                forbidden = np.unique(neighbor_colors[neighbor_colors != -1])
                # Set xác suất của các màu cấm về 0
                probs[forbidden] = 0.0

                # Nếu tất cả xác suất đều 0 hoặc NaN (ví dụ do bị cấm tất cả màu) -> gán đều
                if probs.sum() == 0 or np.isnan(probs.sum()):
                    probs[:] = 1.0
                # Chuẩn hóa phân phối xác suất
                probs /= probs.sum()

                # Chọn màu theo phân phối probs
                chosen_color = np.random.choice(self.n_colors, p=probs)
                coloring[node] = chosen_color

                # --- Cập nhật saturation (chỉ khi dùng DSATUR) ---
                # Khi một đỉnh được tô, saturation của hàng xóm chưa tô có thể thay đổi
                if self.use_dsatur:
                    # Lấy các hàng xóm chưa tô rồi cập nhật saturation cho từng hàng xóm
                    uncolored_neighbors = [nb for nb in neighbors if coloring[nb] == -1]
                    if uncolored_neighbors:
                        for nb in uncolored_neighbors:
                            # neighbor_colors: màu của các hàng xóm của nb
                            neighbor_colors = coloring[self.neighbors[nb]]
                            # saturation[nb] = số màu khác nhau (loại bỏ -1)
                            saturation[nb] = len(np.unique(neighbor_colors[neighbor_colors != -1]))

                # Đánh dấu node đã được tô
                uncolored_mask[node] = False
                # Nếu không còn đỉnh chưa tô thì dừng sớm
                if not uncolored_mask.any():
                    break

        # --- Sau khi tất cả kiến sinh xong lời giải: đánh giá và cập nhật pheromone ---
        # num_used_colors[ant] = số màu được dùng trong lời giải của kiến ant
        num_used_colors = np.array([len(np.unique(c)) for c in all_colorings])

        # Bay hơi pheromone (toàn bộ ma trận)
        self.pheromone *= (1 - self.rho)

        # Lắng đọng pheromone dựa trên chất lượng lời giải (ở đây quality = số màu dùng ít hơn => tốt hơn)
        for ant in range(self.n_ants):
            cost = num_used_colors[ant]
            # delta_pheromone: lượng pheromone thêm vào — chọn q/(cost+1) để cost nhỏ được thưởng nhiều
            delta_pheromone = self.q / (cost + 1.0)
            coloring = all_colorings[ant]
            # Tăng pheromone cho mỗi (node, màu) đã được chọn trong lời giải này
            # Lưu ý: indexing np.arange(self.n_nodes), coloring gán cho từng đỉnh
            self.pheromone[np.arange(self.n_nodes), coloring] += delta_pheromone
            # Cập nhật kết quả tốt nhất nếu tìm được lời giải dùng ít màu hơn
            if cost < self.best_used_colors:
                self.best_used_colors = cost
                self.best_coloring = coloring.copy()

        # In thông tin tiến trình (mỗi iteration). Có thể tắt verbose để tiết kiệm thời gian khi chạy lâu
        # if verbose:
        #     print(f"Iteration {iteration+1:03d} | best colors = {self.best_used_colors}")
        self.history.append(self.best_used_colors)
        
        return self._get_state()

    def _get_state(self):
        """Trả về trạng thái hiện tại của thuật toán để trực quan hóa."""
        return {
            "colors": self.best_coloring, # Có thể là None nếu chưa tìm được
            "fitness_tuple": (self.best_used_colors, self.best_used_colors, self.best_used_colors),
            "history": self.history
        }