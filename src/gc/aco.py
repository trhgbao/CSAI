import numpy as np
import matplotlib.pyplot as plt
import random
import time


class ACO_GraphColoring:
    def __init__(self, adjacency, n_colors, n_ants=10, n_iterations=100,
                 alpha=1.0, beta=2.0, rho=0.1, q=100.0,
                 seed=None, use_dsatur=True, gamma=1e6):

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # ======== CHUYỂN SANG DANH SÁCH KỀ ========
        # adjacency là list of lists
        self.adjacency = adjacency
        self.n_nodes = len(adjacency)
        self.n_colors = n_colors

        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.use_dsatur = use_dsatur
        self.gamma = float(gamma)

        # pheromone[i][c]
        self.pheromone = np.ones((self.n_nodes, self.n_colors))

        # neighbors chính là adjacency
        self.neighbors = adjacency

        # degree = len(neighbors)
        degrees = np.array([len(adjacency[i]) for i in range(self.n_nodes)])
        self.node_order = np.argsort(-degrees)

        self.best_coloring = None
        self.best_used_colors = np.inf
        self.history = []


    # =======================
    # Heuristic bên dưới giữ nguyên
    # =======================
    def heuristic(self, coloring, node):
        neighbor_colors = coloring[self.neighbors[node]]

        used = (coloring != -1)
        used_colors = np.flatnonzero(np.bincount(coloring[used], minlength=self.n_colors))

        counts = np.bincount(
            neighbor_colors[neighbor_colors != -1],
            minlength=self.n_colors
        )

        heuristic_values = 1.0 / (1.0 + counts)

        mask_new_colors = np.ones(self.n_colors, dtype=bool)
        mask_new_colors[used_colors] = False

        heuristic_values[mask_new_colors] /= self.gamma
        return heuristic_values


    def select_next_node_dsatur(self, uncolored, saturation, degrees):
        max_sat = np.max(saturation[uncolored])
        candidates = uncolored[saturation[uncolored] == max_sat]
        if len(candidates) > 1:
            node = candidates[np.argmax(degrees[candidates])]
        else:
            node = candidates[0]
        return node


    def run(self, verbose=True):

        degrees = np.array([len(nbs) for nbs in self.neighbors])

        for iteration in range(self.n_iterations):

            all_colorings = np.full((self.n_ants, self.n_nodes), -1, dtype=int)

            # --- Ant loop ---
            for ant in range(self.n_ants):

                coloring = all_colorings[ant]
                uncolored_mask = np.ones(self.n_nodes, dtype=bool)
                saturation = np.zeros(self.n_nodes, dtype=int)
                order_ptr = 0

                for _ in range(self.n_nodes):

                    # chọn node
                    if self.use_dsatur:
                        uncolored_idx = np.where(uncolored_mask)[0]
                        node = self.select_next_node_dsatur(
                            uncolored_idx, saturation, degrees
                        )
                    else:
                        while order_ptr < self.n_nodes and not uncolored_mask[self.node_order[order_ptr]]:
                            order_ptr += 1
                        node = self.node_order[order_ptr]

                    neighbors = self.neighbors[node]

                    # heuristic + pheromone
                    eta = self.heuristic(coloring, node) ** self.beta
                    tau = self.pheromone[node] ** self.alpha
                    probs = tau * eta

                    # cấm màu của hàng xóm
                    neighbor_colors = coloring[neighbors]
                    forbidden = np.unique(neighbor_colors[neighbor_colors != -1])
                    probs[forbidden] = 0

                    if probs.sum() == 0 or np.isnan(probs.sum()):
                        probs[:] = 1.0

                    probs /= probs.sum()

                    chosen_color = np.random.choice(self.n_colors, p=probs)
                    coloring[node] = chosen_color

                    # update saturation
                    if self.use_dsatur:
                        for nb in neighbors:
                            if coloring[nb] == -1:
                                neighbor_colors_nb = coloring[self.neighbors[nb]]
                                saturation[nb] = len(np.unique(neighbor_colors_nb[neighbor_colors_nb != -1]))

                    uncolored_mask[node] = False
                    if not uncolored_mask.any():
                        break

            # --- Evaluate + Update pheromone ---
            num_used_colors = np.array([len(np.unique(c)) for c in all_colorings])

            self.pheromone *= (1 - self.rho)

            for ant in range(self.n_ants):
                cost = num_used_colors[ant]
                delta = self.q / (cost + 1.0)
                coloring = all_colorings[ant]

                self.pheromone[np.arange(self.n_nodes), coloring] += delta

                if cost < self.best_used_colors:
                    self.best_used_colors = cost
                    self.best_coloring = coloring.copy()

            if verbose:
                print(f"Iteration {iteration+1:03d} | best colors = {self.best_used_colors}")

            self.history.append(self.best_used_colors)

        return self.best_coloring, self.best_used_colors

    def visuazlie(self, img_path):
        plt.figure(figsize=(6,4))
        plt.plot(self.history, marker='o')
        plt.xlabel("Iteration")
        plt.ylabel("Best #Colors")
        plt.title("ACO Graph Coloring - Convergence")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(img_path, dpi=300)
        plt.show()