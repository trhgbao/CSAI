import numpy as np
import time

class HillClimbingColoring:
    def __init__(self, penalty_weight=1000, max_steps=200, num_restarts=5):
        self.PENALTY_WEIGHT = penalty_weight
        self.MAX_STEPS = max_steps
        self.NUM_RESTARTS = num_restarts

    # -------------------------------------------------
    # HÀM TÍNH FIT
    # -------------------------------------------------
    def calculate_fitness(self, solution, adj_list, penalty):
        unique_colors, normalized_solution = np.unique(solution, return_inverse=True)
        num_colors = len(unique_colors)
        
        num_conflicts = 0
        for u, neighbors in enumerate(adj_list):
            for v in neighbors:
                if u < v and normalized_solution[u] == normalized_solution[v]:
                    num_conflicts += 1
                    
        fitness = num_colors + penalty * num_conflicts
        return fitness, num_colors, num_conflicts, normalized_solution

    # -------------------------------------------------
    # MỘT LẦN HILL CLIMBING
    # -------------------------------------------------
    def single_hill_climb_run(self, num_vertices, adj_list, initial_solution):
        solution = initial_solution.copy()
        fitness, _, _, solution = self.calculate_fitness(solution, adj_list, self.PENALTY_WEIGHT)

        for _ in range(self.MAX_STEPS):
            best_neighbor = None
            best_neighbor_fitness = fitness
            
            used_colors = np.unique(solution)

            for v_idx in range(num_vertices):
                original_color = solution[v_idx]

                for color in used_colors:
                    if color == original_color:
                        continue

                    neighbor = solution.copy()
                    neighbor[v_idx] = color
                    
                    neighbor_fitness, _, _, _ = self.calculate_fitness(neighbor, adj_list, self.PENALTY_WEIGHT)

                    if neighbor_fitness < best_neighbor_fitness:
                        best_neighbor_fitness = neighbor_fitness
                        best_neighbor = neighbor
            
            if best_neighbor is not None:
                fitness, _, _, solution = self.calculate_fitness(best_neighbor, adj_list, self.PENALTY_WEIGHT)
            else:
                break

        return solution

    # -------------------------------------------------
    # HILL CLIMBING CẢI TIẾN
    # -------------------------------------------------
    def improved_hill_climbing(self, num_vertices, adj_list):
        global_best_solution = None
        global_best_fitness = float('inf')

        for _ in range(self.NUM_RESTARTS):
            initial_k = np.random.randint(num_vertices // 2, num_vertices)
            initial_solution = np.random.randint(0, initial_k, size=num_vertices)

            local_best_solution = self.single_hill_climb_run(num_vertices, adj_list, initial_solution)

            fitness, _, _, _ = self.calculate_fitness(local_best_solution, adj_list, self.PENALTY_WEIGHT)

            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_solution = local_best_solution
        
        _, final_colors, final_conflicts, _ = self.calculate_fitness(global_best_solution, adj_list, self.PENALTY_WEIGHT)
        return final_colors, final_conflicts
