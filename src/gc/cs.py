import numpy as np
import time
<<<<<<< Updated upstream
=======
import matplotlib.pyplot as plt
>>>>>>> Stashed changes
import math

class CuckooGraphColoring:
    def __init__(self,
<<<<<<< Updated upstream
                 pop_size=50,
                 max_gen=200,
                 p_abandon=0.25,
                 penalty_weight=1000,
                 local_steps=50,
                 levy_beta=1.5,
                 dsatur_ratio=0.8):
=======
                pop_size=50,
                max_gen=200,
                p_abandon=0.25,
                penalty_weight=1000,
                local_steps=50,
                levy_beta=1.5,
                dsatur_ratio=0.8):
>>>>>>> Stashed changes

        self.POP_SIZE = pop_size
        self.MAX_GEN = max_gen
        self.P_ABANDON = p_abandon
        self.PENALTY_WEIGHT = penalty_weight
        self.LOCAL_STEPS = local_steps
        self.LEVY_BETA = levy_beta
        self.DSATUR_RATIO = dsatur_ratio
<<<<<<< Updated upstream
=======
        self.history = None
        self.graph_name = None
>>>>>>> Stashed changes

    # ---------------------------------------------------------
    # DSATUR heuristic
    # ---------------------------------------------------------
    def dsatur_coloring(self, num_vertices, adj_list):
        degrees = np.array([len(adj) for adj in adj_list])
        colors = np.full(num_vertices, -1, dtype=int)
        uncolored_vertices = set(range(num_vertices))

        while uncolored_vertices:
            max_sat = -1
            max_deg = -1
            next_vertex = -1

            sorted_uncolored = sorted(list(uncolored_vertices))

            for v_idx in sorted_uncolored:
                neighbor_colors = {colors[neighbor] for neighbor in adj_list[v_idx] if colors[neighbor] != -1}
                current_sat = len(neighbor_colors)

                if current_sat > max_sat:
                    max_sat = current_sat
                    max_deg = degrees[v_idx]
                    next_vertex = v_idx
                elif current_sat == max_sat and degrees[v_idx] > max_deg:
                    max_deg = degrees[v_idx]
                    next_vertex = v_idx

            used_neighbor_colors = {colors[neighbor] for neighbor in adj_list[next_vertex] if colors[neighbor] != -1}

            c = 0
            while True:
                if c not in used_neighbor_colors:
                    colors[next_vertex] = c
                    break
                c += 1

            uncolored_vertices.remove(next_vertex)

        return colors, len(np.unique(colors))

    # ---------------------------------------------------------
    # Fitness function
    # ---------------------------------------------------------
    def calculate_fitness(self, solution, adj_list):
        num_colors = len(np.unique(solution))
        num_conflicts = 0

        for u, neighbors in enumerate(adj_list):
            for v in neighbors:
                if u < v and solution[u] == solution[v]:
                    num_conflicts += 1

        fitness_score = num_colors + self.PENALTY_WEIGHT * num_conflicts
        return fitness_score, num_colors, num_conflicts

    # ---------------------------------------------------------
    # Levy flight step size
    # ---------------------------------------------------------
    def get_levy_step_size(self, beta, num_vertices):
        gamma_beta = math.gamma(1 + beta)
        sin_pi_beta = math.sin(math.pi * beta / 2)
        gamma_half = math.gamma((1 + beta) / 2)
        pow_two = 2**((beta - 1) / 2)

        sigma_u = (gamma_beta * sin_pi_beta / (gamma_half * beta * pow_two))**(1 / beta)
        u = np.random.normal(0, sigma_u)
        v = np.random.normal(0, 1)

        step = u / (abs(v)**(1 / beta))
        step_size = int(np.ceil(0.1 * abs(step)))

        return max(1, min(step_size, num_vertices // 2))

    # ---------------------------------------------------------
    # Discrete Levy flight
    # ---------------------------------------------------------
    def discrete_levy_flight(self, solution, max_colors, num_vertices):
        new_solution = solution.copy()
        num_to_change = self.get_levy_step_size(self.LEVY_BETA, num_vertices)

        indices_to_change = np.random.choice(len(solution), size=num_to_change, replace=False)
        new_colors = np.random.randint(0, max_colors, size=num_to_change)

        new_solution[indices_to_change] = new_colors
        return new_solution

    # ---------------------------------------------------------
    # Local search
    # ---------------------------------------------------------
    def local_search_intensification(self, solution, adj_list, max_steps):
        improved = solution.copy()
        _, _, current_conflicts = self.calculate_fitness(improved, adj_list)

        for _ in range(max_steps):
            if current_conflicts == 0:
                break

            conflicted_vertices = set()
            for u, neighbors in enumerate(adj_list):
                for v in neighbors:
                    if u < v and improved[u] == improved[v]:
                        conflicted_vertices.add(u)
                        conflicted_vertices.add(v)

            if not conflicted_vertices:
                break

            best_move = None
            max_reduction = -1

            for v_idx in conflicted_vertices:
                original_color = improved[v_idx]
                possible_colors = np.unique(improved)

                for color in possible_colors:
                    if color == original_color:
                        continue

                    temp = improved.copy()
                    temp[v_idx] = color
                    _, _, new_conflicts = self.calculate_fitness(temp, adj_list)
                    reduction = current_conflicts - new_conflicts

                    if reduction > max_reduction:
                        max_reduction = reduction
                        best_move = (v_idx, color)

            if best_move and max_reduction > 0:
                v_idx, color = best_move
                improved[v_idx] = color
                current_conflicts -= max_reduction
            else:
                break

        return improved

    # ---------------------------------------------------------
    # Modified Cuckoo Search
    # ---------------------------------------------------------
    def modified_cuckoo_search(self, num_vertices, adj_list):
        max_colors = num_vertices
        population = [np.random.randint(0, max_colors, size=num_vertices) for _ in range(self.POP_SIZE)]
        fitness_scores = [self.calculate_fitness(sol, adj_list) for sol in population]

        best_fit = float('inf')
        best_sol = None
<<<<<<< Updated upstream

        for gen in range(self.MAX_GEN):
=======
        
        convergence_history = {'best': [], 'average': [], 'worst': []}

        for gen in range(self.MAX_GEN):
            current_fits = [f[0] for f in fitness_scores]
            convergence_history['best'].append(np.min(current_fits))
            convergence_history['average'].append(np.mean(current_fits))
            convergence_history['worst'].append(np.max(current_fits))

>>>>>>> Stashed changes
            i = np.random.randint(0, self.POP_SIZE)
            cuckoo = self.discrete_levy_flight(population[i], max_colors, num_vertices)
            cuckoo_fit, _, _ = self.calculate_fitness(cuckoo, adj_list)

            j = np.random.randint(0, self.POP_SIZE)
            if cuckoo_fit < fitness_scores[j][0]:
                population[j] = cuckoo
                fitness_scores[j] = self.calculate_fitness(cuckoo, adj_list)

            num_abandon = int(self.P_ABANDON * self.POP_SIZE)
            sorted_idx = np.argsort([f[0] for f in fitness_scores])

            best_idx = sorted_idx[0]
            if fitness_scores[best_idx][0] < best_fit:
                best_fit = fitness_scores[best_idx][0]
                best_sol = population[best_idx]

            best_in_gen = population[best_idx]
<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
            for idx in sorted_idx[-num_abandon:]:
                refined = self.local_search_intensification(best_in_gen, adj_list, self.LOCAL_STEPS)
                population[idx] = refined
                fitness_scores[idx] = self.calculate_fitness(refined, adj_list)

<<<<<<< Updated upstream
=======
        self.history = convergence_history
>>>>>>> Stashed changes
        _, final_colors, final_conflicts = self.calculate_fitness(best_sol, adj_list)
        return final_colors, final_conflicts, best_sol

    # ---------------------------------------------------------
    # Cuckoo Search with DSATUR initialization
    # ---------------------------------------------------------
    def cuckoo_search_with_dsatur_init(self, num_vertices, adj_list):
        population = []
        num_dsatur = int(self.POP_SIZE * self.DSATUR_RATIO)

        for _ in range(num_dsatur):
            sol, _ = self.dsatur_coloring(num_vertices, adj_list)
            population.append(sol)

        max_colors = num_vertices
        for _ in range(self.POP_SIZE - num_dsatur):
            population.append(np.random.randint(0, max_colors, size=num_vertices))

        fitness_scores = [self.calculate_fitness(sol, adj_list) for sol in population]
        best_fit = float('inf')
        best_sol = None
<<<<<<< Updated upstream

        for gen in range(self.MAX_GEN):
=======
        
        convergence_history = {'best': [], 'average': [], 'worst': []}

        for gen in range(self.MAX_GEN):
            current_fits = [f[0] for f in fitness_scores]
            convergence_history['best'].append(np.min(current_fits))
            convergence_history['average'].append(np.mean(current_fits))
            convergence_history['worst'].append(np.max(current_fits))

>>>>>>> Stashed changes
            i = np.random.randint(0, self.POP_SIZE)
            cuckoo = self.discrete_levy_flight(population[i], max_colors, num_vertices)
            cuckoo_fit, _, _ = self.calculate_fitness(cuckoo, adj_list)

            j = np.random.randint(0, self.POP_SIZE)
            if cuckoo_fit < fitness_scores[j][0]:
                population[j] = cuckoo
                fitness_scores[j] = self.calculate_fitness(cuckoo, adj_list)

            num_abandon = int(self.P_ABANDON * self.POP_SIZE)
            sorted_idx = np.argsort([f[0] for f in fitness_scores])

            best_idx = sorted_idx[0]
            if fitness_scores[best_idx][0] < best_fit:
                best_fit = fitness_scores[best_idx][0]
                best_sol = population[best_idx]

            best_in_gen = population[best_idx]
<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
            for idx in sorted_idx[-num_abandon:]:
                refined = self.local_search_intensification(best_in_gen, adj_list, self.LOCAL_STEPS)
                population[idx] = refined
                fitness_scores[idx] = self.calculate_fitness(refined, adj_list)
<<<<<<< Updated upstream

        _, final_colors, final_conflicts = self.calculate_fitness(best_sol, adj_list)
        return final_colors, final_conflicts, best_sol
=======
                
        self.history = convergence_history
        _, final_colors, final_conflicts = self.calculate_fitness(best_sol, adj_list)
        return final_colors, final_conflicts, best_sol
    
    def visualize(self, img_path, use_dsatur):
            if not self.history:
                print("Chưa có lịch sử để vẽ. Vui lòng chạy thuật toán trước.")
                return

            plt.figure(figsize=(12, 7))
            
            plt.plot(self.history['best'], color='blue', linestyle='-', label='Best Fitness')
            plt.plot(self.history['average'], color='green', linestyle='--', label='Average Fitness')
            plt.plot(self.history['worst'], color='red', linestyle=':', label='Worst Fitness')
            
            algo_name = "CS-DSATUR" if use_dsatur else "MCOA"
            plt.title(f"{algo_name} Convergence Curve on {self.graph_name}", fontsize=16)
            plt.xlabel("Generation", fontsize=12)
            plt.ylabel("Fitness", fontsize=12)
            plt.grid(True, which="both", ls="--", linewidth=0.5)
            plt.legend()
            plt.tight_layout()
            
            plt.savefig(img_path, dpi=300)
            print(f"Đã lưu biểu đồ vào: {img_path}")
            plt.show()
>>>>>>> Stashed changes
