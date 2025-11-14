import numpy as np
import time
import math
import matplotlib.pyplot as plt


class CuckooSearchHybrid:
    def __init__(self, pop_size, max_gen, p_abandon, levy_beta,
                 initial_alpha, diversity_rate, search_range):
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.p_abandon = p_abandon
        self.levy_beta = levy_beta
        self.initial_alpha = initial_alpha
        self.diversity_rate = diversity_rate
        self.low, self.high = search_range

    # -----------------------------
    #  Objective function (Sphere)
    # -----------------------------
    @staticmethod
    def sphere(x):
        return np.sum(x ** 2)

    # -----------------------------
    #  Levy flight generator
    # -----------------------------
    def get_levy_flight_step(self, beta, dims):
        gamma_beta = math.gamma(1 + beta)
        sin_pi_beta = math.sin(math.pi * beta / 2)
        gamma_half = math.gamma((1 + beta) / 2)
        pow_two = 2 ** ((beta - 1) / 2)

        sigma_u = (gamma_beta * sin_pi_beta / (gamma_half * beta * pow_two)) ** (1 / beta)

        u = np.random.normal(0, sigma_u, size=dims)
        v = np.random.normal(0, 1, size=dims)
        step = u / (np.abs(v) ** (1 / beta))

        return step

    # -----------------------------
    #  Main Cuckoo Search method
    # -----------------------------
    def run(self, dims):
        nests = np.random.uniform(self.low, self.high, size=(self.pop_size, dims))
        fitness = np.array([self.sphere(n) for n in nests])

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_nest = nests[best_idx].copy()

        for gen in range(self.max_gen):
            current_alpha = self.initial_alpha * (1 - (gen + 1) / self.max_gen)

            # --- Lévy flight ---
            i = np.random.randint(0, self.pop_size)
            levy_step = self.get_levy_flight_step(self.levy_beta, dims)
            new_nest = nests[i] + current_alpha * levy_step
            new_nest = np.clip(new_nest, self.low, self.high)
            new_fitness = self.sphere(new_nest)

            # Replace random nest j
            j = np.random.randint(0, self.pop_size)
            if new_fitness < fitness[j]:
                nests[j], fitness[j] = new_nest, new_fitness

            # --- Abandon worst nests ---
            sorted_indices = np.argsort(fitness)
            num_abandon = int(self.p_abandon * self.pop_size)

            for idx in sorted_indices[-num_abandon:]:
                if np.random.rand() < self.diversity_rate:
                    new_nest_abandoned = np.random.uniform(self.low, self.high, size=dims)
                else:
                    j_rand, k_rand = np.random.choice(self.pop_size, 2, replace=False)
                    random_walk = np.random.rand() * (nests[j_rand] - nests[k_rand])
                    new_nest_abandoned = best_nest + random_walk

                new_nest_abandoned = np.clip(new_nest_abandoned, self.low, self.high)
                nests[idx] = new_nest_abandoned
                fitness[idx] = self.sphere(new_nest_abandoned)

            # Update best
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_fitness = fitness[current_best_idx]
                best_nest = nests[current_best_idx].copy()

        return best_nest, best_fitness


# =============================
#       MAIN EXECUTION
# =============================
if __name__ == '__main__':

    # --- Params ---
    POP_SIZE = 50
    MAX_GEN = 200
    P_ABANDON = 0.25
    LEVY_BETA = 1.5
    INITIAL_ALPHA = 1.0
    DIVERSITY_RATE = 0.1
    SEARCH_RANGE = (-5.12, 5.12)

    DIMENSIONS_TO_TEST = [5, 10, 20, 40, 60, 80, 100, 200, 300]

    # Create optimizer object
    optimizer = CuckooSearchHybridImproved(
        pop_size=POP_SIZE,
        max_gen=MAX_GEN,
        p_abandon=P_ABANDON,
        levy_beta=LEVY_BETA,
        initial_alpha=INITIAL_ALPHA,
        diversity_rate=DIVERSITY_RATE,
        search_range=SEARCH_RANGE
    )

    print("Thực thi thuật toán Cuckoo Search cải tiến trên hàm Sphere...")
    results = []

    # Run for each dimension
    for d in DIMENSIONS_TO_TEST:
        print(f"\nĐang xử lý với số chiều d = {d}...")
        start_time = time.time()
        _, min_value = optimizer.run(d)
        end_time = time.time()

        results.append({'dims': d, 'min_value': min_value, 'time': end_time - start_time})
        print(f"  -> Hoàn tất. Giá trị tối ưu tìm được: {min_value:.6e}, "
              f"Thời gian: {end_time - start_time:.4f} giây")

    # Print result table
    print("\n\n--- BẢNG KẾT QUẢ TỔNG HỢP ---")
    header = "| {:<15} | {:<25} | {:<15} |".format("Số chiều (d)", "Giá trị f(x) nhỏ nhất", "Thời gian (s)")
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for res in results:
        row = "| {:<15} | {:<25.6e} | {:<15.4f} |".format(
            res['dims'], res['min_value'], res['time'])
        print(row)

    print("-" * len(header))
