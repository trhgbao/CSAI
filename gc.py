from pathlib import Path
from src.gc.graph import Graph 

from src.gc.abc import *
from src.gc.pso import *

from src.gc.aco import *
from src.gc.ga import *

from src.gc.annealing import *
from src.gc.fa import *


file_path = Path("./data/200-3991.txt")
graph = Graph(file_path=file_path)

# linh
def abc():
    random.seed()

    SN = 50
    LIMIT = 20
    MAX_ITER = 200

    start = time.time()
    abc = ABC_GC(n_=graph.num_vertices, m_=graph.num_edges, sn_=SN, limit_=LIMIT, max_iter_=MAX_ITER, adj_=graph.adjacency)
    abc.run()
    end = time.time()
    print("Runtime:", end - start, "seconds")
    

def pso():
    max_color = graph.max_degree + 1
    pso = PSO_Coloring_Real(graph=graph.adjacency, max_color=max_color,
                            swarm_size=50, max_iter=500,
                            w=0.7, c1=1.5, c2=1.5)

    start = time.time()
    best_fitness, best_conf, best_color, used_colors = pso.optimize()
    end = time.time()

    print("\n=== Kết quả cuối cùng ===")
    print(f"Best Fitness = {best_fitness}")
    print(f"Conflicts = {best_conf}")
    print(f"Number of colors used = {used_colors}")
    print(f"Coloring = {best_color}")
    print(f"Time: {end - start:.3f} s")

# kim
def aco():
    # Initial colors (k) thường lấy max_degree + 1 (theo lý thuyết đồ thị)
    initial_colors = graph.max_degree + 1
    t0 = time.time()
    use_dsatur = int(input("use dsatur [type: 0/1]"))
    
    aco = ACO_GraphColoring(
        adjacency=graph.adjacency, 
        n_colors=initial_colors, 
        n_ants=40, 
        n_iterations=10, 
        alpha=1.66, 
        beta=0.8, 
        rho=0.35, 
        q=100.0, 
        seed=42,
        use_dsatur=use_dsatur,
        gamma=1e6
    )
    # Chạy thuật toán
    best_coloring, best_used_colors, history = aco.run(verbose=True)

    t1 = time.time()

    # Nếu tìm được lời giải tốt nhất, in kết quả và vẽ đồ thị
    if best_coloring is not None:
        print("\n--Results:")
        print(f"Execution time: {t1 - t0:.4f} seconds")
        print(f"Number of colors used: {best_used_colors}")
        # print(f"Best coloring found: {best_coloring}")
        

def ga():
    t0 = time.time()
    initial_colors = graph.max_degree + 1
    # Khởi tạo và chạy Genetic Algorithm
    ga = GeneticAlgorithm_GraphColoring(
        adjacency=graph.adjacency,
        n_colors=initial_colors,
        n_pop=100,
        n_generations=500,
        crossover_rate=0.85,
        mutation_rate=0.01,
        n_elite=5,
        tournament_size=3,
        seed=42
    )

    best_coloring, best_used_colors, history = ga.run(verbose=True)
    t1 = time.time()
    
    # In kết quả và vẽ đồ thị
    if best_coloring is not None:
        print("\n--Results:")
        print(f"Execution time: {t1 - t0:.4f} seconds")
        print(f"Number of colors used: {best_used_colors}")
        
        # Kiểm tra lại tính hợp lệ của lời giải cuối cùng
        final_conflicts = ga._calculate_fitness(best_coloring)
        print(f"Final solution conflicts: {final_conflicts}")
        
# # hiep
def annealing():
    print("=" * 70)
    print("🔥 SIMULATED ANNEALING - GRAPH COLORING")
    print("=" * 70)

    # Đọc đồ thị
    graph.print_info()

    # Chạy thuật toán
    print("\n🚀 Running Simulated Annealing...")
    sa = SimulatedAnnealingGraphColoring(
        graph,
        T0=1000,  # Nhiệt độ ban đầu
        T_min=0.1,  # Nhiệt độ tối thiểu
        alpha=0.95,  # Hệ số làm lạnh
        max_iterations=100  # Iterations mỗi nhiệt độ
    )
    best = sa.solve()

    # In kết quả cuối cùng
    print(f"\n{'=' * 70}")
    print(f"FINAL RESULTS")
    print(f"{'=' * 70}")
    print(f"Energy: {best.energy()}")
    print(f"Colors Used: {best.count_colors()}")
    print(f"Conflicts: {best.count_conflicts()}")
    print(f"Valid Solution: {'✅ Yes' if best.is_valid() else '❌ No'}")
    print(f"{'=' * 70}\n")

    # Vẽ đồ thị convergence
    print("📊 Plotting convergence curves...")
    sa.plot_convergence('convergence_sa_graph_coloring.png')

    print("\n✅ All done!")

def fa():
    print("=" * 70)
    print("🔥 FIREFLY ALGORITHM - GRAPH COLORING")
    print("=" * 70)

    graph.print_info()

    # ========== CẤU HÌNH: BẬT/TẮT DSATUR ==========
    USE_DSATUR = True  # Đổi thành False để tắt DSATUR
    # ===============================================

    # Chạy thuật toán
    print(f"\n🚀 Running Firefly Algorithm (DSATUR: {'ON' if USE_DSATUR else 'OFF'})...")
    fa_graph = FireflyAlgorithmGraphColoring(
        graph,
        num_fireflies=40,
        max_iterations=400,
        use_dsatur=USE_DSATUR  # Điều khiển DSATUR ở đây
    )
    best = fa_graph.solve()

    # In kết quả cuối cùng
    print(f"\n{'=' * 70}")
    print(f"FINAL RESULTS")
    print(f"{'=' * 70}")
    print(f"Colors Used: {best.count_colors()}")
    print(f"Conflicts: {best.count_conflicts()}")
    print(f"Fitness: {best.fitness():.2f}")
    print(f"Valid Solution: {'✅ Yes' if best.is_valid() else '❌ No'}")
    print(f"{'=' * 70}\n")

    # Vẽ đồ thị convergence
    print("📊 Plotting convergence curves...")
    suffix = '_dsatur' if USE_DSATUR else '_random'
    fa_graph.plot_convergence(f'convergence_graph_coloring{suffix}.png')

    print("\n✅ All done!")

# abc()
# pso()

# aco()
# ga()

# fa()
annealing()


