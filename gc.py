import argparse
import yaml
from pathlib import Path
import time

from src.gc.graph import Graph
from src.gc.aco import ACO_GraphColoring
from src.gc.ga import GeneticAlgorithm_GraphColoring
from src.gc.pso import PSO_Coloring_Real
from src.gc.abc import ABC_GC
from src.gc.annealing import SimulatedAnnealingGraphColoring
from src.gc.fa import FireflyAlgorithmGraphColoring
from src.gc.bfs import bfs_coloring, bfs_coloring_layer_sorted
from src.gc.cs import *
from src.gc.hc import HillClimbingColoring

CONFIG_FOLDER = Path("./config/gc")


def load_config(name):
    path = CONFIG_FOLDER / f"{name}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)



def run_aco(cfg, graph):
    start_time = time.time()

    n_colors = cfg.get("n_colors", "auto")
    if n_colors == "auto":
        n_colors = graph.max_degree + 1

    aco = ACO_GraphColoring(
        adjacency=graph.adjacency,
        n_colors=n_colors,
        n_ants=cfg["n_ants"],
        n_iterations=cfg["n_iterations"],
        alpha=cfg["alpha"],
        beta=cfg["beta"],
        rho=cfg["rho"],
        q=cfg["q"],
        seed=cfg.get("seed", 42),
        use_dsatur=cfg.get("use_dsatur", False),
        gamma=cfg["gamma"]
    )
    _, used = aco.run(verbose=cfg["verbose"])
    aco.visuazlie(img_path=cfg.get("plot_path", "result/result.png"))
    print("Best colors:", used)

    end_time = time.time()
    print(f"Runtime: {end_time - start_time:.6f} s")



def run_fa(cfg, graph):
    start_time = time.time()

    fa = FireflyAlgorithmGraphColoring(
        graph,
        num_fireflies=cfg["num_fireflies"],
        max_iterations=cfg["max_iterations"],
        use_dsatur=cfg.get("use_dsatur", False),
    )
    best = fa.solve()
    print("Colors:", best.count_colors(), "| Conflicts:", best.count_conflicts())
    fa.plot_convergence(save_path=cfg.get("plot_path", "result/result.png"))

    end_time = time.time()
    print(f"Runtime: {end_time - start_time:.6f} s")



def run_ga(cfg, graph):
    start_time = time.time()

    n_colors = cfg.get("n_colors", "auto")
    if n_colors == "auto":
        n_colors = graph.max_degree + 1

    ga = GeneticAlgorithm_GraphColoring(
        adjacency=graph.adjacency,
        n_colors=n_colors,
        n_pop=cfg["population"],
        n_generations=cfg["generations"],
        crossover_rate=cfg["crossover_rate"],
        mutation_rate=cfg["mutation_rate"],
        n_elite=cfg["elite"],
        tournament_size=cfg["tournament_size"],
        seed=cfg.get("seed", 42),
    )
    _, used = ga.run(verbose=cfg.get("verbose", False))
    ga.visuazlie(img_path=cfg.get("plot_path", "result/result.png"))
    print("Best colors:", used)

    end_time = time.time()
    print(f"Runtime: {end_time - start_time:.6f} s")



def run_pso(cfg, graph):
    start_time = time.time()

    n_colors = cfg.get("n_colors", "auto")
    if n_colors == "auto":
        n_colors = graph.max_degree + 1

    pso = PSO_Coloring_Real(
        graph=graph.adjacency,
        max_color=n_colors,
        swarm_size=cfg["swarm_size"],
        max_iter=cfg["max_iter"],
        w=cfg["w"],
        c1=cfg["c1"],
        c2=cfg["c2"],
    )
    _, _, _, used = pso.optimize()
    print("Best colors:", used)

    end_time = time.time()
    print(f"Runtime: {end_time - start_time:.6f} s")



def run_abc(cfg, graph):
    start_time = time.time()

    abc = ABC_GC(
        n_=graph.num_vertices,
        m_=graph.num_edges,
        sn_=cfg["sn"],
        limit_=cfg["limit"],
        max_iter_=cfg["max_iter"],
        adj_=graph.adjacency
    )
    abc.run()

    end_time = time.time()
    print(f"Runtime: {end_time - start_time:.6f} s")



def run_annealing(cfg, graph):
    start_time = time.time()

    sa = SimulatedAnnealingGraphColoring(
        graph,
        T0=cfg["T0"],
        T_min=cfg["T_min"],
        alpha=cfg["alpha"],
        max_iterations=cfg["iters_per_temp"],
    )
    best = sa.solve()
    print("Best colors:", best.count_colors())
    sa.plot_convergence(save_path=cfg.get("plot_path", "result/result.png"))

    end_time = time.time()
    print(f"Runtime: {end_time - start_time:.6f} s")



def run_bfs(cfg, graph: Graph):
    start_time = time.time()

    sdl = cfg["SDL"]
    if sdl:
        color = bfs_coloring_layer_sorted(graph.num_vertices, graph.adjacency)
    else:
        color = bfs_coloring(graph.num_vertices, graph.adjacency)

    end_time = time.time()
    min_colors_used = max(color) + 1
    print("\nTotal colors used:", min_colors_used)
    print(f"Runtime: {end_time - start_time:.6f} s")



def run_cs(cfg, graph: Graph):
    start_time = time.time()

    model = CuckooGraphColoring(
        pop_size=cfg["pop_size"],
        max_gen=cfg["max_gen"],
        p_abandon=cfg["p_abandon"],
        penalty_weight=cfg["penalty_weight"],
        local_steps=cfg["local_steps"],
        levy_beta=cfg["levy_beta"],
        dsatur_ratio=cfg["dsatur_ratio"]
    )
<<<<<<< Updated upstream
    if not cfg["use_dsatur"]:
        num_colors, num_conflicts, best_solution = model.modified_cuckoo_search(graph.num_vertices, graph.adjacency)
    else:
        num_colors, num_conflicts, best_solution = model.cuckoo_search_with_dsatur_init(graph.num_vertices, graph.adjacency)
=======

    if not cfg["use_dsatur"]:
        num_colors, num_conflicts, best_solution = model.modified_cuckoo_search(graph.num_vertices, graph.adjacency)
        algo_suffix = "mcoa"
    else:
        num_colors, num_conflicts, best_solution = model.cuckoo_search_with_dsatur_init(graph.num_vertices, graph.adjacency)
        algo_suffix = "cs_dsatur"

    end_time = time.time()
    
    print("\n--- Cuckoo Search Results ---")
    print(f"Algorithm: {'CS-DSATUR' if cfg['use_dsatur'] else 'MCOA'}")
    print(f"Best colors found: {num_colors}")
    print(f"Remaining conflicts: {num_conflicts}")
    print(f"Runtime: {end_time - start_time:.6f} s")

    img_save_path = f"result/gc_cs.png"
    model.visualize(img_save_path, use_dsatur=cfg["use_dsatur"])   
    start_time = time.time()

    model = CuckooGraphColoring(
        pop_size=cfg["pop_size"],
        max_gen=cfg["max_gen"],
        p_abandon=cfg["p_abandon"],
        penalty_weight=cfg["penalty_weight"],
        local_steps=cfg["local_steps"],
        levy_beta=cfg["levy_beta"],
        dsatur_ratio=cfg["dsatur_ratio"]
    )
    if not cfg["use_dsatur"]:
        num_colors, num_conflicts, best_solution = model.modified_cuckoo_search(graph.num_vertices, graph.adjacency)
    else:
        num_colors, num_conflicts, best_solution
        n = model.cuckoo_search_with_dsatur_init(graph.num_vertices, graph.adjacency)
>>>>>>> Stashed changes

    end_time = time.time()
    print("\nTotal colors used:", num_colors)
    print(f"Runtime: {end_time - start_time:.6f} s")



def run_hc(cfg, graph: Graph):
    start_time = time.time()

<<<<<<< Updated upstream
    model = HillClimbingColoring(
=======
    solver = HillClimbingColoring(
>>>>>>> Stashed changes
        penalty_weight=cfg["penalty_weight"],
        max_steps=cfg["max_steps"],
        num_restarts=cfg["num_restarts"]
    )

<<<<<<< Updated upstream
    num_colors, num_conflicts = model.improved_hill_climbing(
        graph.num_vertices, graph.adjacency
    )

    print("Colors =", num_colors)
    print("Conflicts =", num_conflicts)

    end_time = time.time()
    print(f"Runtime: {end_time - start_time:.6f} s")

=======
    num_colors, num_conflicts = solver.improved_hill_climbing(graph.num_vertices, graph.adjacency)
    
    end_time = time.time()

    print("\n--- Hill Climbing Results ---")
    print(f"Best colors found: {num_colors}")
    print(f"Remaining conflicts: {num_conflicts}")
    print(f"Runtime: {end_time - start_time:.6f} s")

    img_save_path = f"result/gc_hc.png"
    
    solver.visualize(img_save_path)

>>>>>>> Stashed changes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True, help="Algorithm name")

    args = parser.parse_args()
    algo = args.algo.lower()

    cfg = load_config(algo)

    graph_path = cfg["graph_file"]
    graph = Graph(file_path=graph_path)
    graph.print_info()

    if algo == "abc":
        run_abc(cfg, graph)
    elif algo == "aco":
        run_aco(cfg, graph)
    elif algo == "annealing":
        run_annealing(cfg, graph)
    elif algo == "fa":
        run_fa(cfg, graph)
    elif algo == "ga":
        run_ga(cfg, graph)
    elif algo == "pso":
        run_pso(cfg, graph)
    elif algo == "bfs":
        run_bfs(cfg, graph)
    elif algo == "cs":
        run_cs(cfg, graph)
    elif algo == "hc":
        run_hc(cfg, graph)
    else:
        print(f"Unknown algorithm: {algo}")


if __name__ == "__main__":
    main()