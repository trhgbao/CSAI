import argparse
import yaml
from pathlib import Path

from src.gc.graph import Graph
from src.gc.aco import ACO_GraphColoring
from src.gc.ga import GeneticAlgorithm_GraphColoring
from src.gc.pso import PSO_Coloring_Real
from src.gc.abc import ABC_GC
from src.gc.annealing import SimulatedAnnealingGraphColoring
from src.gc.fa import FireflyAlgorithmGraphColoring

CONFIG_FOLDER = Path("./config/gc")


def load_config(name):
    path = CONFIG_FOLDER / f"{name}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_aco(cfg, graph):
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
    _, used, _ = aco.run(verbose=cfg.get("verbose", False))
    print("Best colors:", used)


def run_fa(cfg, graph):
    fa = FireflyAlgorithmGraphColoring(
        graph,
        num_fireflies=cfg["num_fireflies"],
        max_iterations=cfg["max_iterations"],
        use_dsatur=cfg.get("use_dsatur", False),
    )
    best = fa.solve()
    print("Colors:", best.count_colors(), "| Conflicts:", best.count_conflicts())


def run_ga(cfg, graph):
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
    _, used, _ = ga.run(verbose=cfg.get("verbose", False))
    print("Best colors:", used)


def run_pso(cfg, graph):
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


def run_abc(cfg, graph):
    abc = ABC_GC(
        n_=graph.num_vertices,
        m_=graph.num_edges,
        sn_=cfg["sn"],
        limit_=cfg["limit"],
        max_iter_=cfg["max_iter"],
        adj_=graph.adjacency
    )
    abc.run()


def run_annealing(cfg, graph):
    sa = SimulatedAnnealingGraphColoring(
        graph,
        T0=cfg["T0"],
        T_min=cfg["T_min"],
        alpha=cfg["alpha"],
        max_iterations=cfg["iters_per_temp"],
    )
    best = sa.solve()
    print("Best colors:", best.count_colors())


ALGO_DISPATCH = {
    "aco": run_aco,
    "fa": run_fa,
    "ga": run_ga,
    "pso": run_pso,
    "abc": run_abc,
    "annealing": run_annealing,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True, help="Algorithm name")
    parser.add_argument("--input", help="Graph file override")

    args = parser.parse_args()
    algo = args.algo.lower()

    if algo not in ALGO_DISPATCH:
        raise ValueError(f"Unknown algorithm '{algo}'")

    # Load config
    cfg = load_config(algo)

    # Load graph from config, or override using --input
    graph_path = args.input if args.input else cfg["graph_file"]
    graph = Graph(file_path=graph_path)
    graph.print_info()

    # Run corresponding algorithm
    ALGO_DISPATCH[algo](cfg, graph)


if __name__ == "__main__":
    main()
