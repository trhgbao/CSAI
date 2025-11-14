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

CONFIG_FOLDER = Path("./config/gc")

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True, help="Name of algorithm")
    parser.add_argument("--input", default="./data/myciel3.txt", help="Graph file path")

    args = parser.parse_args()

    algo_name = args.algo
    cfg = load_config(CONFIG_FOLDER / (algo_name + ".yaml"))

    graph = Graph(file_path=args.input)
    graph.print_info()

    if algo_name == "aco":
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

        best_coloring, used_colors, history = aco.run(verbose=cfg.get("verbose", False))
        print("\nBest number of colors:", used_colors)

    elif algo_name == "fa":

        fa = FireflyAlgorithmGraphColoring(
            graph,
            num_fireflies=cfg["num_fireflies"],
            max_iterations=cfg["max_iterations"],
            use_dsatur=cfg.get("use_dsatur", False),
        )

        best = fa.solve()

        print("\nColors used:", best.count_colors())
        print("Conflicts:", best.count_conflicts())


    elif algo_name == "ga":
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
            seed=cfg.get("seed", 42)
        )

        best_coloring, used_colors, history = ga.run(verbose=cfg.get("verbose"))
        print("\nBest number of colors:", used_colors)

    elif algo_name == "pso":
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
            c2=cfg["c2"]
        )

        bf, bc, best_color, used_colors = pso.optimize()
        print("\nBest number of colors:", used_colors)


    elif algo_name == "abc":

        abc = ABC_GC(
            n_=graph.num_vertices,
            m_=graph.num_edges,
            sn_=cfg["sn"],
            limit_=cfg["limit"],
            max_iter_=cfg["max_iter"],
            adj_=graph.adjacency
        )

        abc.run()

    elif algo_name == "annealing":
        sa = SimulatedAnnealingGraphColoring(
            graph,
            T0=cfg["T0"],
            T_min=cfg["T_min"],
            alpha=cfg["alpha"],
            max_iterations=cfg["iters_per_temp"],
        )
        best = sa.solve()

        print("\nBest:", best.count_colors())

    else:
        print("Unknown algorithm config filename")


if __name__ == "__main__":
    main()
