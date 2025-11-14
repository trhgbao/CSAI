import argparse
import yaml
import time
import numpy as np
import random

from src.sphere.abc import *
from src.sphere.aco import *
from src.sphere.annealing import *
from src.sphere.fa import *
from src.sphere.ga import *
from src.sphere.pso import *
from pathlib import Path 
CONFIG_FOLDER = Path("./config/sphere/")


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_abc(cfg):
    random.seed(time.time())
    np.random.seed(int(time.time()))

    abc = ABC_Sphere_NP(
        n=cfg["dim"],
        SN=cfg["SN"],
        MAX_ITER=cfg["max_iter"],
        LIMIT=cfg["limit"],
        lb=cfg["bounds"][0],
        ub=cfg["bounds"][1],
    )

    start = time.time()
    abc.run()
    end = time.time()
    print(f"\nTotal runtime: {end - start:.4f} seconds")
    abc.visualize(img_path=cfg["plot_path"])



def run_annealing(cfg):
    sa = SimulatedAnnealingSphereFunction(
        dim=cfg["dim"],
        bounds=tuple(cfg["bounds"]),
        T0=cfg["T0"],
        T_min=cfg["T_min"],
        alpha=cfg["alpha"],
        max_iterations=cfg["max_iter"],
        step_size=cfg["step_size"]
    )
    best_solution, best_fitness = sa.solve()
    print(f"best solution: {best_solution}")
    print(f"best fitness: {best_fitness}")
    sa.plot_convergence(save_path=cfg["plot_path"])


def run_aco(cfg):
    aco = ContinuousACO(
        dim=cfg["dim"],
        bounds=cfg["bounds"],
        n_ants=cfg["n_ants"],
        n_archive=cfg["n_archive"],
        q=cfg["q"],
        xi=cfg["xi"],
        max_iter=cfg["max_iter"],
        use_roulette=cfg["use_roulette"],
        seed=cfg["seed"],
    )

    best_x, best_f = aco.optimize(verbose=True)
    print(f"best solution: {best_x}")
    print(f"best fitness: {best_f}")
    aco.visualize(img_path=cfg["plot_path"])


def run_fa(cfg):
    fa = FireflyAlgorithmSphereFunction(
        dim=cfg["dim"],
        bounds=tuple(cfg["bounds"])
    )
    best_solution, best_fitness = fa.solve(verbose=True)
    print(f"best solution: {best_solution}")
    print(f"best fitness: {best_fitness}")
    fa.plot_convergence(save_path=cfg["plot_path"])


def run_ga(cfg):
    ga = GeneticAlgorithm(
        dim=cfg["dim"],
        bounds=cfg["bounds"],
        n_pop=cfg["n_pop"],
        crossover_rate=cfg["crossover_rate"],
        mutation_rate=cfg["mutation_rate"],
        mutation_strength=cfg["mutation_strength"],
        n_elite=cfg["n_elite"],
        tournament_size=cfg["tournament_size"],
        max_iter=cfg["max_iter"],
        seed=cfg["seed"],
    )

    best_x, best_f = ga.optimize(verbose=True)
    print(f"best solution: {best_x}")
    print(f"best fitness: {best_f}")
    ga.visualize(img_path=cfg["plot_path"])

def run_pso(cfg):
    pso = PSO(
        dim=cfg["dim"],
        max_iter=cfg["max_iter"]
    )
    best_fitness, best_solution = pso.optimize()
    pso.visualize(img_path=cfg["plot_path"])
    print(f"best solution: {best_solution}")
    print(f"best fitness: {best_fitness}",)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True, help="YAML config file")
    args = parser.parse_args()

    cfg = load_config(CONFIG_FOLDER / (args.algo + ".yaml"))
    algo = args.algo

    if algo == "abc":
        run_abc(cfg)
    elif algo == "aco":
        run_aco(cfg)
    elif algo == "annealing":
        run_annealing(cfg)
    elif algo == "fa":
        run_fa(cfg)
    elif algo == "ga":
        run_ga(cfg)
    elif algo == "pso":
        run_pso(cfg)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


if __name__ == "__main__":
    main()
