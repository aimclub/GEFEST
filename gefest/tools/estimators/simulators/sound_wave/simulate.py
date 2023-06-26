import os
import sys
sys.path.append('C:/Users/user2/GEFEST')
import pickle
import click
import multiprocessing
from joblib import Parallel, delayed
import numpy as np
num_cores = multiprocessing.cpu_count()

import cases.sound_waves.configuration.sw_domain as area
from simulator import SoundSimulator, generate_random_map, generate_map
domain, task = area.configurate_domain(poly_num=5, points_num=20, is_closed=True)

DEFAULT_OPTIONS = {
    'nb_ex': 1,
    'duration': 300,
    'data_path': '.'
}

HELP_MSG = {
    'nb_ex': 'Number of random examples to simulate',
    'duration': 'Number of iterations of the simulation',
    'data_path': 'Path to the folder in which simulation results are stored'
}

@click.command()
@click.option("--nb-ex", "-n", type=int, 
              default=DEFAULT_OPTIONS["nb_ex"],
              help=HELP_MSG["nb_ex"])
@click.option("--duration", "-d", type=int, 
              default=DEFAULT_OPTIONS["duration"],
              help=HELP_MSG["duration"])
@click.option("--data-path", "-p", type=str, 
              default=DEFAULT_OPTIONS["data_path"],
              help=HELP_MSG["data_path"])
def launch(nb_ex, duration, data_path):

    map_size = (domain.len_x, domain.len_y)

    def run_iteration(iteration, random_seed):
        print(f"Simulating random example {iteration+1}/{nb_ex} ...")
        file_name = f"example_{iteration}.pickle"
        file_path = os.path.join(data_path, file_name)
        obstacle_map = generate_map(map_size, domain)
        # obstacle_map = generate_random_map(map_size, 16)
        print(obstacle_map.shape)
        simulation = SoundSimulator(map_size=map_size, 
                                    obstacle_map=obstacle_map, 
                                    duration= duration)
        simulation.run()
        spl = simulation.spl(integration_interval=duration)
        with open(f"{file_path}", "wb") as f:
            pickle.dump((obstacle_map, spl), f)

    Parallel(n_jobs=num_cores)(delayed(run_iteration)(i, 17) for i in range(nb_ex))

if __name__ == "__main__":
    launch()