import sys
from functools import partial
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from gefest.core.configs.utils import load_config
from gefest.core.geometry import Point, Polygon, Structure
from gefest.core.utils.logger import LogDispatcher
from gefest.core.viz.struct_vizualizer import GIFMaker, StructVizualizer
from gefest.tools.tuners.tuner import GolemTuner
from gefest.tools.tuners.utils import percent_edge_variance

if __name__ == '__main__':

    opt_params = load_config(
        str(Path(__file__).parent) + '\cases\sound_waves\configuration\config_parallel.py'
    )

    optimized_struct = Structure(
        polygons=[
            Polygon(
                points=[
                    Point(x=94.8399485030149, y=20.40034861129441),
                    Point(x=85.5624968428982, y=21.059993710397926),
                    Point(x=79.00084052541479, y=21.862700852017394),
                    Point(x=76.06076964736816, y=22.476934992526946),
                    Point(x=74.90849793039871, y=29.131517380279682),
                    Point(x=74.61676812657737, y=32.94679344410641),
                    Point(x=74.56239446772611, y=33.69226990738204),
                    Point(x=74.24162218587203, y=40.70530130547375),
                    Point(x=87.01622361737279, y=40.96813944432962),
                    Point(x=95.64079625726104, y=40.78198185566877),
                    Point(x=95.58245342099814, y=25.459142743749343),
                    Point(x=95.57086138193173, y=25.035926422500204),
                    Point(x=95.46231197117629, y=21.849202076811995),
                    Point(x=94.8399485030149, y=20.40034861129441),
                ],
                fitness=[0.8770969580695486],
            )
        ]
    )

    best_structure = Structure(
        polygons=(
            Polygon(
                points=[
                    Point(x=95.0, y=20.0),
                    Point(x=75.0, y=20.0),
                    Point(x=75.0, y=40.0),
                    Point(x=95.0, y=40.0),
                    Point(x=95.0, y=20.0),
                ],
            ),
        ),
        fitness=[],
    )

    ## plot structs
    # sv = StructVizualizer(opt_params.domain)
    # sv.plot_structure(optimized_struct, opt_params.domain)
    # sv.plot_structure(best_structure, opt_params.domain)
    # from matplotlib import pyplot as plt
    # plt.show(block=True)

    tuner_names = ['sequential', 'simulataneous', 'iopt', 'optuna']
    for tuner_name in tuner_names:
        opt_params.tuner_cfg.tuner_type = tuner_name
        # opt_params.tuner_cfg.variacne_generator = partial(percent_edge_variance, percent=0.5)
        tuner = GolemTuner(opt_params)
        tuned_individuals = tuner.tune(optimized_struct)
        LogDispatcher(run_name=tuner_name).log_pop(tuned_individuals, '0')

    # # MP4
    # gm = GIFMaker(domain=opt_params.domain)
    # for st in tqdm([best_structure, optimized_struct]):
    #     gm.create_frame(st, {'Tuned': st.fitness})

    # gm.make_gif('Tuned individuals', 500)
