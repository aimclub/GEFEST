from loguru import logger
from tqdm import tqdm

from gefest.core.configs.utils import load_config
from gefest.core.viz.struct_vizualizer import GIFMaker
from gefest.tools.tuners.tuner import GolemTuner


def run_experiment(config_path: str):
    """Simple experiment runner."""
    opt_params = load_config(config_path)

    optimizer = opt_params.optimizer(opt_params)
    optimized_pop = optimizer.optimize()

    # Optimized pop visualization
    logger.info('Collecting plots of optimized structures...')
    gm = GIFMaker(domain=opt_params.domain)
    for st in tqdm(optimized_pop):
        gm.create_frame(st, {'Optimized': st.fitness})

    gm.make_gif('Optimized population', 500)

    if opt_params.tuner_cfg:
        tuner = GolemTuner(opt_params)
        tuned_individuals = tuner.tune(optimized_pop[: opt_params.tuner_cfg.tune_n_best])

        # Tuned structures visualization
        logger.info('Collecting plots of tuned structures...')
        gm = GIFMaker(domain=opt_params.domain)
        for st in tqdm(tuned_individuals):
            gm.create_frame(st, {'Tuned': st.fitness})

        gm.make_gif('Tuned individuals', 500)
