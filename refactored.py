from gefest.core.configs.utils import load_config
from gefest.core.viz.struct_vizualizer import GIFMaker
from gefest.tools.optimizers.GA.GA import BaseGA
from gefest.tools.tuners.tuner import GolemTuner

if __name__ == '__main__':

    opt_params = load_config(
        'F:\\Git_Repositories\\gef_ref\\GEFEST\\gefest_config.py',
    )

    optimizer = BaseGA(opt_params)
    optimized_pop = optimizer.optimize()

    tuner = GolemTuner(opt_params)
    tuned_individuals = tuner.tune(optimized_pop[0])

    # visualization
    gm = GIFMaker(domain=opt_params.domain)
    for st in optimized_pop:
        gm.create_frame(st, {'Optimized': st.fitness})

    gm.make_gif('diag_test', 500)
