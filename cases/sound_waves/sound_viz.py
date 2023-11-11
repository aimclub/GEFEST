
import matplotlib.pyplot as plt

from gefest.core.configs.utils import load_config
from gefest.tools.utils import load_pickle

init_path = 'best_structure.pickle'
optimized_path = 'optimized_structure.pickle'

if __name__ == '__main__':

    opt_params = load_config('F:\\Git_Repositories\\gef_ref\\GEFEST\\cases\\sound_waves\\configuration\\config.py')    

    init_structure = load_pickle(init_path)
    optimized_archive = load_pickle(optimized_path)
    optimized_structure = optimized_archive[0]

    sound_sim = opt_params.objectives[0].estimator

    spl_0 = sound_sim.estimate(init_structure)
    spl_1 = sound_sim.estimate(optimized_structure)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), sharey=True)
    spl_plt_0 = ax1.pcolormesh(spl_0, cmap='coolwarm')
    plt.colorbar(spl_plt_0, ax=ax1)

    spl_plt_1 = ax2.pcolormesh(spl_1, cmap='coolwarm')
    plt.colorbar(spl_plt_1, ax=ax2)

    plt.show()
