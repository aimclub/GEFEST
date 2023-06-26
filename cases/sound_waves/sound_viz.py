import pickle
import matplotlib.pyplot as plt
from cases.main_conf import opt_params
from cases.sound_waves.configuration import sound_domain
from gefest.tools.estimators.simulators.sound_wave.sound_interface import SoundSimulator


domain, _ = sound_domain.configurate_domain(
    poly_num=opt_params.n_polys,
    points_num=opt_params.n_points,
    is_closed=opt_params.is_closed,
)

with open("best_structure.pickle", "rb") as f:
    structure_0 = pickle.load(f)
    f.close()

with open("optimized_structure.pickle", "rb") as f:
    archive_1 = pickle.load(f)
    f.close()

structure_1 = archive_1[0]

sound = SoundSimulator(domain)


if __name__ == "__main__":
    spl_0 = sound.estimate(structure_0)
    spl_1 = sound.estimate(structure_1)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 4), sharey=True)
    spl_plt_0 = ax1.pcolormesh(spl_0, cmap="coolwarm")
    plt.colorbar(spl_plt_0, ax=ax1)

    spl_plt_1 = ax2.pcolormesh(spl_1, cmap="coolwarm")
    plt.colorbar(spl_plt_1, ax=ax2)

    plt.show()
