import pickle
import matplotlib.pyplot as plt
from cases.main_conf import opt_params
from cases.sound_waves.configuration import sound_domain
from gefest.tools.estimators.simulators.sound_wave.sound_interface import SoundSimulator


def upload_file(path: str):
    with open(path, "rb") as f:
        file = pickle.load(f)
        f.close()
    return file


init_path = "best_structure.pickle"
optimized_path = "optimized_structure.pickle"

if __name__ == "__main__":
    domain, _ = sound_domain.configurate_domain(
        poly_num=opt_params.n_polys,
        points_num=opt_params.n_points,
        is_closed=opt_params.is_closed,
    )

    init_structure = upload_file(init_path)
    optimized_archive = upload_file(optimized_path)
    optimized_structure = optimized_archive[0]

    sound = SoundSimulator(domain)

    spl_0 = sound.estimate(init_structure)
    spl_1 = sound.estimate(optimized_structure)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), sharey=True)
    spl_plt_0 = ax1.pcolormesh(spl_0, cmap="coolwarm")
    plt.colorbar(spl_plt_0, ax=ax1)

    spl_plt_1 = ax2.pcolormesh(spl_1, cmap="coolwarm")
    plt.colorbar(spl_plt_1, ax=ax2)

    plt.show()
