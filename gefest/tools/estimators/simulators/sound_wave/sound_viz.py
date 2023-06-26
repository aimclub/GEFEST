import pickle
import matplotlib.pyplot as plt
import os



if __name__ == "__main__":
    # Choose your data path
    data_path = './'
    # Choose the simulation result to load
    file_url = os.path.join(data_path, "best_spl.pickle")
    with open(file_url, "rb") as f:
        spl_map = pickle.load(f)

    # Visualize the sound pressure map:
    fig, ax= plt.subplots(nrows=1, ncols=1, figsize=(8,4), sharey=True)
    spl_plt = ax.pcolormesh(spl_map, cmap="coolwarm")
    plt.colorbar(spl_plt, ax=ax)
    ax.set_title("Sound Pressure Level (dB)")
    # ax.pcolormesh(field_map, cmap="gray")
    # ax.set_title("Field of Propagation")
    fig.tight_layout()

    plt.show()