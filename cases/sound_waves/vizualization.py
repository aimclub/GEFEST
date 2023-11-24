import matplotlib.pyplot as plt
import numpy as np

from cases.sound_waves.poly_from_point import poly_from_comsol_txt
from gefest.core.utils.functions import parse_structs, project_root

pr_root = project_root()
# p_s = parse_structs(f'{pr_root}/cases/sound_waves/logs/run_name_2023-10-10_16_25_55/tuned.log')[0]
tuned = parse_structs(f'{pr_root}/cases/sound_waves/logs/run_name_2023-10-12_12_27_12/tuned.log')[0]
tuned_2 = parse_structs(f'{pr_root}/cases/sound_waves/logs/run_name_2023-10-11_12_58_51/00100.log')[
    -1
]
tuned_1 = parse_structs(f'{pr_root}/cases/sound_waves/logs/run_name_2023-10-11_15_25_40/00100.log')[
    -1
]

grid_resolution_x = 300  # Number of points on x-axis
grid_resolution_y = 300
coord_X = np.linspace(20, 100, grid_resolution_x + 1)  # X coordinate for spatial grid
coord_Y = np.linspace(20, 100, grid_resolution_y + 1)
allowed_area = [
    (min(coord_X), min(coord_Y)),
    (min(coord_X), max(coord_Y)),
    (max(coord_X), max(coord_Y)),
    (max(coord_X), min(coord_Y)),
    (min(coord_X), min(coord_Y)),
]
path_to_init_figure = f'figures/bottom_square.txt'
best_structure = poly_from_comsol_txt(path_to_init_figure)
d_p = tuned[0].points
tuned_p = tuned_1[0].points
tuned_2_p = tuned_2[0].points
print([i[0] for i in allowed_area])
plt.plot([i[0] for i in allowed_area], [i[1] for i in allowed_area])
plt.plot([p.coords[0] for p in d_p], [p.coords[1] for p in d_p], label='tuned')
plt.plot(
    [p.coords[0] for p in tuned_p],
    [p.coords[1] for p in tuned_p],
    label=f'notTuned_1,fitness :{tuned_1.fitness[0]}',
)
plt.plot(
    [p.coords[0] for p in tuned_2_p],
    [p.coords[1] for p in tuned_2_p],
    label=f'notTuned_2,fitness :{tuned_2.fitness[0]}',
)
plt.plot(
    [x[0] for x in [i.coords for i in best_structure.polygons[0].points]],
    [x[1] for x in [i.coords for i in best_structure.polygons[0].points]],
    label='Init fig',
)
plt.legend()
plt.show()
print()
