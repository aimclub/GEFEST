Tutorial for beginners
======================

*Here is an example of breakwaters optimization. SWAN model need to be installed.
You can find our configuration in simulators folder in INPUT file.
It consist water area with two fixed breakwaters, bathymetry (specified in bathymetry folder) and land.
Output file (wave height at each point of the water are) located is in the 'r' folder.*

**1. Install last stable version of the GEFEST**

Tested on python 3.7

.. code-block:: python

    pip install https://github.com/ITMO-NSS-team/GEFEST/archive/master.zip

**2. Import needed libraries**

.. code-block:: python

    import timeit
    import pickle
    from types import SimpleNamespace

    import matplotlib.pyplot as plt
    import numpy as np

    from gefest.core.geometry.geometry_2d import Geometry2D
    from gefest.tools.estimators.simulators.swan.swan_interface import Swan
    from gefest.core.opt.setup import Setup
    from gefest.tools.optimizers.optimizer import Optimizer
    from gefest.core.structure.domain import Domain
    from gefest.core.structure.structure import Structure
    from gefest.core.opt.analytics import EvoAnalytics
    from gefest.core.viz.struct_vizualizer import StructVizualizer
    from gefest.tools.estimators.estimator import Estimator
    from gefest.tools.samplers.standard.standard import StandardSampler
    from gefest.tools.samplers.sampler import Sampler
    from gefest.tools.optimizers.SPEA2.SPEA2 import SPEA2
    from gefest.core.opt.operators.operators import default_operators
    from gefest.core.opt.gen_design import design
    from gefest.tools.estimators.simulators.swan import swan_model
    from gefest.core.configs.utils import create_prohibited

**3. Settings for domain to be researched**

*You have to set grid resolution on each axis (x and y), spatial grid
and coordinates of your target (or targets) for which you want to optimize height of wave*

.. code-block:: python

    grid_resolution_x = 83  # Number of points on x-axis
    grid_resolution_y = 58  # Number of points on y-axis
    coord_X = np.linspace(0, 2075, grid_resolution_x + 1)  # X coordinate for spatial grid
    coord_Y = np.linspace(0, 1450, grid_resolution_y + 1)  # Y coordinate for spatial grid
    X, Y = np.meshgrid(coord_X, coord_Y)  # Two dimensional spatial grid
    grid_target_X = 25  # X-grid coordinate of your target
    grid_target_Y = 25  # Y-grid coordinate of your target

**4. Create domain grid and coordinates of your targets**

*As you can see, in this exampe we consider only one target*

.. code-block:: python

    grid = [grid_resolution_x, grid_resolution_y]
    targets = [[grid_target_X, grid_target_Y]]

**5. Set up domain configuration that GEFEST requires for every task**

*Here we are working with open polygons*

.. code-block:: python

    fixed_area = [
    [[471, 5], [1335, 2], [1323, 214], [1361, 277], [1395, 327], [1459, 405], [1485, 490], [1449, 521], [1419, 558],
     [1375, 564], [1321, 469], [1248, 318], [1068, 272], [921, 225], [804, 231], [732, 266], [634, 331], [548, 405],
     [485, 482], [424, 569], [381, 625], [310, 662], [271, 684], [244, 706], [203, 708], [182, 647], [214, 638],
     [234, 632], [275, 588], [346, 475], [427, 366], [504, 240], [574, 166], [471, 5]],
    [[652, 1451], [580, 1335], [544, 1253], [468, 1190], [439, 1170], [395, 1150], [378, 1115], [438, 1070],
     [481, 1059], [508, 1076], [539, 1133], [554, 1183], [571, 1244], [594, 1305], [631, 1366], [657, 1414],
     [671, 1449], [652, 1451]]
    ]
    fixed_targets = [[coord_X[26], coord_Y[49]], [coord_X[37], coord_Y[11]], [coord_X[60], coord_Y[5]]]
    fixed_poly = [
        [[878, 1433], [829, 1303], [739, 1116], [619, 995], [447, 962], [306, 1004], [254, 1092], [241, 1184],
        [269, 1244],
        [291, 1338], [370, 1450]],
        [[878, 1433], [829, 1303], [739, 1116], [619, 995], [447, 962], [274, 868], [180, 813], [126, 717], [146, 580],
        [203, 480], [249, 469], [347, 471]]
    ]

    # Creation of prohibited structure consist of targets, lines, areas
    prohibited_structure = create_prohibited(
                                targets=fixed_targets, 
                                fixed_area=fixed_area,
                                fixed_points=fixed_poly
    )

    fixed_points = [[[1000, 50], [700, 600], [800, 800]], 
                    [[1900, 540], [1750, 1000]]]
    is_closed = False
    geometry = Geometry2D(is_closed=is_closed)
    domain = Domain(allowed_area=[(min(coord_X), min(coord_Y)),
                                (min(coord_X), max(coord_Y)),
                                (max(coord_X), max(coord_Y)),
                                (max(coord_X), min(coord_Y))],
                    geometry=geometry,
                    max_poly_num=3,
                    min_poly_num=1,
                    max_points_num=10,
                    min_points_num=2,
                    prohibited_area=prohibited_structure,
                    fixed_points=fixed_points,
                    is_closed=is_closed)
    task_setup = Setup(domain=domain)

**6. Preparation of the SWAN model**

*You need to set path to folder with swan.exe file. 
Our SWAN interface uses this path, domain grid, GEFEST domain and coordinates of targets*

.. code-block:: python

    path = swan_model.__file__[:-11]
    swan = Swan(path=path,
                targets=targets,
                grid=grid,
                domain=domain)
    max_length = np.linalg.norm(np.array([max(coord_X) - min(coord_X), max(coord_Y) - min(coord_Y)]))

**7. Definition of the cost function and estimator**

*There is a cost function as sum of cost of structure and wave height at the target points*

.. code-block:: python

    def cost(struct, estimator):
        max_length = np.linalg.norm(
            np.array([max(coord_X) - min(coord_X), 
                    max(coord_Y) - min(coord_Y)]))
        lengths = 0
        for poly in struct.polygons:
            if poly.id != 'fixed':
                length = geometry.get_length(poly)
                lengths += length

        _, hs = estimator.estimate(struct)
        loss = [hs, 2 * lengths / max_length]

        return loss
    
    estimator = Estimator(estimator=swan, loss=cost)

**8. Definition of the sampler** 

.. code-block:: python

    sampler = Sampler(sampler=StandardSampler(), domain=domain)

**9. Definition of the optimizer**

.. code-block:: python
    
    pop_size = 10
    n_steps = 10

    params = SPEA2.Params(pop_size=pop_size,
                          crossover_rate=0.6,
                          mutation_rate=0.6,
                          mutation_value_rate=[])

    spea2_optimizer = SPEA2(params=params,
                            evolutionary_operators=default_operators(),
                            task_setup=task_setup)

**10. Run optimization**

.. code-block:: python

    start = timeit.default_timer()
    optimized_pop = design(n_steps=n_steps,
                           pop_size=pop_size,
                           estimator=estimator,
                           sampler=sampler,
                           optimizer=spea2_optimizer)
    spend_time = timeit.default_timer() - start

**11. Vizualization of the result**

.. code-block:: python

    with open(f'HistoryFiles/performance_{n_steps-1}.pickle', 'rb') as f:
        performance = pickle.load(f)
    with open(f'HistoryFiles/population_{n_steps-1}.pickle', 'rb') as f:
        population = pickle.load(f)
            
    performance_sum = [sum(pair) for pair in performance]
    idx_of_best = performance_sum.index(min(performance_sum))

    visualiser = StructVizualizer(task_setup.domain)
    plt.figure(figsize=(7, 7))
        
    best = performance[idx_of_best]
    info_optimized = {
        'spend time': f'{spend_time:.2f}',
        'fitness': f'[{best[0]:.3f}, {best[1]:.3f}]',
        'type': 'prediction'}
    visualiser.plot_structure(
        [domain.prohibited_area, population[idx_of_best]], 
        ['prohibited structures', info_optimized], 
        [':', '-'])


