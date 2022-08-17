Tutorial for beginners
======================

*Here is an example of breakwaters optimization. SWAN model need to be installed.
You can find our configuration in simulators folder in INPUT file.
It consist water area with two fixed breakwaters, bathymetry (specified in bathymetry folder) and land.
Output file (wave height at each point of the water are) located is in the 'r' folder.*

**1. Install last stable version of the GEFEST**

.. code-block:: python

    pip install https://github.com/ITMO-NSS-team/GEFEST/archive/master.zip

**2. Import needed libraries**

.. code-block:: python

    import timeit

    import matplotlib.pyplot as plt
    import numpy as np

    from gefest.core.geometry.geometry_2d import Geometry2D
    from gefest.simulators.swan.swan_interface import Swan
    from gefest.core.opt.optimize import optimize
    from gefest.core.opt.setup import Setup
    from gefest.core.structure.domain import Domain
    from gefest.core.structure.structure import Structure
    from gefest.core.opt.analytics import EvoAnalytics
    from gefest.core.viz.struct_vizualizer import StructVizualizer

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


    fixed_points = [[[1000, 50], [700, 600], [800, 800]], [[1900, 540], [1750, 1000]]]
    is_closed = False
    geometry = Geometry2D(is_closed=is_closed)
    domain = Domain(allowed_area=[(min(coord_X), min(coord_Y)),
                                (min(coord_X), max(coord_Y)),
                                (max(coord_X), max(coord_Y)),
                                (max(coord_X), min(coord_Y))],
                    geometry=geometry,
                    max_poly_num=1,
                    min_poly_num=1,
                    max_points_num=10,
                    min_points_num=2,
                    fixed_points=fixed_points,
                    is_closed=is_closed)
    task_setup = Setup(domain=domain)

**6. Preparation of the SWAN model**

*You need to set path to folder with swan.exe file. 
Our SWAN interface uses this path, domain grid, GEFEST domain and coordinates of targets*

.. code-block:: python

    path = '../../gefest/simulators/swan/swan_model/' #set your own path to SWAN in GEFEST on the local machine
    swan = Swan(path=path,
                targets=targets,
                grid=grid,
                domain=domain)

    max_length = np.linalg.norm(np.array([max(coord_X) - min(coord_X), max(coord_Y) - min(coord_Y)]))

**7. Definition of the cost function**

*There is a cost function as sum of cost of structure and wave height at the target points*

.. code-block:: python

    def cost(struct: Structure):
        lengths = 0
        for poly in struct.polygons:
            if poly.id != 'fixed':
                length = geometry.get_length(poly)
                lengths += length

        Z, hs = swan.evaluate(struct)
        loss = lengths / max_length + hs

        return loss

**8. Optimization stage**

.. code-block:: python

    start = timeit.default_timer()
    optimized_structure = optimize(task_setup=task_setup,
                                objective_function=cost,
                                pop_size=10,
                                max_gens=10)
    spend_time = timeit.default_timer() - start

**9. Vizualization of the result**

.. code-block:: python

    viz = StructVizualizer(domain)
    viz.plot_structure(optimized_structure)
