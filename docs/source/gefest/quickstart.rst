.. _quickstart:

Quickstart
==========

GEFEST Framework quick start guide


How to install
--------------

Tested on python 3.9-3.10

.. code::

 pip install gefest

How to run  
----------

To run examples or custom config just use `run_experiments` and provide absolute path to config file.

You can use it as API:

.. code:: python

    from gefest.tools.run_experiments import run_experiments    

    path = 'C:\\Your\\Folder\\GEFEST\\cases\\synthetic\\circle\\multi_objective.py'
    run_experiments(path)

Or as CLI script:

.. code:: 

    python run_experiments.py C:\Path\To\Config\python_config.py

How to design experiment with GEFEST
------------------------------------

To run an experiment, you need to define several entities: 
    1. Objectives
    2. Domain
    3. TunerParams (if needed)
    4. OptimizationParams

They can be defined in the experiment startup script, 
or placed in a separate python file and loaded using `gefest.core.configs.utils.load_config` function.

All of them aggreagted into single `OptimizationParams` object.

Let's take a step-by-step look at how to do this.

-  **Step 0**. Import required GEFEST modules.

.. code:: python

    from gefest.core.configs.optimization_params import OptimizationParams
    from gefest.core.configs.tuner_params import TunerParams
    from gefest.core.geometry.datastructs.structure import Structure
    from gefest.core.geometry.domain import Domain
    from gefest.core.opt.objective.objective import Objective
    from gefest.tools.estimators.estimator import Estimator

-  **Step 1**. Define objectives using fitness function and simulator of the physical process if required.

Objective for finding a polygon that seems like circle showed below.

Inherit from Objective, pass domain and estimator through `__init__`.
Define logic of objective evaluation in `_evaluate` method, which returns float value.
If you want to solve multiobjective optimisation task, just define more objectives classes below.

.. code:: python

    import numpy as np

    class AreaLengthRatio(Objective):
        """Area length ratio metric."""
        def __init__(self, domain: Domain, estimator: Estimator = None) -> None:
            super().__init__(domain, estimator)

        def _evaluate(self, ind: Structure) -> float:

            num = 3
            num_polys = len(ind.polygons)
            loss = 0
            for poly in ind.polygons:
                area = self.domain.geometry.get_square(poly)
                length = self.domain.geometry.get_length(poly)
                if area == 0:
                    ratio = None
                else:
                    ratio = 1 - 4 * np.pi * area / length ** 2

                loss += ratio

            loss = loss + 20 * abs(num_polys - num)
            return loss



-  **Step 2**. Define task domain.

Domain describes geometric constraints for individuals.

.. code:: python
    
    domain_cfg = Domain(
        allowed_area=[
            [0, 0],
            [0, 300],
            [300, 300],
            [300, 0],
            [0, 0],
        ],
        min_poly_num=1,
        max_poly_num=4,
        min_points_num=3,
        max_points_num=15,
        polygon_side=0.0001,
        min_dist_from_boundary=0.0001,
        geometry_is_convex=True,
        geometry_is_closed=True,
    )

-  **Step 3** Create sampler to generate population in specified domain.

By default, the standard sampler is used.
You can select another sampler or define custom for spicific task.
How to define your own sampler described in the tutorials section of the documentation.

-  **Step 4**. Define tuner configuraton.

You can tune coordinates of optimized structures points to achieve better objective metric using GOLEM tuners.
To use this feature define `TunerParams` configuration.

.. code:: python

    tuner_cfg = TunerParams(
        tuner_type='optuna',
        n_steps_tune=10,
        hyperopt_dist='uniform',
        verbose=True,
        timeout_minutes=60,
    )

-  **Step 5**. Define OptimisationParams config.

To know more about configuration options see :ref:`configuration` section of API reference. 

.. code:: python

    opt_params = OptimizationParams(
        optimizer='gefest_ga',
        domain=domain_cfg,
        tuner_cfg=tuner_cfg,
        n_steps=50,
        pop_size=50,
        postprocess_attempts=3,
        mutation_prob=0.6,
        crossover_prob=0.6,
        mutations=[
            'rotate_poly',
            'resize_poly',
            'add_point',
            'drop_point',
            'add_poly',
            'drop_poly',
            'pos_change_point',
        ],
        selector='tournament_selection',
        mutation_each_prob=[0.125, 0.125, 0.15, 0.35, 0.00, 0.00, 0.25],
        crossovers=[
            'polygon_level',
            'structure_level',
        ],
        crossover_each_prob=[0.0, 1.0],
        postprocess_rules=[
            'not_out_of_bounds',
            'valid_polygon_geom',
            'not_self_intersects',
            'not_too_close_polygons',
            'not_too_close_points',
        ],
        extra=5,
        n_jobs=-1,
        log_dir='logs',
        run_name='run_name',
        golem_keep_histoy=False,
        golem_genetic_scheme_type='steady_state',
        golem_surrogate_each_n_gen=5,
        objectives=[
            AreaLengthRatio(domain_cfg),
        ],
    )

-  **Step 5**. Run generative design and results visualisation. 

Now you can run the optimization as it was described above in *How to run* section of this tutorial.
Let's take a look at code in `run_experiments.py` script.

.. code:: python

    from loguru import logger
    from tqdm import tqdm

    from gefest.core.configs.utils import load_config
    from gefest.core.viz.struct_vizualizer import GIFMaker
    from gefest.tools.tuners.tuner import GolemTuner

    config_path = 'your/config/absolute/path.py'

    # Load config
    opt_params = load_config(
        config_path
    )

    # Initialize and run optimizer
    optimizer = opt_params.optimizer(opt_params)
    optimized_pop = optimizer.optimize()

    # Optimized pop visualization
    logger.info('Collecting plots of optimized structures...')
    # GIFMaker object creates mp4 from optimized structures plots
    gm = GIFMaker(domain=opt_params.domain)
    for st in tqdm(optimized_pop):
        gm.create_frame(st, {'Optimized': st.fitness})

    gm.make_gif('Optimized population', 500)

    # Run tuning if it defined in cofiguration
    if opt_params.tuner_cfg:
        tuner = GolemTuner(opt_params)
        tuned_individuals = tuner.tune(optimized_pop[: opt_params.tuner_cfg.tune_n_best])

        # Tuned structures visualization
        logger.info('Collecting plots of tuned structures...')
        gm = GIFMaker(domain=opt_params.domain)
        for st in tqdm(tuned_individuals):
            gm.create_frame(st, {'Tuned': st.fitness})

        gm.make_gif('Tuned individuals', 500)

To plot spicific structures with matplotlib.pyplot see :ref:`structvizualizer` examples. 
