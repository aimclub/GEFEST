.. role:: raw-html-m2r(raw)
   :format: html

Optimisation
============

Optimisers summary
------------------

To solve the optimisation problems 4 optimisers are available in GEFEST - 1 native and 2 based on GOLEM.
All of them have a single interface and can be imported from ``gefest.tools.optimizers``.

.. list-table:: Optimizers comparation
   :header-rows: 1

   * - 
     - ``BaseGA``
     - ``StandardOptimizer``
     - ``SurrogateOptimizer``
   * - **Backend**
     - GEFEST
     - GOLEM
     - GOLEM
   * - **Muti objective**
     - :raw-html-m2r:`<code style="background:green;color:white;font-weight:bold">Yes</code>`
     - :raw-html-m2r:`<code style="background:green;color:white;font-weight:bold">Yes</code>`
     - :raw-html-m2r:`<code style="background:green;color:white;font-weight:bold">Yes</code>`
   * - **Evolutionary schemes**
     - :raw-html-m2r:`<code style="background:red;color:white;font-weight:bold">No</code>`
     - :raw-html-m2r:`<code style="background:green;color:white;font-weight:bold">Yes</code>`
     - :raw-html-m2r:`<code style="background:green;color:white;font-weight:bold">Yes</code>`
   * - **Adaptive mutation strategies**
     - :raw-html-m2r:`<code style="background:red;color:white;font-weight:bold">No</code>`
     - :raw-html-m2r:`<code style="background:green;color:white;font-weight:bold">Yes</code>`
     - :raw-html-m2r:`<code style="background:green;color:white;font-weight:bold">Yes</code>`
   * - **Surrogate optimisation**
     - :raw-html-m2r:`<code style="background:red;color:white;font-weight:bold">No</code>`
     - :raw-html-m2r:`<code style="background:red;color:white;font-weight:bold">No</code>`
     - :raw-html-m2r:`<code style="background:green;color:white;font-weight:bold">Yes</code>`


``BaseGA`` implements the base genetic algorithm, that performs sampling of the initial population,
crossover and mutation operations, fitness estimation and selection.
Each of the steps is encapsulated in a separate executor, which allows you to change the logic of individual steps.
Thus, BaseGA essentially only implements the sequence of their call.

``StandardOptimizer`` is a wrapper for GOLEM`s ``EvoGraphOptimizer``.
It allows to use different evolutionary schemes, adaptive mutation strategies and some other features.
To use multiobjective optimisation set `golem_selection_type` in ``OptimizationParams`` config to 'spea2'.

``SurrogateOptimizer`` is the extension of ``StandardOptimizer`` with the ability 
to use a surrogate model to evaluate fitness along with the main estimator.

Selectors summary
-----------------

``OptimizationParams`` have 3 parameters to configure selection strategy:
 * ``golem_selection_type`` defines which selection function will be used by GOLEM optimisers. Available values: 'spea2' for multi objective and 'tournament' for single objective problems.
 * ``selector`` defines which selection function will be used by GEFEST for single objective problems and also for multi objective fitnesses if it possible. Available values: 'tournament_selection' and 'roulette_selection'.
 * ``multiobjective_selector`` defines which selection function will be used by GEFEST for multiobjective problems. Available values: 'spea2' and 'moead'.

How to optimise
---------------

Easiest way to run optimiser described in :ref:`quickstart`.

If you want to get some more control you can do it in your code:

.. code-block:: python

   from gefest.tools.optimizers BaseGA, StandardOptimizer, SurrogateOptimizer

   from gefest.core.configs.optimization_params import OptimizationParams
   from gefest.core.geometry.datastructs.structure import Structure

   opt_params: OptimizationParams = ...
   
   optimizer = BaseGA(opt_params)
   optimized_population = optimizer.optimize(n_steps=42)

By default initial population generates automatically with sampler from `opt_params`.
It also can be provided as optional argument for optimiser constructor.
