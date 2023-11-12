.. role:: raw-html-m2r(raw)
   :format: html

Optimizers
==========

To solve the optimization problem, 4 optimizers are available in GEFEST - 2 native and 2 based on GOLEM.
All of them have a single interface and can be imported from ``gefest.tools.optimizers``.

.. list-table:: Optimizers comparation
   :header-rows: 1

   * - 
     - ``BaseGA``
     - ``SPEA2``
     - ``StandardOptimizer``
     - ``SurrogateOptimizer``
   * - **Backend**
     - GEFEST
     - GEFEST
     - GOLEM
     - GOLEM
   * - **Muti objective**
     - :raw-html-m2r:`<code style="background:red;color:white;font-weight:bold">No</code>`
     - :raw-html-m2r:`<code style="background:green;color:white;font-weight:bold">Yes</code>`
     - :raw-html-m2r:`<code style="background:green;color:white;font-weight:bold">Yes</code>`
     - :raw-html-m2r:`<code style="background:green;color:white;font-weight:bold">Yes</code>`
   * - **Evolutionary schemes**
     - :raw-html-m2r:`<code style="background:red;color:white;font-weight:bold">No</code>`
     - :raw-html-m2r:`<code style="background:red;color:white;font-weight:bold">No</code>`
     - :raw-html-m2r:`<code style="background:green;color:white;font-weight:bold">Yes</code>`
     - :raw-html-m2r:`<code style="background:green;color:white;font-weight:bold">Yes</code>`
   * - **Adaptive mutation strategies**
     - :raw-html-m2r:`<code style="background:red;color:white;font-weight:bold">No</code>`
     - :raw-html-m2r:`<code style="background:red;color:white;font-weight:bold">No</code>`
     - :raw-html-m2r:`<code style="background:green;color:white;font-weight:bold">Yes</code>`
     - :raw-html-m2r:`<code style="background:green;color:white;font-weight:bold">Yes</code>`
   * - **Surrogate optimization**
     - :raw-html-m2r:`<code style="background:red;color:white;font-weight:bold">No</code>`
     - :raw-html-m2r:`<code style="background:red;color:white;font-weight:bold">No</code>`
     - :raw-html-m2r:`<code style="background:red;color:white;font-weight:bold">No</code>`
     - :raw-html-m2r:`<code style="background:green;color:white;font-weight:bold">Yes</code>`

Some details
------------

``BaseGA`` implements the base genetic algorithm, that performs generation of the initial population,
crossover and mutation operations, fitness estimation and selection.
Each of the steps is encapsulated in a separate executor, which allows you to change the logic of individual steps.
Thus, BaseGA essentially only implements the sequence of their call. 

``SPEA2`` implements Strength Pareto Evolutionary Algorithm 2 for multiobjective optimization.

``StandardOptimizer`` is a wrapper for GOLEM`s ``EvoGraphOptimizer`` optimizer.
It allows to select different evolutionary schemes, adaptive mutation strategies and some other features.
To use multiobjective optimization set `golem_selection_type` in ``OptimizationParams`` config to 'spea2'.

``SurrogateOptimizer`` is the extension of ``StandardOptimizer`` with the ability 
to use a surrogate model to evaluate fitness along with the main estimator.

How to run
----------

Easiest way to run optimizer described in :ref:`quickstart`.

If you want to get some more control you can do it in code by import corresponding classes:

.. code-block:: python

   from gefest.tools.optimizers BaseGA, SPEA2, StandardOptimizer, SurrogateOptimizer

   from gefest.core.configs.optimization_params import OptimizationParams
   from gefest.core.geometry.datastructs.structure import Structure

   opt_params: OptimizationParams = ...
   
   optimizer = BaseGA(opt_params)
   optimized_population = optimizer.optimize(n_steps=42)
