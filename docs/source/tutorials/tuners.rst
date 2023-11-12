.. role:: raw-html-m2r(raw)
   :format: html

Tuners
======

GEFEST provides api for 4 tuners from `GOLEM <https://thegolem.readthedocs.io/en/latest/api/tuning.html>`_\ :

.. list-table:: Tuners comparation
   :header-rows: 1

   * - 
     - iopt
     - optuna
     - sequential
     - simultaneous
   * - **Nodes bypass**
     - simultaneous
     - simultaneous
     - sequential
     - simultaneous
   * - ****Backend**
     - `iOpt <https://github.com/aimclub/iOpt>`_
     - `Optuna <https://github.com/optuna/optuna>`_
     - `Hyperopt <https://github.com/hyperopt/hyperopt>`_
     - `Hyperopt <https://github.com/hyperopt/hyperopt>`_
   * - **Multi objective**
     - :raw-html-m2r:`<code style="background:red;color:white;font-weight:bold">No</code>`
     - :raw-html-m2r:`<code style="background:green;color:white;font-weight:bold">Yes</code>`
     - :raw-html-m2r:`<code style="background:red;color:white;font-weight:bold">No</code>`
     - :raw-html-m2r:`<code style="background:red;color:white;font-weight:bold">No</code>`


How to tune
-----------

To initialize tuner and run tuning, similarly to the optimizers, you need an ``OptimizationParams`` 
with defined ``TunerParams`` and also one or more ``Structure`` objects.

More details about ``OptimizationParams`` you can find in `cases` and `API referece` sections of this documentation.

Here we will take a closer look at several ``TunerParams`` attributes which may be unclear.

.. code-block:: python

   from gefest.tools.tuners.utils import verage_edge_variance

   tuner_cfg = TunerParams(
       tuner_type='optuna',
       n_steps_tune=10,
       hyperopt_dist='uniform',
       variance_generator=verage_edge_variance,
       verbose=True,
       timeout_minutes=60,
   )


* 
  ``tuner_type`` is responsible for which tuner will be used.

* 
  ``hyperopt_dist`` is the type of distribution from which random values will be taken during tuning. Available values are names of `hpyeropt hp module finctions <https://github.com/hyperopt/hyperopt/blob/master/hyperopt/hp.py>`_.

* 
  ``variance_generator`` is function that generates bounds of intervals from which random values should pe picked for all components of all point in structure. If normal distribution set they will be automatically converted into means and varicances.

``verage_edge_variance`` function setes variance to 50% of average edge length for each polygon. This solution can be much more "greedy" than necessary, which can lead to many invalid intermediate variants during tuning. To improve fitness in fewer tuning steps, it is worth creating variance generation functions for selecting smaller intervals based on the conditions of a specific task.

Now that the ``OptimizationParams`` have been defined and some structures have been created, we can run tuning with couple lines of code:

.. code-block:: python

   from gefest.core.configs.optimization_params import OptimizationParams
   from gefest.core.geometry.datastructs.structure import Structure
   from gefest.tools.tuners.tuner import GolemTuner

   structs: list[Structure] | Structure = ...
   opt_params: OptimizationParams = ...

   tuner = GolemTuner(opt_params)
   tuned_structs = tuner.tune(structs)
