==========
Optimizers
==========

GEFEST provides optimizers for single and multi objective tasks.


Optimizer interface
~~~~~~~~~~~~~~~~~~~

.. automodule:: gefest.tools.optimizers.optimizer
   :members:

Defalt single-objective GA
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: gefest.tools.optimizers.GA.GA
   :members:

SPEA2
~~~~~

.. automodule:: gefest.tools.optimizers.SPEA2.SPEA2
   :members:

Standard optimizer with GOLEM backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: gefest.tools.optimizers.golem_optimizer.standard
   :members:

Golem based surrogate optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: gefest.tools.optimizers.golem_optimizer.surrogate
   :members:

*Note*: GOLEM based optimizers provides single/multiobjective optimizations and other features, 
e.g. adaptive mutation strategies. For details see OptimizationParams class and GOLEM docs (https://thegolem.readthedocs.io/en/latest/). 

