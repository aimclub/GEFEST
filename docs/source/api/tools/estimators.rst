==========
Estimators
==========

The purpose of an estimator is to calculate the component of the fitness function 
that requires physical simulation, for example, using differential equations or surrogate models. 
The estimator should mainly implement an interface of interaction with external stimulatior, 
such as comsol multiphysics, SWAN or neural network models.

GEFEST provides estimators for several tasks.

Estimator interface
~~~~~~~~~

.. automodule:: gefest.tools.estimators.estimator
   :members:

SWAN estimator
~~~~~~~~~

.. automodule:: gefest.tools.estimators.simulators.swan.swan_interface
   :members:

Sound waves estimator
~~~~~~~~~

.. automodule:: gefest.tools.estimators.simulators.sound_wave.sound_interface
   :members:

Comsol estimator
~~~~~~~~~

.. automodule:: gefest.tools.estimators.simulators.comsol.comsol_interface
   :members: