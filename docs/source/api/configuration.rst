.. _configuration:

=============
Configuration
=============

To run the experiment, the user should define 3 components: the domain of the problem, 
objectives and hyperparameters of the algorithm. 
Since it is easiest way to define objective in a python script, GEFEST provides two options 
for experiments configuration: python config and mixed config.

**Python** configuration, as the name implies, is just python script, in which user shoud define 
task objectives, Domain and OptimizationParams objects (TunerParams also if required), and some domain 
pre-computaions.

**Mixed** configuration assumes that configuration objects (Domain, OptimizationParams, TunerParams) will be 
read from yaml file, while objective stay in python script.


OptimizationParams
~~~~~~~~~~~~~~~~~~

.. autopydantic_model:: gefest.core.configs.optimization_params.OptimizationParams

TunerParams
~~~~~~~~~~~

.. autopydantic_model:: gefest.core.configs.tuner_params.TunerParams

Domain
~~~~~~

.. autopydantic_model:: gefest.core.geometry.domain.Domain

Utils
~~~~~~

.. automodule:: gefest.core.configs.utils
   :members:
