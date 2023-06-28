Quickstart
==========

GEFEST Framework quick start guide


How to install
--------------

Tested on python 3.7

.. code::

 pip install https://github.com/ITMO-NSS-team/GEFEST/archive/master.zip

How to design your own polygon in manual way
----------------------------------------------------

-  **Step 1**. Define estimator using loss function or simulator of the physical process.

Loss function for finding a polygon that seems like circle showed below. 

.. code:: python

 from types import SimpleNamespace
 import numpy as np 
 from gefest.tools.estimators.estimator import Estimator
 from gefest.core.geometry.geometry_2d import Geometry2D
 
 geometry = Geometry2D()


 def circle_loss(structure):
     #calculating area and length of designed polygon via GEFEST.gefest methods
 
     best = 999999
     for idx, polygon in enumerate(structure.polygons):
         area = geometry.get_square(polygon)
         length = geometry.get_length(polygon)
         centroid = geometry.get_centroid(polygon)

         if area == 0:
             continue
         #checking "area/length" ratio (equal 1 for circle)
         ratio = 4 * np.pi * area / length ** 2
         loss = 1 - ratio

         if loss < best:
             best = loss

     return best
 
 estimator = SimpleNamespace()
 estimator.estimate = circle_loss
 estimator = Estimator(estimator=estimator)

-  **Step 2**. Specify border coordinates of area where GEFEST will solve the task.

Put the *Domain* to *Setup()* class for creating a task variable.

.. code:: python
    
 from gefest.core.structure.domain import Domain
 from gefest.core.opt.setup import Setup

 domain = Domain(allowed_area=[
                   (0, 0), 
                   (0, 300), 
                   (300, 300),
                   (300, 0), 
                   (0, 0)
                ],

                # specify processing way
                geometry=geometry,

                # every designed polygon locates into Structure(),
                # these parameters determine number of polygons per Structure()
                max_poly_num=3,
                min_poly_num=1,

                # every designed polygon might сontain up to 20 points
                max_points_num=20,
                min_points_num=5,

                # designed polygons have closed borders
                is_closed=True)

 task_setup = Setup(domain=domain)

-  **Step 3** Create sampler to generate population in specified domain.

.. code:: python

 from gefest.tools.samplers.standard.standard import StandardSampler
 from gefest.tools.samplers.sampler import Sampler

 sampler = Sampler(StandardSampler(), domain)

-  **Step 4**. Create optimizer. 

.. code:: python

 from gefest.tools.optimizers.GA.GA import GA
 from gefest.tools.optimizers.optimizer import Optimizer
 from gefest.core.opt.operators.operators import default_operators
 from gefest.tools.samplers.standard.standard import StandardSampler

 n_steps = 25
 pop_size = 25

 params = GA.Params(pop_size=pop_size,
                    crossover_rate=0.6, 
                    mutation_rate=0.6,
                    mutation_value_rate=[])
 ga = GA(params=params,
         evolutionary_operators=default_operators(), 
         task_setup=task_setup)

 optimizer = Optimizer(ga)

-  **Step 5**. Run generative design. 

.. code:: python

 from gefest.core.opt.gen_design import design

 optimized_population = design(n_steps=n_steps,
                               pop_size=pop_size,
                               estimator=estimator,
                               sampler=sampler,
                               optimizer=optimizer)

-  **Step 6**. Create visualization of the best structure in designed population.

.. code:: python
    
 import pickle
 import matplotlib.pyplot as plt
 from gefest.core.viz.struct_vizualizer import StructVizualizer

 
 with open(f'HistoryFiles/performance_{n_steps-1}.pickle', 'rb') as f:
     performance = pickle.load(f)
 with open(f'HistoryFiles/population_{n_steps-1}.pickle', 'rb') as f:
     population = pickle.load(f)

 idx_of_best = performance.index(min(performance))
 visualiser = StructVizualizer(task_setup.domain)
 plt.figure(figsize=(7, 7))
 info = {'fitness': f'{performance[idx_of_best]:.2f}',
         'type': 'prediction'}
 visualiser.plot_structure(population[idx_of_best], info)
 plt.show()
 