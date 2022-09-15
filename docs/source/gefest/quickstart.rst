Quickstart
==========

GEFEST Framework quick start guide

How to install
--------------
.. code::

 pip install https://github.com/ITMO-NSS-team/GEFEST/archive/master.zip

How to design your own polygon in manual way
----------------------------------------------------

-  **Step 1**. Create loss function or simulator of the physical process.

Loss function for finding a polygon that seems like circle showed below. 

.. code:: python

 import numpy as np
 from gefest.core.geometry.geometry_2d import Geometry2D

 geometry = Geometry2D()


 def circle_loss(polygon):
    #calculating area and length of designed polygon via GEFEST methods
    area = geometry.get_square(polygon)
    length = geometry.get_length(polygon)

    if area == 0:
        return None
    #checking "area/length" ratio (equal 1 for circle)
    ratio = 4 * np.pi * area / length ** 2

    loss = 1 - ratio
    return loss

-  **Step 2**. Specify border coordinates of area where GEFEST will solve the task.

Put the *Domain* to *Setup()* class for creating a task variable.

.. code:: python

 from gefest.core.structure.domain import Domain
 from gefest.core.opt.setup import Setup

 border_coord = [(0, 0), (0, 300), (300, 300),
                 (300, 0), (0, 0)]

 domain = Domain(allowed_area=border_coord,

                # specify processing way
                geometry=geometry,

                # every designed polygon locates into Structure(),
                # these parameters determine number of polygons per Structure()
                max_poly_num=7,
                min_poly_num=1,

                # every designed polygon might сontain up to 20 points
                max_points_num=20,
                min_points_num=5,

                # designed polygons have closed borders
                is_closed=True)

 task_setup = Setup(domain=domain)

-  **Step 3**. Create an optimized structure via *optimize* method. 

The structure will contain number of polygons that previously specified in *Domain*

.. code:: python

 from gefest.core.opt.optimize import optimize

 optimized_structure = optimize(task_setup=task_setup,
                                objective_function=circle_loss,

                                # Choosen population size and max number of generations
                                # for evolutionary optimization process
                                pop_size=100,
                                max_gens=220)

-  **Step 4**. Create visualization of designed structure.

.. code:: python
    
 from gefest.core.viz.struct_vizualizer import StructVizualizer

 visualiser = StructVizualizer(task_setup.domain)
 plt.figure(figsize=(7, 7))

 info = {'fitness': circle_loss(optimized_structure.polygons[0]),
         'type': 'prediction'}
 visualiser.plot_structure(optimized_structure, info)
 