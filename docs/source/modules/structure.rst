~~~~~~~~~
Structure
~~~~~~~~~

There is a description of structural elements that allow to create
geometrical objects. Any geometrical figure might be created via set of
3D :obj:`Point`, :obj:`Polygon` and :obj:`Structure` objects.  

class Point
^^^^^^^^^^^
.. autoclass:: gefest.core.structure.point.Point
   :members:
   :no-undoc-members:

class Polygon
^^^^^^^^^^^
.. autoclass:: gefest.core.structure.polygon.Polygon

class Structure
^^^^^^^^^^^
.. autoclass:: gefest.core.structure.structure.Structure
   :members:
   :no-undoc-members:
.. image:: https://i.ibb.co/1q0CVNJ/structure-plot.png
   :alt: Output of Examples

~~~~~~~~~
Domain
~~~~~~~~~

There is a description of ...

class Domain
^^^^^^^^^^^

.. autoclass:: gefest.core.structure.domain.Domain
   :members:
   :no-undoc-members:


Geometry 2D
^^^^^^^^^^^

There is an object included set of methods for processing 2D
geometrical figures. The most of implemented methods based on
the **shapely** library.


.. autoclass:: gefest.core.geometry.geometry_2d.Geometry2D
   :members:
   :no-undoc-members:


Validation
^^^^^^^^^^^

There is one of the processing layer of optimization. 


.. automethod:: gefest.core.geom.validation