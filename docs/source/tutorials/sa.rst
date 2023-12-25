Sensitivity analysis
====================

SA
--

Use SA to run local search near an optimized structure.


.. code-block:: python

   from gefest.core.configs.optimization_params import OptimizationParams
   from gefest.core.geometry.datastructs.structure import Structure
   from gefest.tools.tuners.sa import SensitivityAnalysis, report_viz
   from matplotlib import pyplot as plt

   structure: list[Structure] | Structure = ...
   opt_params: OptimizationParams = ...

   sa = SensitivityAnalysis([optimized_struct], opt_params)
   res = sa.analysis()

   # plot analysis history
   report_viz(res)


   # plot initial and best structure
   sv = StructVizualizer()
   sv.plot_structure(res[1][0])
   sv.plot_structure(res[1][1])

   plt.show(block=True)

   # animated history of structures during SA
   gm = GIFMaker()
   for st in tqdm(res[1]):
       gm.create_frame(st, {'sa': st.fitness})

   gm.make_gif('sa individuals', 500)
