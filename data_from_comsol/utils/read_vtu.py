import vtk

# The source file
file_name = "GEFEST\data_from_comsol\plot_data.vtu"

# Read the source file.
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(file_name)
reader.Update()  # Needed because of GetScalarRange
output = reader.GetOutput()
potential = output.GetPointData().GetArray("potential")
print()