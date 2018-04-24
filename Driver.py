import pandas as pd
import numpy as np

# Create the cell array which designates the material present in each cell
cell_array_128 = np.linspace(0,20., num=128)
material_cell = np.zeros(len(cell_array_128))
material_cell = material_cell.reshape(16,8)
material_cell[:, :] = 2
material_cell[:8, 2:6] = 0
material_cell[8:, 2:6] = 1
material_cell = material_cell.reshape(1, 128)
geometry_cells = pd.DataFrame(material_cell, index=['Material'])

# Create the material specifications for each problem
materials_a = [[0.2, 0.2, 0.0, 0.0, 1.0, 0.6, 0.0, 0.0, 0.78, 0.0],
               [0.2, 0.2, 0.0, 0.0, 1.0, 0.2, 0.0, 0.0, 0.26, 0.0],
               [0.2, 0.17, 0.03, 0.0, 0.0, 1.1, 1.1, 0.0, 0.0, 0.0]]
materials_b = [[0.2, 0.175, 0.025, 0.0, 1.0, 1.2, 0.9, 0.0, 0.39, 0.0],
               [0.2, 0.175, 0.25, 0.0, 1.0, 1.2, 0.9, 0.0, 0.13, 0.0],
               [0.2, 0.17, 0.03, 0.0, 1.1, 1.1, 0.0, 0.0, 0.0, 0.0]]

# Create a data from for the material cross sections to be used for each material
material_data = pd.DataFrame(materials_b, columns=['total_fast', 'inscatter_fast', 'downscatter_fast',
                                                   'nusigmaf_fast', 'chi_fast', 'total_thermal', 'inscatter_thermal',
                                                   'downscatter_thermal', 'nusigmaf_thermal', 'chi_thermal'],
                             index=['MOX', 'U', 'Water'])
material_data = material_data.T

# Create a two dictionaries for ease of access to the material data structure
material_type = {0:'MOX',1:'U',2:'Water'}
interaction_type = {0:'total_fast', 1:'inscatter_fast', 2:'downscatter_fast', 3:'nusigmaf_fast', 4:'chi_fast',
                    5:'total_thermal', 6:'inscatter_thermal', 7:'downscatter_thermal', 8:'nusigmaf_thermal',
                    9:'chi_thermal'}
mat = 0
inte = 0
test = material_data.at[interaction_type[inte], material_type[mat]]
print(test)
a new line