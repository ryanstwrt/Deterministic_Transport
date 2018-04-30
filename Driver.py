import pandas as pd
import numpy as np

# Create the cell array which designates the material present in each cell
mox_cell = ['water', 'water', 'MOX', 'MOX', 'MOX', 'MOX', 'water', 'water']
u_cell = ['water', 'water', 'U', 'U', 'U', 'U', 'water', 'water']
material_cell = []
for i in range(16):
    if i < 8:
        material_cell = material_cell + mox_cell
    else:
        material_cell = material_cell + u_cell

# Create the material specifications for each problem
materials_a = [[0.2, 0.2, 0.0, 0.0, 1.0, 0.6, 0.0, 0.0, 0.78, 0.0],
               [0.2, 0.2, 0.0, 0.0, 1.0, 0.2, 0.0, 0.0, 0.26, 0.0],
               [0.2, 0.17, 0.03, 0.0, 0.0, 1.1, 1.1, 0.0, 0.0, 0.0]]
materials_b = [[0.2, 0.175, 0.025, 0.0, 1.0, 1.2, 0.9, 0.0, 0.39, 0.0],
               [0.2, 0.175, 0.025, 0.0, 1.0, 1.0, 0.9, 0.0, 0.13, 0.0],
               [0.2, 0.17, 0.03, 0.0, 0.0, 1.1, 1.1, 0.0, 0.0, 0.0]]

# Create a data from for the material cross sections to be used for each material
material_data = pd.DataFrame(materials_a, columns=['total_fast', 'inscatter_fast', 'downscatter_fast',
                                                   'nusigmaf_fast', 'chi_fast', 'total_thermal', 'inscatter_thermal',
                                                   'downscatter_thermal', 'nusigmaf_thermal', 'chi_thermal'],
                             index=['MOX', 'U', 'water'])
material_data = material_data.T

# Create a two dictionaries for ease of access to the material data structure
material_type = {0: 'MOX', 1: 'U', 2: 'Water'}
interaction_type = {0:'total_fast', 1:'inscatter_fast', 2:'downscatter_fast', 3:'nusigmaf_fast', 4:'chi_fast',
                    5:'total_thermal', 6:'inscatter_thermal', 7:'downscatter_thermal', 8:'nusigmaf_thermal',
                    9:'chi_thermal'}
mat = 0
inter = 0
test = material_data.at[interaction_type[inter], material_type[mat]]
#print(test)

# Initialize k, the fission product, angular flux at 0, in the positive direction, scalar flux,
# convergence criteria, and the number of iterations for convergence

k_old = 1
fission_source_old = np.ones(128)
scalar_flux = np.zeros(128)
angular_flux_pos_lhs = np.zeros((5, 2))

k_conv = 0.00001
k_conv_test = 0.0001

fission_source_conv = 0.00001
fission_source_conv_test = 0.0001

source_convergence = 0.000001
source_convergence_test = 0.00001

num_power_iter = 0
num_source_iter_fast = 0
num_source_iter_thermal = 0

print(material_data.ix['chi_fast', 'water'])

# Outermost loop which performs a power iteration over the problem
# This is also where we will solve for a new fission soruce/k value and
# where we check for convergence.
while k_conv < k_conv_test or fission_source_conv < fission_source_conv_test:
    # loop over the energy groups (we only have two energy groups:
    # 0 for fast and 1 for thermal
    for energy_group in [0, 1]:
        if energy_group == 0:
            for cell_num, cell in enumerate(material_cell):
                source = material_data.ix['chi_fast', cell] * fission_source_old[cell_num] + \
                    material_data.ix['downscatter_fast', cell] * scalar_flux[cell_num] + \
                    material_data.ix['inscatter_fast', cell] * scalar_flux[cell_num]
        if energy_group == 0:
            for cell_num, cell in enumerate(material_cell):
                source = material_data.ix['chi_fast', cell] * fission_source_old[cell_num] + \
                    material_data.ix['downscatter_fast', cell] * scalar_flux[cell_num] + \
                    material_data.ix['inscatter_fast', cell] * scalar_flux[cell_num]
    print(source)
    break

#    source =
    num_power_iter += 1
    if num_power_iter > 1000:
        break

print(num_power_iter)




