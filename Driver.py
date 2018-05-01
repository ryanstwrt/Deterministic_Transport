import pandas as pd
import numpy as np


def step_characteristic(angular_flux_pos_lhs, scalar_flux_new, current_new, material_data, cell_width, mu_n):
    cell_average_angular_flux = np.zeros(128)
    angular_flux_pos_rhs = np.zeros(10)
    # sweep over the angles (starting with the positive angles, followed by the cells to solve for the angular flux
    for angle in reversed(range(len(mu_n[0]))):
        print(angular_flux_pos_rhs)
        print(angular_flux_pos_lhs)
        if angle > 4:
            for cell_num, cell in enumerate(material_cell):
                angular_flux_pos_rhs[angle] = angular_flux_pos_lhs[angle] * np.exp(-material_data.ix['total_fast', cell] \
                    * cell_width/mu_n[(0, angle)]) \
                    + Q[cell_num] / material_data.ix['total_fast', cell] \
                    * (1 - np.exp(-material_data.ix['total_fast', cell]*cell_width/mu_n[(0, angle)]))

                cell_average_angular_flux[cell_num] += (cell_width * Q[cell_num] -
                    mu_n[(0, angle)] * (angular_flux_pos_lhs[angle] - angular_flux_pos_rhs[angle])) \
                    / (material_data.ix['total_fast', cell] * cell_width)

                scalar_flux_new[(cell_num, 0)] += cell_average_angular_flux[cell_num] * mu_n[(1, angle)]

                current_new[(cell_num, 0)] += cell_average_angular_flux[cell_num] * mu_n[(1, angle)] * mu_n[(0, angle)]

                angular_flux_pos_lhs[angle] = angular_flux_pos_rhs[angle]
            # for mu < 0, we remove the - sign, as the change in delta x would create an additional negative to cancel
        else:
            for cell_num, cell in enumerate(reversed(material_cell)):
                rev_cell_num = 127 - cell_num
                angular_flux_pos_lhs[angle] = angular_flux_pos_rhs[angle] * np.exp(material_data.ix['total_fast', cell] \
                    * cell_width / mu_n[(0, angle)]) \
                    + Q[rev_cell_num] / material_data.ix['total_fast', cell] \
                    * (1 - np.exp(material_data.ix['total_fast', cell] * cell_width / mu_n[(0, angle)]))

                cell_average_angular_flux[rev_cell_num] += (cell_width * Q[rev_cell_num] -
                    mu_n[(0, angle)] * (angular_flux_pos_rhs[angle] - angular_flux_pos_lhs[angle])) \
                    / (material_data.ix['total_fast', cell] * cell_width)

                scalar_flux_new[(rev_cell_num, 0)] += cell_average_angular_flux[rev_cell_num] * mu_n[(1, angle)]

                current_new[(rev_cell_num, 0)] += cell_average_angular_flux[rev_cell_num] * mu_n[(1, angle)] * mu_n[(0, angle)]

                angular_flux_pos_rhs[angle] = angular_flux_pos_lhs[angle]

            #print(angular_flux_pos_rhs[angle], angular_flux_pos_lhs[angle], cell_average_angular_flux[cell_num])

    return cell_average_angular_flux

#
# Beginning of the Deterministic solver program.
#
#

# Create the cell array which designates the material present in each cell
mox_cell = ['water', 'water', 'MOX', 'MOX', 'MOX', 'MOX', 'water', 'water']
u_cell = ['water', 'water', 'U', 'U', 'U', 'U', 'water', 'water']
material_cell = []
cell_width = 0.15625
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

# Initialize k, the fission product, angular flux at 0, in the positive direction, scalar flux,
# convergence criteria, and the number of iterations for convergence

k_old = 1
fission_source_old = np.ones(128)
scalar_flux_old = np.zeros((128, 2))
scalar_flux_new = np.zeros((128, 2))
current_old = np.zeros((128, 2))
current_new = np.zeros((128, 2))
angular_flux_pos_lhs = np.zeros(10)
mu_n = np.array([[-0.973906528517, -0.865063366689, -0.679409568299, -0.433395394129, -0.148874338982,
                0.148874338982, 0.433395394129, 0.679409568299, 0.865063366689, 0.973906528517],
                [0.0666713444544, 0.149451349057, 0.219086362450, 0.269266719318, 0.295524224712,
                0.295524224712, 0.269266719318, 0.219086362450, 0.149451349057, 0.0666713444544]])

Q = np.zeros(128)
source = np.zeros(128)

k_conv = 0.00001
k_conv_test = 0.0001

fission_source_conv = 0.00001
fission_source_conv_test = 0.0001

source_convergence = 0.000001
source_convergence_test = 0.00001

num_power_iter = 0
num_source_iter_fast = 0
num_source_iter_thermal = 0

# Outermost loop which performs a power iteration over the problem
# This is also where we will solve for a new fission soruce/k value and
# where we check for convergence.
while k_conv < k_conv_test or fission_source_conv < fission_source_conv_test:
    # loop over the energy groups (we only have two energy groups:
    # 0 for fast and 1 for thermal
    for energy_group in [0, 1]:
        if energy_group == 0:
            for cell_num, cell in enumerate(material_cell):
                source[cell_num] = fission_source_old[cell_num] / k_old
        elif energy_group == 1:
            for cell_num, cell in enumerate(material_cell):
                source[cell_num] = material_data.ix['downscatter_fast', cell] * scalar_flux_new[cell_num]
        else:
            print("Error: Energy group not defined!")
            quit()

        # Inner loop to determine source convergence
        # start by determining the source term based on the energy and the Q term from above.
        while source_convergence < source_convergence_test:
            if energy_group == 0:
                for cell_num, cell in enumerate(material_cell):
                    Q[cell_num] = 0.5 * (material_data.ix['inscatter_fast', cell] * scalar_flux_old[cell_num, 0] + source[cell_num])
            elif energy_group == 1:
                for cell_num, cell in enumerate(material_cell):
                    Q[cell_num] = 0.5 * (material_data.ix['inscatter_thermal', cell] * scalar_flux_new[cell_num, 0] + source[cell_num])
            else:
                print("Error: Energy group not defined!")
                quit()

            # loop over the angles to determine the angular flux
            cell_average_angular_flux = step_characteristic(angular_flux_pos_lhs, scalar_flux_new, current_new, material_data, cell_width, mu_n)


            break
        break
    break

    num_power_iter += 1
    if num_power_iter > 1000:
        break
