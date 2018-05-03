import pandas as pd
import numpy as np
import matplotlib.pyplot as mpl


def step_characteristic(angular_flux_pos_lhs, angular_flux_pos_rhs, scalar_flux_new, current_new, material_data,
                             cell_width, mu_n, nrg):
    if nrg == 0:
        total = 'total_fast'
    elif nrg == 1:
        total = 'total_thermal'

    # sweep over the angles (starting with the positive angles, followed by the cells to solve for the angular flux
    for angle in reversed(range(len(mu_n[0]))):
        if angle > 4:
            for cell_num, cell in enumerate(material_cell):
                tau = material_data.ix[total, cell] * cell_width / abs(mu_n[(0, angle)])

                angular_flux_pos_rhs[(angle, nrg)] = angular_flux_pos_lhs[(angle, nrg)] * np.exp(-tau) + (Q[cell_num]
                                                    / material_data.ix[total, cell]) * (1 - np.exp(-tau))

                cell_average_angular_flux = (cell_width * Q[cell_num] - mu_n[(0, angle)] *
                                            (angular_flux_pos_rhs[(angle, nrg)] - angular_flux_pos_lhs[(angle, nrg)])) \
                                            / (material_data.ix[total, cell] * cell_width)

                scalar_flux_new[(cell_num, nrg)] += cell_average_angular_flux * mu_n[(1, angle)]
                current_new[(cell_num, nrg)] += cell_average_angular_flux * mu_n[(1, angle)] * mu_n[(0, angle)]
                angular_flux_pos_lhs[(angle, nrg)] = angular_flux_pos_rhs[(angle, nrg)]

            # for mu < 0, we remove the - sign, as the change in delta x would create an additional negative to cancel
        else:
            for cell_num, cell in enumerate(reversed(material_cell)):
                rev_cell_num = 127 - cell_num

                tau = material_data.ix[total, cell] * cell_width / abs(mu_n[(0, angle)])

                # Special case to flip the neutrons from the first sweep backwards.
                if rev_cell_num == 127:
                    angular_flux_pos_lhs[(angle, nrg)] = angular_flux_pos_rhs[(9 - angle, nrg)] * np.exp(-tau) \
                                                       + Q[rev_cell_num] / material_data.ix[total, cell] \
                                                       * (1 - np.exp(-tau))

                    cell_average_angular_flux = (cell_width * Q[rev_cell_num] - mu_n[(0, angle)]
                                                 * (angular_flux_pos_rhs[(9 - angle, nrg)]
                                                    - angular_flux_pos_lhs[(angle, nrg)])) \
                                                 / (material_data.ix[total, cell] * cell_width)
                else:
                    angular_flux_pos_lhs[(angle, nrg)] = angular_flux_pos_rhs[(angle, nrg)] * np.exp(-tau) \
                                                       + Q[rev_cell_num] / material_data.ix[total, cell] * (
                                                               1 - np.exp(-tau))
                    cell_average_angular_flux = (cell_width * Q[rev_cell_num] - mu_n[(0, angle)]
                                                 * (angular_flux_pos_rhs[(angle, nrg)]
                                                    - angular_flux_pos_lhs[(angle, nrg)])) \
                                                / (material_data.ix[total, cell] * cell_width)

                scalar_flux_new[(rev_cell_num, nrg)] += cell_average_angular_flux * mu_n[(1, angle)]
                current_new[(rev_cell_num, nrg)] += cell_average_angular_flux * mu_n[(1, angle)] * mu_n[(0, angle)]
                angular_flux_pos_rhs[(angle, nrg)] = angular_flux_pos_lhs[(angle, nrg)]

    return

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
materials_a = [[0.2, 0.2, 0.0, 0.0, 1.0, 0.6, 0.0, 0.0, 0.90, 0.0],
               [0.2, 0.2, 0.0, 0.0, 1.0, 0.2, 0.0, 0.0, 0.30, 0.0],
               [0.2, 0.17, 0.03, 0.0, 0.0, 1.1, 1.1, 0.0, 0.0, 0.0]]
materials_b = [[0.2, 0.185, 0.015, 0.0, 1.0, 1.2, 0.9, 0.0, 0.45, 0.0],
               [0.2, 0.185, 0.015, 0.0, 1.0, 1.0, 0.9, 0.0, 0.15, 0.0],
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
k_new = np.zeros(50)
k_new[0] = 1
fission_source_old = np.ones(128)
fission_source_new = np.ones(128)
scalar_flux_old = np.zeros((128, 2))
current_old = np.zeros((128, 2))
angular_flux_pos_lhs = np.zeros((10, 2))
angular_flux_pos_rhs = np.zeros((10, 2))
mu_n = np.array([[-0.973906528517, -0.865063366689, -0.679409568299, -0.433395394129, -0.148874338982,
                  0.148874338982, 0.433395394129, 0.679409568299, 0.865063366689, 0.973906528517],
                 [0.0666713444544, 0.149451349057, 0.219086362450, 0.269266719318, 0.295524224712,
                  0.295524224712, 0.269266719318, 0.219086362450, 0.149451349057, 0.0666713444544]])

Q = np.zeros(128)
source = np.zeros(128)

k_conv = 0.00001
fission_source_conv = 0.00001
source_convergence = 0.000001
fs_convergence = 1
k_convergence = 1

fast_source_convergence = 0
thermal_source_convergence = 0
num_power_iter = 0
# Outermost loop which performs a power iteration over the problem
# This is also where we will solve for a new fission soruce/k value and
# where we check for convergence.
while k_conv < k_convergence or fission_source_conv < fs_convergence:
    num_source_iter_fast = 0
    num_source_iter_thermal = 0
    # loop over the energy groups (we only have two energy groups:
    # 0 for fast and 1 for thermal
    for energy_group in [0, 1]:
        if energy_group == 0:
            source[:] = fission_source_old[:] / k_old
        elif energy_group == 1:
            for cell_num, cell in enumerate(material_cell):
                source[cell_num] = material_data.ix['downscatter_fast', cell] * scalar_flux_old[(cell_num, 0)]
        else:
            print("Error: Energy group not defined!")
            quit()

        # Inner loop to determine source convergence
        # start by determining the source term based on the energy and the Q term from above.
        test_iter = 0
        while test_iter < 100:  # source_convergence < source_convergence_test:
            scalar_flux_new = np.zeros((128, 2))
            current_new = np.zeros((128, 2))

            if energy_group == 0:
                for cell_num, cell in enumerate(material_cell):
                    Q[cell_num] = 0.5 * (material_data.ix['inscatter_fast', cell] * scalar_flux_old[cell_num, 0]
                                         + source[cell_num])

                step_characteristic(angular_flux_pos_lhs, angular_flux_pos_rhs, scalar_flux_new, current_new,
                                         material_data, cell_width, mu_n, energy_group)
                max = np.amax(scalar_flux_new[:, 0])
                max_old = np.amax(scalar_flux_old[:, 0])
                fast_source_convergence = abs((max - max_old) / max)

                num_source_iter_fast += 1
                scalar_flux_old[:, 0] = scalar_flux_new[:, 0]
                current_old[:, 0] = current_new[:, 0]

                if fast_source_convergence < source_convergence:
                    break


            elif energy_group == 1:
                for cell_num, cell in enumerate(material_cell):
                    Q[cell_num] = 0.5 * (material_data.ix['inscatter_thermal', cell] * scalar_flux_old[cell_num, 0]
                                         + source[cell_num])

                step_characteristic(angular_flux_pos_lhs, angular_flux_pos_rhs, scalar_flux_new, current_new,
                                         material_data, cell_width, mu_n, energy_group)

                max = np.amax(scalar_flux_new[:, 1])
                max_old = np.amax(scalar_flux_old[:, 1])
                thermal_source_convergence = abs((max - max_old) / max)

                num_source_iter_thermal += 1
                scalar_flux_old[:, 1] = scalar_flux_new[:, 1]
                current_old[:, 1] = current_new[:, 1]

                if thermal_source_convergence < source_convergence:
                    break

    # Create the new fission source
    for cell_num, cell in enumerate(material_cell):
        fission_source_new[cell_num] = (material_data.ix['nusigmaf_fast', cell] * scalar_flux_old[(cell_num, 0)] \
                                       + material_data.ix['nusigmaf_thermal', cell] * scalar_flux_old[(cell_num, 1)]) \
                                        * cell_width
    # Determine the new k value
    k_new[num_power_iter + 1] = k_old * sum(fission_source_new) * cell_width / (sum(fission_source_old) * cell_width)

    # Calculate convergence criteria
    fs_convergence = abs(np.amax(fission_source_new) - np.amax(fission_source_old))
    print("New k iteration number: %d" % num_power_iter)
    print("k_eff: %f" % k_new[num_power_iter + 1])
    print("k_eff convergence: %f, %f" % (k_convergence, fs_convergence))

    k_convergence = abs((k_new[num_power_iter + 1] - k_old) / k_new[num_power_iter + 1])

    fission_source_old[:] = fission_source_new[:]
    k_old = k_new[num_power_iter + 1]

    num_power_iter += 1
    # loop over the angles to determine the angular flux

    if num_power_iter > 1000:
        break

mpl.plot(fission_source_new)
mpl.show()
mpl.plot(scalar_flux_old)
mpl.show()
mpl.plot(current_old)
mpl.show()
