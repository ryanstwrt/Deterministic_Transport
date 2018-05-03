import pandas as pd
import numpy as np
import matplotlib.pyplot as mpl
import post_process as pp


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

# Beginning of the Deterministic solver program.
# Create the cell array which designates the material present in each cell
mox_cell = ['water', 'water', 'MOX', 'MOX', 'MOX', 'MOX', 'water', 'water']
u_cell = ['water', 'water', 'U', 'U', 'U', 'U', 'water', 'water']
material_cell = []
for i in range(16):
    material_cell += mox_cell if i < 8 else u_cell

# Create the material specifications for each problem
materials_a = [[0.2, 0.2, 0.0, 0.0, 1.0, 0.6, 0.0, 0.0, 0.90, 0.0],
               [0.2, 0.2, 0.0, 0.0, 1.0, 0.2, 0.0, 0.0, 0.30, 0.0],
               [0.2, 0.17, 0.03, 0.0, 0.0, 1.1, 1.1, 0.0, 0.0, 0.0]]
materials_b = [[0.2, 0.185, 0.015, 0.0, 1.0, 1.2, 0.9, 0.0, 0.45, 0.0],
               [0.2, 0.185, 0.015, 0.0, 1.0, 1.0, 0.9, 0.0, 0.15, 0.0],
               [0.2, 0.17, 0.03, 0.0, 0.0, 1.1, 1.1, 0.0, 0.0, 0.0]]

# Create a data from for the material cross sections to be used for each material
material_data = pd.DataFrame(materials_b, columns=['total_fast', 'inscatter_fast', 'downscatter_fast',
                                                   'nusigmaf_fast', 'chi_fast', 'total_thermal', 'inscatter_thermal',
                                                   'downscatter_thermal', 'nusigmaf_thermal', 'chi_thermal'],
                             index=['MOX', 'U', 'water'])
material_data = material_data.T
mu_n = np.array([[-0.973906528517, -0.865063366689, -0.679409568299, -0.433395394129, -0.148874338982,
                  0.148874338982, 0.433395394129, 0.679409568299, 0.865063366689, 0.973906528517],
                 [0.0666713444544, 0.149451349057, 0.219086362450, 0.269266719318, 0.295524224712,
                  0.295524224712, 0.269266719318, 0.219086362450, 0.149451349057, 0.0666713444544]])
# Initialize k, the fission product, angular flux at 0, in the positive direction, scalar flux,
# convergence criteria, and the number of iterations for convergence

k_old = 1
k_new = 0
fission_source_old = np.ones(128)
fission_source_new = np.ones(128)
scalar_flux_old = np.zeros((128, 2))
current_old = np.zeros((128, 2))
angular_flux_pos_lhs = np.zeros((10, 2))
angular_flux_pos_rhs = np.zeros((10, 2))
Q = np.zeros(128)
source = np.zeros(128)

cell_width = 0.15625
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
                source[cell_num] = material_data.ix['downscatter_fast', cell] * scalar_flux_new[(cell_num, 0)]

        # Inner loop to determine source convergence
        # Start by determining the source term based on the energy and the Q term from above.
        # Warning: there is not kill for this loop if things don't converge!
        while 1 < 2:  # source_convergence < source_convergence_test:
            scalar_flux_new = np.zeros((128, 2))
            current_new = np.zeros((128, 2))

            if energy_group == 0:
                for cell_num, cell in enumerate(material_cell):
                    Q[cell_num] = 0.5 * (material_data.ix['inscatter_fast', cell] * scalar_flux_old[cell_num, energy_group]
                                         + source[cell_num])
                step_characteristic(angular_flux_pos_lhs, angular_flux_pos_rhs, scalar_flux_new, current_new,
                                         material_data, cell_width, mu_n, energy_group)
                num_source_iter_fast += 1

            elif energy_group == 1:
                for cell_num, cell in enumerate(material_cell):
                    Q[cell_num] = 0.5 * (material_data.ix['inscatter_thermal', cell] * scalar_flux_old[cell_num, energy_group]
                                         + source[cell_num])

                step_characteristic(angular_flux_pos_lhs, angular_flux_pos_rhs, scalar_flux_new, current_new,
                                         material_data, cell_width, mu_n, energy_group)
                num_source_iter_thermal += 1

            group_source_convergence = abs((np.amax(scalar_flux_new[:, energy_group])
                                        - np.amax(scalar_flux_old[:, energy_group]))
                                        / np.amax(scalar_flux_new[:, energy_group]))
            scalar_flux_old[:, energy_group] = scalar_flux_new[:, energy_group]
            current_old[:, energy_group] = current_new[:, energy_group]
            #print("Source Convergence: %0.8f" % group_source_convergence)
            if group_source_convergence < source_convergence:
                break

    # Create the new fission source
    for cell_num, cell in enumerate(material_cell):
        fission_source_new[cell_num] = (material_data.ix['nusigmaf_fast', cell] * scalar_flux_old[(cell_num, 0)] \
                                       + material_data.ix['nusigmaf_thermal', cell] * scalar_flux_old[(cell_num, 1)])
    # Determine the new k value
    k_new = k_old * sum(fission_source_new) * cell_width / (sum(fission_source_old) * cell_width)

    # Calculate convergence criteria
    fs_convergence = abs(np.amax(fission_source_new) - np.amax(fission_source_old))
    k_convergence = abs((k_new - k_old) / k_new)
    fission_source_old[:] = fission_source_new[:]
    k_old = k_new
    print("New k iteration number: %d" % num_power_iter)
    print("k_eff: %f" % k_new)
    print("k_eff convergence: %f, %f" % (k_convergence, fs_convergence))
    print("Source convergence: %d, %d" % (num_source_iter_fast, num_source_iter_thermal))

    num_power_iter += 1
    if num_power_iter > 1000:
        break

# Create the pin cell average
fast = pp.pin_cell_average_flux(scalar_flux_old[:, 0])
thermal = pp.pin_cell_average_flux(scalar_flux_old[:, 1])
pin_cell_average = np.concatenate(([fast], [thermal]))
pin_cell_average = pin_cell_average.T

# Write out the flux, current and pin average flux to excel
source_excel = pd.DataFrame(scalar_flux_old)
current_excel = pd.DataFrame(current_old)
pin_average_excel = pd.DataFrame(pin_cell_average)
filepath = 'deterministic.xlsx'

source_excel.to_excel(filepath, index=False, sheet_name='source')
current_excel.to_excel(filepath, index=False, sheet_name='current')
pin_average_excel.to_excel(filepath, index=False, sheet_name='pin_average')

# Print out plots of each
pp.plot_flux(scalar_flux_old, "Flux", "Cell", "Flux (1/cm^2)", "Fast Flux", "Thermal Flux")
pp.plot_1d_array(fission_source_new, "Fission Source", "Cell", "Unscaled Probability", "Fission Source")
pp.plot_flux(current_old, "Current", "Cell #", "Flux (1/cm^2)", "Fast Current", "Thermal Current")
pp.plot_flux(pin_cell_average, "Pin Averaged Flux", "Pin Cell", "Flux (1/cm^2)", "Fast Flux", "Thermal Flux")
pp.flux_histogram(pin_cell_average, "Pin Averaged Histogram", "Pin Cell", "Flux (1/cm^2)", "Fast Flux", "Thermal Flux")


