import pandas as pd
import numpy as np
import post_process as pp
from scipy.linalg import lu_factor, lu_solve

def step_characteristic(angular_flux_pos_lhs, angular_flux_pos_rhs, scalar_flux_new,
                        current_new, material_data, cell_width, mu_n, nrg, cell_edge_flux_new):
    if nrg == 0:
        total = 'total_fast'
    elif nrg == 1:
        total = 'total_thermal'

    # sweep over the angles (starting with the positive angles, followed by the
    # cells to solve for the angular flux
    for angle in reversed(range(len(mu_n[0]))):
        if angle > 4:
            for cell_num, cell in enumerate(material_cell):
                tau = material_data.ix[total, cell] * cell_width / abs(mu_n[(0, angle)])

                if cell_num == 0:
                    cell_edge_flux_new[(cell_num, nrg)] += angular_flux_pos_lhs[(9-angle, nrg)] * mu_n[(1, angle)]
                    angular_flux_pos_rhs[(angle, nrg)] = \
                        angular_flux_pos_lhs[(9 - angle, nrg)] * np.exp(-tau) \
                        + (Q[cell_num] / material_data.ix[total, cell]) \
                        * (1 - np.exp(-tau))

                    cell_average_angular_flux = (cell_width * Q[cell_num] -
                        mu_n[(0, angle)]
                        * (angular_flux_pos_rhs[(angle, nrg)]
                        - angular_flux_pos_lhs[(9 - angle, nrg)])) \
                        / (material_data.ix[total, cell] * cell_width)

                else:
                    angular_flux_pos_rhs[(angle, nrg)] = \
                        angular_flux_pos_lhs[(angle, nrg)] * np.exp(-tau) + (Q[cell_num]
                        / material_data.ix[total, cell]) * (1 - np.exp(-tau))

                    cell_average_angular_flux = (cell_width * Q[cell_num] -
                        mu_n[(0, angle)] * (angular_flux_pos_rhs[(angle, nrg)]
                        - angular_flux_pos_lhs[(angle, nrg)])) \
                        / (material_data.ix[total, cell] * cell_width)

                scalar_flux_new[(cell_num, nrg)] += cell_average_angular_flux * mu_n[(1, angle)]
                current_new[(cell_num, nrg)] += cell_average_angular_flux * mu_n[(1, angle)] * mu_n[(0, angle)]
                cell_edge_flux_new[(cell_num+1, nrg)] += angular_flux_pos_rhs[(angle, nrg)] * mu_n[(1, angle)]
                cell_edge_current_new[(cell_num, nrg)] += \
                    angular_flux_pos_rhs[(angle, nrg)] * mu_n[(1, angle)] * mu_n[(0, angle)]

                angular_flux_pos_lhs[:, nrg] = np.copy(angular_flux_pos_rhs[:, nrg])


        # for mu < 0, we remove the - sign, as the change in delta
        # x would create an additional negative to cancel
        else:
            for cell_num, cell in enumerate(reversed(material_cell)):
                rev_cell_num = 127 - cell_num
                tau = material_data.ix[total, cell] * cell_width / abs(mu_n[(0, angle)])

                # Special case to flip the neutrons from the first sweep backwards.
                if rev_cell_num == 127:
                    cell_edge_flux_new[(rev_cell_num + 1, nrg)] += angular_flux_pos_rhs[(9-angle, nrg)] * mu_n[(1, angle)]
                    angular_flux_pos_lhs[(angle, nrg)] = \
                        angular_flux_pos_rhs[(9 - angle, nrg)] * np.exp(-tau) \
                        + Q[rev_cell_num] / material_data.ix[total, cell] \
                        * (1 - np.exp(-tau))

                    cell_average_angular_flux = \
                        (cell_width * Q[rev_cell_num] - mu_n[(0, angle)]
                        * (angular_flux_pos_rhs[(9 - angle, nrg)]
                        - angular_flux_pos_lhs[(angle, nrg)])) \
                        / (material_data.ix[total, cell] * cell_width)
                else:
                    angular_flux_pos_lhs[(angle, nrg)] = \
                        angular_flux_pos_rhs[(angle, nrg)] * np.exp(-tau) \
                        + Q[rev_cell_num] / material_data.ix[total, cell] \
                        * (1 - np.exp(-tau))

                    cell_average_angular_flux = \
                        (cell_width * Q[rev_cell_num] - mu_n[(0, angle)]
                        * (angular_flux_pos_rhs[(angle, nrg)]
                        - angular_flux_pos_lhs[(angle, nrg)])) \
                        / (material_data.ix[total, cell] * cell_width)

                scalar_flux_new[(rev_cell_num, nrg)] += cell_average_angular_flux * mu_n[(1, angle)]
                current_new[(rev_cell_num, nrg)] += cell_average_angular_flux * mu_n[(1, angle)] * mu_n[(0, angle)]
                cell_edge_flux_new[(rev_cell_num, nrg)] += angular_flux_pos_lhs[(angle, nrg)] * mu_n[(1, angle)]
                cell_edge_current_new[(rev_cell_num, nrg)] += \
                    angular_flux_pos_lhs[(angle, nrg)] * mu_n[(1, angle)] * mu_n[(0, angle)]

                angular_flux_pos_rhs[:, nrg] = np.copy(angular_flux_pos_lhs[:, nrg])

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
material_data = \
    pd.DataFrame(materials_b, columns=['total_fast', 'inscatter_fast', 'downscatter_fast',
                                        'nusigmaf_fast', 'chi_fast',
                                       'total_thermal', 'inscatter_thermal',
                                       'downscatter_thermal', 'nusigmaf_thermal',
                                       'chi_thermal'],
                             index=['MOX', 'U', 'water'])
material_data = material_data.T
mu_n = np.array([[-0.973906528517, -0.865063366689, -0.679409568299,
                  -0.433395394129, -0.148874338982,
                  0.148874338982, 0.433395394129, 0.679409568299,
                  0.865063366689, 0.973906528517],
                 [0.0666713444544, 0.149451349057, 0.219086362450,
                  0.269266719318, 0.295524224712,
                  0.295524224712, 0.269266719318, 0.219086362450,
                  0.149451349057, 0.0666713444544]])
# Initialize k, the fission product, angular flux at 0,
# in the positive direction, scalar flux, convergence criteria,
# and the number of iterations for convergence

k_old = 1
k_new = 0
fission_source_old = np.ones(128)
fission_source_new = np.ones(128)
cell_edge_flux_new = np.zeros((129, 2))
cell_edge_current_new = np.zeros((128, 2))
cell_edge_flux_old = np.zeros((129, 2))
cell_edge_current_old = np.zeros((128, 2))
scalar_flux_old = np.zeros((128, 2))
current_old = np.zeros((128, 2))
angular_flux_pos_lhs = np.zeros((10, 2))
angular_flux_pos_rhs = np.zeros((10, 2))
Q = np.zeros(128)
source = np.zeros(128)

cell_width = 0.15625
k_conv = 0.1
fission_source_conv = 0.1
source_convergence = 0.1
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
                source[cell_num] = material_data.ix['downscatter_fast', cell] \
                                   * scalar_flux_new[(cell_num, 0)]

        # Inner loop to determine source convergence
        # Start by determining the source term based on the energy and the Q term from above.
        # Warning: there is not kill for this loop if things don't converge!
        while 1 < 2:  # source_convergence < source_convergence_test:
            scalar_flux_new = np.zeros((128, 2))
            current_new = np.zeros((128, 2))
            cell_edge_current_new = np.zeros((128, 2))
            cell_edge_flux_new = np.zeros((129, 2))

            if energy_group == 0:
                for cell_num, cell in enumerate(material_cell):
                    Q[cell_num] = 0.5 * (material_data.ix['inscatter_fast', cell]
                                         * scalar_flux_old[cell_num, energy_group]
                                         + source[cell_num])
                step_characteristic(angular_flux_pos_lhs, angular_flux_pos_rhs,
                                    scalar_flux_new, current_new, material_data,
                                    cell_width, mu_n, energy_group, cell_edge_flux_new)
                num_source_iter_fast += 1

            elif energy_group == 1:
                for cell_num, cell in enumerate(material_cell):
                    Q[cell_num] = 0.5 * (material_data.ix['inscatter_thermal', cell]
                                         * scalar_flux_old[cell_num, energy_group]
                                         + source[cell_num])

                step_characteristic(angular_flux_pos_lhs, angular_flux_pos_rhs,
                                    scalar_flux_new, current_new, material_data,
                                    cell_width, mu_n, energy_group, cell_edge_flux_new)
                num_source_iter_thermal += 1

            group_source_convergence = abs((np.amax(scalar_flux_new[:, energy_group])
                                        - np.amax(scalar_flux_old[:, energy_group]))
                                        / np.amax(scalar_flux_new[:, energy_group]))
            scalar_flux_old[:, energy_group] = scalar_flux_new[:, energy_group]
            current_old[:, energy_group] = current_new[:, energy_group]
            cell_edge_flux_old[:, energy_group] = cell_edge_flux_new[:, energy_group]
            cell_edge_current_old[:, energy_group] = \
                cell_edge_current_new[:, energy_group]
            if group_source_convergence < source_convergence:
                break

    # Create the new fission source
    for cell_num, cell in enumerate(material_cell):
        fission_source_new[cell_num] = (material_data.ix['nusigmaf_fast', cell]
                                        * scalar_flux_old[(cell_num, 0)] \
                                       + material_data.ix['nusigmaf_thermal', cell]
                                        * scalar_flux_old[(cell_num, 1)])
    # Determine the new k value
    k_new = k_old * sum(fission_source_new) * cell_width \
            / (sum(fission_source_old) * cell_width)

    # Calculate convergence criteria
    fs_convergence = abs(np.amax(fission_source_new) - np.amax(fission_source_old))
    k_convergence = abs((k_new - k_old) / k_new)
    fission_source_old[:] = fission_source_new[:]
    k_old = k_new

    num_power_iter += 1
    if num_power_iter > 1000:
        break

# Create the pin cell average
fast = pp.pin_cell_average_flux(scalar_flux_old[:, 0])
thermal = pp.pin_cell_average_flux(scalar_flux_old[:, 1])
pin_cell_average = np.concatenate(([fast], [thermal]))
pin_cell_average = pin_cell_average.T

# Write out the flux, current and pin average flux to excel
flux_excel = pd.DataFrame(scalar_flux_old)
edge_flux_excel = pd.DataFrame(cell_edge_flux_old)
current_excel = pd.DataFrame(current_old)
edge_current_excel = pd.DataFrame(cell_edge_current_old)
pin_average_excel = pd.DataFrame(pin_cell_average)

filepath = 'deterministic.xlsx'
writer = pd.ExcelWriter(filepath)

flux_excel.to_excel(writer, index=False, sheet_name='cell_flux')
edge_flux_excel.to_excel(writer, index=False, sheet_name='cell_edge_flux')
current_excel.to_excel(writer, index=False, sheet_name='current')
edge_current_excel.to_excel(writer, index=False, sheet_name='cell_edge_current')
pin_average_excel.to_excel(writer, index=False, sheet_name='pin_average')
writer.save()

# Print out plots of each
#pp.flux_histogram(pin_cell_average, "Pin Averaged Flux with Cell Average Flux",
#                  "Pin Cell", "Flux (1/cm^2)", "Fast Flux", "Thermal Flux",
#                  scalar_flux_old)
#pp.plot_flux(scalar_flux_old, "Cell Average Flux", "Cell", "Flux (1/cm^2)",
#             "Fast Flux", "Thermal Flux")
#pp.plot_flux(cell_edge_flux_old, "Cell Edge Flux", "Cell Edge", "Flux (1/cm^2)",
#             "Fast Flux", "Thermal Flux")
#pp.plot_1d_array(fission_source_new, "Fission Source", "Cell",
#                 "Unscaled Probability", "Fission Source")
#pp.plot_flux(current_old, "Cell Average Current", "Cell", "Flux (1/cm^2)",
#             "Fast Current", "Thermal Current")
#pp.plot_flux(cell_edge_current_old, "Cell Edge Current", "Cell Edge",
#             "Flux (1/cm^2)", "Fast Current", "Thermal Current")
#pp.plot_flux(pin_cell_average, "Pin Averaged Flux", "Pin Cell",
#             "Flux (1/cm^2)", "Fast Flux", "Thermal Flux")

flux_coeff = np.zeros((10,2))
fast_source = np.zeros((10,2))
thermal_source = np.zeros((10,2))

# Import the homogenized data
homogenized_data = pd.read_excel('Homogenized_XC_Ryan_Stewart.xlsx')

# assign the homigenized data for the fast energy group
delta_x_one = 0.15625
diff_fast_1 = homogenized_data.A3.fast.diffusion
disc_fast_left_1 = homogenized_data.A3.fast.discontinuity_left
disc_fast_right_1 = homogenized_data.A3.fast.discontinuity_right
rem_fast_1 = homogenized_data.A3.fast.removal
nusigmaf_fast_1 = homogenized_data.A3.fast.nusigmaf
delta_x_two = 0.15625
diff_fast_2 = homogenized_data.A1.fast.diffusion
disc_fast_left_2 = homogenized_data.A1.fast.discontinuity_left
disc_fast_right_2 = homogenized_data.A1.fast.discontinuity_right
rem_fast_2 = homogenized_data.A1.fast.removal
nusigmaf_fast_2 = homogenized_data.A1.fast.nusigmaf


# create the coefficient matrix for fast energy group
coeff_matrix_fast = np.vstack([[0, 1, -3, 6, -10, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 3, 6, 10],
                     [0, (diff_fast_1 / delta_x_one), 3 * (diff_fast_1 / delta_x_one),
                      6 * (diff_fast_1 / delta_x_one), 10 * (diff_fast_1 / delta_x_one),
                      0, -(diff_fast_2 / delta_x_two), 3 * (diff_fast_2 / delta_x_two),
                      -6 * (diff_fast_2 / delta_x_two), 10 * (diff_fast_2 / delta_x_two)],
                     [disc_fast_right_1, disc_fast_right_1, disc_fast_right_1,
                      disc_fast_right_1, disc_fast_right_1,
                      -disc_fast_left_2, disc_fast_left_2, -disc_fast_left_2,
                      disc_fast_left_2, -disc_fast_left_2],
                     [rem_fast_1, 0, -12 * (diff_fast_1 / (delta_x_one**2)), 0,
                      -40 * (diff_fast_1 / (delta_x_one**2)), 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, rem_fast_2, 0, -12 * (diff_fast_2 / (delta_x_two**2)), 0,
                      -40 * (diff_fast_2 / (delta_x_two**2))],
                     [0, rem_fast_1, 0, -60 * (diff_fast_1 / (delta_x_one**2)), 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, rem_fast_2, 0, -60 * (diff_fast_2 / (delta_x_two**2)), 0],
                     [0, 0, rem_fast_1, 0, -140 * (diff_fast_1 / (delta_x_one**2)), 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, rem_fast_2, 0, -140 * (diff_fast_2 / (delta_x_two**2))]])

# assign the homigenized data for the thermal energy group
delta_x_one = 0.15625
diff_thermal_1 = homogenized_data.A3.thermal.diffusion
disc_thermal_left_1 = homogenized_data.A3.thermal.discontinuity_left
disc_thermal_right_1 = homogenized_data.A3.thermal.discontinuity_right
rem_thermal_1 = homogenized_data.A3.thermal.removal
nusigmaf_thermal_1 = homogenized_data.A3.thermal.nusigmaf
delta_x_two = 0.15625
diff_thermal_2 = homogenized_data.A1.thermal.diffusion
disc_thermal_left_2 = homogenized_data.A1.thermal.discontinuity_left
disc_thermal_right_2 = homogenized_data.A1.thermal.discontinuity_right
rem_thermal_2 = homogenized_data.A1.thermal.removal
nusigmaf_thermal_2 = homogenized_data.A1.thermal.nusigmaf

# create the coefficient matrix for thermal energy group
coeff_matrix_thermal = np.vstack([[0, 1, -3, 6, -10, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 3, 6, 10],
                     [0, (diff_thermal_1 / delta_x_one), 3 * (diff_thermal_1 / delta_x_one),
                      6 * (diff_thermal_1 / delta_x_one), 10 * (diff_thermal_1 / delta_x_one),
                      0, -(diff_thermal_2 / delta_x_two), 3 * (diff_thermal_2 / delta_x_two),
                      -6 * (diff_thermal_2 / delta_x_two), 10 * (diff_thermal_2 / delta_x_two)],
                     [disc_thermal_right_1, disc_thermal_right_1, disc_thermal_right_1,
                      disc_thermal_right_1, disc_thermal_right_1,
                      -disc_thermal_left_2, disc_thermal_left_2, -disc_thermal_left_2,
                      disc_thermal_left_2, -disc_thermal_left_2],
                     [rem_thermal_1, 0, -12 * (diff_thermal_1 / (delta_x_one**2)), 0,
                      -40 * (diff_thermal_1 / (delta_x_one**2)), 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, rem_thermal_2, 0, -12 * (diff_thermal_2 / (delta_x_two**2)), 0,
                      -40 * (diff_thermal_2 / (delta_x_two**2))],
                     [0, rem_thermal_1, 0, -60 * (diff_thermal_1 / (delta_x_one**2)), 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, rem_thermal_2, 0, -60 * (diff_thermal_2 / (delta_x_two**2)), 0],
                     [0, 0, rem_thermal_1, 0, -140 * (diff_thermal_1 / (delta_x_one**2)), 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, rem_thermal_2, 0, -140 * (diff_thermal_2 / (delta_x_two**2))]])

#  Perform the LU decomposition
coeff_matrix_fast_LU = lu_factor(coeff_matrix_fast)
coeff_matrix_thermal_LU = lu_factor(coeff_matrix_thermal)

#Start looping over
k_old = 1
k_new = 1

k_conv = 0.01
fission_source_conv = 0.01
source_convergence = 0.01
fs_convergence = 1
k_convergence = 1

old_fast_flux = np.zeros(10)
new_fast_flux = np.zeros(10)
old_thermal_flux = np.zeros(10)
new_thermal_flux = np.zeros(10)

old_fission_source = np.ones(10)
new_fission_source = np.ones(10)


i=0
while 1 < 2:
    for energy_group in [0, 1]:
        if energy_group == 0:
            fast_flux_RHS = [0, 0, 0, 0, (1/k_old)*old_fission_source[0], (1/k_old)*old_fission_source[5],
                             (1/k_old)*old_fission_source[1], (1/k_old)*old_fission_source[6],
                             (1/k_old)*old_fission_source[2], (1/k_old)*old_fission_source[7]]
            new_fast_flux = lu_solve(coeff_matrix_fast_LU, fast_flux_RHS)
            #new_fast_flux = np.linalg.solve(coeff_matrix_fast, fast_flux_RHS)
        else:
            thermal_flux_RHS = [0, 0, 0, 0, new_fast_flux[0] * rem_fast_1, new_fast_flux[5] * rem_fast_1,
                                new_fast_flux[1] * rem_fast_1, new_fast_flux[6] * rem_fast_2,
                                new_fast_flux[2] * rem_fast_2, new_fast_flux[7] * rem_fast_2]
            new_thermal_flux = lu_solve(coeff_matrix_thermal_LU, thermal_flux_RHS)
            #new_thermal_flux = np.linalg.solve(coeff_matrix_thermal, thermal_flux_RHS)
        # Calculate the new fission source
        for coeff, flux in enumerate(new_fast_flux):
            if coeff < 5:
                new_fission_source[coeff] = new_fast_flux[coeff] * nusigmaf_fast_1 + new_thermal_flux[coeff] * nusigmaf_thermal_1
            else:
                new_fission_source[coeff] = new_fast_flux[coeff] * nusigmaf_fast_2 + new_thermal_flux[coeff] * nusigmaf_thermal_2
    k_new = k_old * sum(new_fission_source) / sum(old_fission_source)
    k_conv = abs(k_new-k_old)
    fs_convergence = max(new_fission_source - old_fission_source)
    old_fission_source[:] = new_fission_source[:]

    print(k_new, k_old)
    k_old = k_new

    print(fs_convergence)
    #print(fs_convergence)
    i+=1
    if i > 10:
        break

