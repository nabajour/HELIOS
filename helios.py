# ==============================================================================
# This is the main file of HELIOS.
# Copyright (C) 2018 Matej Malik
#
# To run HELIOS simply execute this file with Python 3.x
# ==============================================================================
# This file is part of HELIOS.
#
#     HELIOS is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     HELIOS is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You find a copy of the GNU General Public License in the main
#     HELIOS directory under <license.txt>. If not, see
#     <http://www.gnu.org/licenses/>.
# ==============================================================================

import pycuda.driver as cuda
import numpy as np
from source import Vcoupling_modification as Vmod
from source import clouds
from source import realtime_plotting as rt_plot
from source import computation as comp
from source import write
from source import host_functions as hsfunc
from source import quantities as quant
from source import read
import sys
sys.path.append("/home/nab/prog/astro/unibe/THOR-dev/Alfrodull/build/")
import pylfrodull  # noqa


def run_helios():
    """ runs a normal HELIOS run with standard I/O """

    ##########################
    pylfrodull.init_alfrodull()
    ##########################

    reader = read.Read()
    keeper = quant.Store()
    computer = comp.Compute()
    writer = write.Write()
    plotter = rt_plot.Plot()
    cloudy = clouds.Cloud()
    Vmodder = Vmod.Vcoupling()

    # read input files and do preliminary calculations
    reader.read_param_file(keeper, Vmodder)
    reader.read_command_line(keeper, Vmodder)

    reader.read_star(keeper)
    print("Tstar: ", keeper.T_star)

    if Vmodder.V_coupling == 1:
        Vmodder.read_or_create_iter_count()
        Vmodder.read_species()
        Vmodder.read_molecular_opacities(keeper)
        Vmodder.read_layer_molecular_abundance(keeper)

    ##################################################
    # transfer data from opacity table
    (dev_opac_wave_ptr,
     dev_interwave_ptr,
     dev_deltawave_ptr,
     dev_opac_y_ptr,
     nbin,
     ny) = pylfrodull.get_opac_data_for_helios()

    keeper.nbin = np.uint32(nbin)
    keeper.ny = np.uint32(ny)

    keeper.dev_opac_wave = np.uint64(dev_opac_wave_ptr)

    keeper.opac_wave = cuda.from_device(dev_opac_wave_ptr,
                                        (keeper.nbin),
                                        np.float64)

    keeper.dev_opac_interwave = np.uint64(dev_interwave_ptr)
    keeper.opac_interwave = cuda.from_device(dev_interwave_ptr,
                                             (keeper.nbin + 1),
                                             np.float64)

    keeper.dev_opac_deltawave = np.uint64(dev_deltawave_ptr)

    keeper.opac_deltawave = cuda.from_device(dev_deltawave_ptr,
                                             (keeper.nbin),
                                             np.float64)

    keeper.dev_opac_y = np.uint64(dev_opac_y_ptr)

    ##################################################

    # reader.read_opac_file(keeper, Vmodder)
    reader.read_entropy_table(keeper)
    cloudy.main_cloud_method(keeper)
    keeper.dimensions()

    hsfunc.planet_param(keeper, reader)
    hsfunc.set_up_numerical_parameters(keeper)
    hsfunc.construct_grid(keeper)
    hsfunc.initial_temp(keeper, reader, Vmodder)
    if keeper.approx_f == 1:
        hsfunc.approx_f_from_formula(keeper, reader)
    hsfunc.calc_F_intern(keeper)

    # get ready for GPU computations
    keeper.create_zero_arrays(Vmodder)
    keeper.convert_input_list_to_array(Vmodder)
    keeper.copy_host_to_device(Vmodder)
    keeper.allocate_on_device(Vmodder)

    ##########################
    pylfrodull.init_parameters(keeper.nlayer,
                               keeper.iso,
                               keeper.T_star,
                               keeper.real_star,
                               keeper.fake_opac,
                               keeper.T_surf,
                               keeper.surf_albedo,
                               keeper.g_0,
                               keeper.epsi,
                               keeper.mu_star,
                               keeper.scat,
                               keeper.scat_corr,
                               keeper.R_planet,
                               keeper.R_star,
                               keeper.a,
                               keeper.dir_beam,
                               keeper.geom_zenith_corr,
                               keeper.f_factor,
                               keeper.w_0_limit,
                               keeper.surf_albedo  # TODO: check, there is only one of these albedo vars
                               )

    pylfrodull.set_clouds_data(keeper.clouds,
                               keeper.dev_cloud_opac_lay.ptr,
                               keeper.dev_cloud_opac_int.ptr,
                               keeper.dev_cloud_scat_cross_lay.ptr,
                               keeper.dev_cloud_scat_cross_int.ptr,
                               keeper.dev_g_0_tot_lay.ptr,
                               keeper.dev_g_0_tot_int.ptr)

    pylfrodull.allocate()

    pylfrodull.prepare_planck_table()
    pylfrodull.correct_incident_energy(
        keeper.dev_starflux.ptr,
        keeper.real_star,
        keeper.energy_correction
    )

    # device pointers
    pointers = pylfrodull.get_dev_pointers()
    for i, p in enumerate(pointers):
        print(f"{i}: {p:x}")

    (dev_scat_cross_section_lay_ptr,
     dev_scat_cross_section_int_ptr,
     dev_opac_wg_lay_ptr,
     dev_planck_lay_ptr,
     dev_planck_int_ptr,
     dev_planck_grid_ptr,
     dev_delta_tau_wg_ptr,
     dev_delta_tau_wg_upper_ptr,
     dev_delta_tau_wg_lower_ptr,
     dev_delta_colmass_ptr,
     dev_delta_col_upper_ptr,
     dev_delta_col_lower_ptr,
     dev_meanmolmass_ptr,
     dev_trans_wg_ptr,
     dev_trans_wg_upper_ptr,
     dev_trans_wg_lower_ptr,
     plancktable_dim,
     plancktable_step) = pointers

    keeper.dev_delta_tau_wg = np.uint64(dev_delta_tau_wg_ptr)
    keeper.dev_delta_tau_wg_upper = np.uint64(dev_delta_tau_wg_upper_ptr)
    keeper.dev_delta_tau_wg_lower = np.uint64(dev_delta_tau_wg_lower_ptr)
    keeper.dev_planckband_lay = np.uint64(dev_planck_lay_ptr)
    keeper.dev_planckband_grid = np.uint64(dev_planck_grid_ptr)

    keeper.dev_delta_colmass = np.uint64(dev_delta_colmass_ptr)
    keeper.dev_delta_col_upper = np.uint64(dev_delta_col_upper_ptr)
    keeper.dev_delta_col_lower = np.uint64(dev_delta_col_lower_ptr)

    # used in printout, needs to be copied back
    keeper.dev_meanmolmass_lay = np.uint64(dev_meanmolmass_ptr)

    # used in final values computation (postprocess), no need to copy back
    keeper.dev_trans_wg = np.uint64(dev_trans_wg_ptr)
    keeper.dev_trans_wg_upper = np.uint64(dev_trans_wg_upper_ptr)
    keeper.dev_trans_wg_lower = np.uint64(dev_trans_wg_lower_ptr)
    # used in final values computation (postprocess), no need to copy back
    keeper.dev_opac_wg_lay = np.uint64(dev_opac_wg_lay_ptr)

    # print("dev_planck_grid: ", dev_planck_grid_ptr)
    # print("dev_delta_colmass_ptr", dev_delta_colmass_ptr)
    # delta_colmass = cuda.from_device(dev_delta_colmass_ptr,
    #                                  (keeper.nlayer),
    #                                  np.float64)
    # print("delta_colmass", delta_colmass)
    # print(cuda.from_device(dev_planck_grid_ptr,
    #                        ((plancktable_dim+1)*323),
    #                        np.float64))

    # print(cuda.from_device(dev_interwave_ptr,
    #                        (323+1),
    #                        np.float64))
    # print(cuda.from_device(dev_deltawave_ptr,
    #                        (323),
    #                        np.float64))

    keeper.plancktable_dim = np.uint32(plancktable_dim)
    keeper.plancktable_step = np.uint32(plancktable_step)

    ##########################

    # conduct the GPU core computations

    # computer.correct_incident_energy(keeper)

    if Vmodder.V_coupling == 1:
        if Vmodder.V_iter_nr > 0:
            Vmodder.interpolate_f_molecule_and_meanmolmass(keeper)
            Vmodder.combine_to_scat_cross(keeper)

    computer.radiation_loop(keeper, writer, plotter, Vmodder)

    computer.convection_loop(keeper, writer, plotter, Vmodder)

    # TODO: understand these
    # calculates the transmission function in each layer
    # parameters: (or same with uper and lower in noniso case)
    # quant.dev_trans_wg,
    # quant.dev_trans_band,
    # quant.dev_delta_tau_wg,
    # quant.dev_delta_tau_band,
    # quant.dev_gauss_weight,

    computer.integrate_optdepth_transmission(keeper)
    #  calculate the transmission weighting function and the contribution function for each layer and waveband
    # parameters: (or same with uper and lower in noniso case)
    # quant.dev_trans_wg,
    # quant.dev_trans_weight_band,
    # quant.dev_contr_func_band,
    # quant.dev_gauss_weight,
    # quant.dev_planckband_lay,
    # quant.epsi,
    computer.calculate_contribution_function(keeper)

    # (quant.dev_T_lay,
    # quant.dev_entr_temp,
    # quant.dev_p_lay,
    # quant.dev_entr_press,
    # quant.dev_entropy_lay,
    # quant.dev_opac_entropy,
    # quant.entr_npress,
    # quant.entr_ntemp,
    computer.interpolate_entropy(keeper)
    # calculates the atmospheric Planck and Rosseland mean opacities
    # quant.dev_planck_opac_T_pl,
    # quant.dev_ross_opac_T_pl,
    # quant.dev_planck_opac_T_star,
    # quant.dev_ross_opac_T_star,
    # quant.dev_opac_wg_lay,
    # quant.dev_cloud_opac_lay,
    # quant.dev_planckband_lay,
    # quant.dev_opac_interwave,
    # quant.dev_opac_deltawave,
    # quant.dev_T_lay,
    # quant.dev_gauss_weight,
    # quant.dev_opac_y,
    # quant.dev_opac_band_lay,
    computer.calculate_mean_opacities(keeper)
    # integrates the spectral direct beam flux first over each bin and then the whole spectral range
    # (quant.dev_F_dir_tot,
    # quant.dev_F_dir_band,
    # quant.dev_opac_deltawave,
    # quant.dev_gauss_weight,
    computer.integrate_beamflux(keeper)

    # copy everything back to host and write to files
    ##########################
    # if self.dev_delta_colmass is not None:
    #     print("Copy from device:",
    #           np.uint64(self.dev_delta_colmass),
    #           type(self.dev_delta_colmass),
    #           (self.nlayer),
    #           np.float64)
    ##########################
    # stuff to copy back

    nlayer = keeper.nlayer
    nbin = keeper.nbin
    ninterface = keeper.ninterface
    cuda.Context.synchronize()
    print(f"scat_cross_lay_ptr: {dev_scat_cross_section_lay_ptr}")
    keeper.scat_cross_lay = cuda.from_device(dev_scat_cross_section_lay_ptr,
                                             (nlayer*nbin),
                                             np.float64)

    # print(f"scat_cross_lay_ptr: {dev_scat_cross_section_lay_ptr}")
    keeper.planckband_int = cuda.from_device(dev_planck_int_ptr,
                                             (ninterface*nbin),
                                             np.float64)
    # print(f"scat_cross_int_ptr: {dev_scat_cross_section_int_ptr}")
    keeper.planckband_lay = cuda.from_device(dev_planck_lay_ptr,
                                             ((nlayer+2)*nbin),
                                             np.float64)
    delta_colmass = cuda.from_device(dev_delta_colmass_ptr,
                                     (keeper.nlayer,),
                                     np.float64)
    keeper.delta_colmass = delta_colmass

    keeper.scat_cross_int = cuda.from_device(dev_scat_cross_section_int_ptr,
                                             (ninterface*nbin),
                                             np.float64)

    keeper.meanmolmass_lay = cuda.from_device(dev_meanmolmass_ptr,
                                              (nlayer),
                                              np.float64)

    cuda.Context.synchronize()

    keeper.copy_device_to_host()
    ##########################

    hsfunc.calculate_conv_flux(keeper)
    hsfunc.calc_F_ratio(keeper)
    writer.write_info(keeper, reader, Vmodder)
    writer.write_colmass_mu_cp_entropy(keeper, reader)
    writer.write_integrated_flux(keeper, reader)
    writer.write_downward_spectral_flux(keeper, reader)
    writer.write_upward_spectral_flux(keeper, reader)
    writer.write_TOA_flux_eclipse_depth(keeper, reader)
    writer.write_direct_spectral_beam_flux(keeper, reader)
    writer.write_planck_interface(keeper, reader)
    writer.write_planck_center(keeper, reader)
    writer.write_tp(keeper, reader)
    writer.write_tp_cut(keeper, reader)
    writer.write_opacities(keeper, reader)
    writer.write_Rayleigh_cross_sections(keeper, reader)
    writer.write_cloud_scat_cross_sections(keeper, reader)
    writer.write_cloud_absorption(keeper, reader)
    writer.write_g_0(keeper, reader)
    writer.write_transmission(keeper, reader)
    writer.write_opt_depth(keeper, reader)
    writer.write_trans_weight_function(keeper, reader)
    writer.write_contribution_function(keeper, reader)
    writer.write_mean_extinction(keeper, reader)
    writer.write_flux_ratio_only(keeper, reader)
    if Vmodder.V_coupling == 1:
        Vmodder.write_tp_VULCAN(keeper)
    if keeper.approx_f == 1:
        hsfunc.calc_tau_lw_sw(keeper, reader)

    # prints the success message - yay!
    hsfunc.success_message(keeper)

    if Vmodder.V_coupling == 1:
        Vmodder.test_coupling_convergence(keeper)

    pylfrodull.deinit_alfrodull()


def main():
    """ runs the HELIOS RT computation if this file is executed """

    if __name__ == "__main__":

        run_helios()


main()
