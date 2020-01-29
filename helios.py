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

    if Vmodder.V_coupling == 1:
        Vmodder.read_or_create_iter_count()
        Vmodder.read_species()
        Vmodder.read_molecular_opacities(keeper)
        Vmodder.read_layer_molecular_abundance(keeper)
    reader.read_opac_file(keeper, Vmodder)
    reader.read_entropy_table(keeper)
    cloudy.main_cloud_method(keeper)
    keeper.dimensions()
    reader.read_star(keeper)
    hsfunc.planet_param(keeper, reader)
    hsfunc.set_up_numerical_parameters(keeper)
    hsfunc.initial_temp(keeper, reader, Vmodder)
    if keeper.approx_f == 1:
        hsfunc.approx_f_from_formula(keeper)
    hsfunc.calc_F_intern(keeper)

    # get ready for GPU computations
    keeper.create_zero_arrays(Vmodder)
    keeper.convert_input_list_to_array(Vmodder)
    keeper.copy_host_to_device(Vmodder)
    keeper.allocate_on_device(Vmodder)

    ##########################
    pylfrodull.init_parameters(keeper.nlayer, keeper.iso)

    pylfrodull.allocate()
    ##########################
    # conduct the GPU core computations
    computer.construct_planck_table(keeper)
    computer.correct_incident_energy(keeper)
    computer.construct_grid(keeper)

    if Vmodder.V_coupling == 1:
        if Vmodder.V_iter_nr > 0:
            Vmodder.interpolate_f_molecule_and_meanmolmass(keeper)
            Vmodder.combine_to_scat_cross(keeper)

    computer.radiation_loop(keeper, writer, plotter, Vmodder)
    computer.convection_loop(keeper, writer, plotter, Vmodder)
    ##########################
    dev_scat_cross_section_lay_ptr = 0
    dev_scat_cross_section_int_ptr = 0
    dev_interwave_ptr = 0
    dev_deltawave_ptr = 0
    (dev_scat_cross_section_int_ptr,
     dev_scat_cross_section_lay_ptr,
     dev_interwave_ptr,
     dev_deltawave_ptr) = pylfrodull.get_dev_pointers()

    ##########################
    computer.integrate_optdepth_transmission(keeper)
    computer.calculate_contribution_function(keeper)
    computer.interpolate_entropy(keeper)
    computer.calculate_mean_opacities(keeper)
    computer.integrate_beamflux(keeper)

    # copy everything back to host and write to files
    keeper.copy_device_to_host()
    nlayer = keeper.nlayer
    nbin = keeper.nbin
    ninterface = keeper.ninterface

    print(f"scat_cross_lay_ptr: {dev_scat_cross_section_lay_ptr}")
    keeper.scat_cross_lay = cuda.from_device(dev_scat_cross_section_lay_ptr,
                                             (nlayer*nbin),
                                             np.float64)
    print(f"scat_cross_int_ptr: {dev_scat_cross_section_int_ptr}")
    keeper.scat_cross_int = cuda.from_device(dev_scat_cross_section_int_ptr,
                                             (ninterface*nbin),
                                             np.float64)

    hsfunc.calculate_conv_flux(keeper)
    hsfunc.calc_F_ratio(keeper)
    writer.write_info(keeper, reader, Vmodder)
    writer.write_colmass_mu_cp_entropy(keeper)
    writer.write_integrated_flux(keeper)
    writer.write_downward_spectral_flux(keeper)
    writer.write_upward_spectral_flux(keeper)
    writer.write_TOA_flux_eclipse_depth(keeper)
    writer.write_direct_spectral_beam_flux(keeper)
    writer.write_planck_interface(keeper)
    writer.write_planck_center(keeper)
    writer.write_tp(keeper)
    writer.write_tp_pure(keeper)
    writer.write_tp_cut(keeper)
    writer.write_opacities(keeper)
    writer.write_Rayleigh_cross_sections(keeper)
    writer.write_cloud_scat_cross_sections(keeper)
    writer.write_cloud_absorption(keeper)
    writer.write_g_0(keeper)
    writer.write_transmission(keeper)
    writer.write_opt_depth(keeper)
    writer.write_trans_weight_function(keeper)
    writer.write_contribution_function(keeper)
    writer.write_mean_extinction(keeper)
    writer.write_T10(keeper)
    writer.write_TOA_flux_Ang(keeper)
    writer.write_flux_ratio_only(keeper)
    if Vmodder.V_coupling == 1:
        Vmodder.write_tp_VULCAN(keeper)
    hsfunc.calc_tau_lw_sw(keeper)  # for Daniel Koll collaboration

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
