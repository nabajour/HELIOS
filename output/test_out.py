#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare two cheopsim outputs
"""

import argparse
from pathlib import Path
import difflib

parser = argparse.ArgumentParser(description='Compare two HELIOS datasets')
parser.add_argument('folder1', default="0")
parser.add_argument('folder2', default="pyfrodull")

args = parser.parse_args()
folder_name1 = args.folder1
folder_name2 = args.folder2

folder1 = Path(folder_name1)
folder2 = Path(folder_name2)

file_templates = [
    "_cloud_absorption.dat",
    "_cloud_scat_cross_sect.dat",
    "_colmass_mu_cp_kappa_entropy.dat",
    "_contribution.dat",
    "_direct_beamflux.dat",
    "_g_0.dat",
    "_integrated_flux.dat",
    "_mean_extinct.dat",
    "_opacities.dat",
    "_optdepth.dat",
    "_output_info.dat",
    "_planck_cent.dat",
    "_planck_int.dat",
    # "_ratio_only.dat",
    "_Rayleigh_cross_sect.dat",
    # "_restart_tp.dat",
    "_spec_downflux.dat",
    "_spec_upflux.dat",
    # "_T10.dat",
    # "_tau_lw_sw.dat",
    # "_TOA_flux_Ang.dat",
    "_TOA_flux_eclipse.dat",
    "_tp_cut.dat",
    "_tp.dat",
    # "_tp_only.dat",
    "_transmission.dat",
    "_transweight.dat"
]

for base_name in file_templates:

    f1 = folder1 / (folder_name1 + base_name)
    f2 = folder2 / (folder_name2 + base_name)
    a = f1.open("r").readlines()
    b = f2.open("r").readlines()
    s = difflib.SequenceMatcher(a=a, b=b)

    opcodes_count = {}
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        # print('{:7}   a[{}:{}] --> b[{}:{}] {!r:>8} --> {!r}'.format(
        #    tag, i1, i2, j1, j2, a[i1:i2], b[j1:j2]))
        if tag in opcodes_count:
            opcodes_count[tag] += 1
        else:
            opcodes_count[tag] = 1

    if (('equal' in opcodes_count and opcodes_count['equal'] != 1)
        or 'replace' in opcodes_count
        or 'insert' in opcodes_count
            or 'delete' in opcodes_count):

        print(base_name)
        print(len(base_name)*"-")
        for tag, count in opcodes_count.items():
            print(f"{tag}: {count}")

        print()
