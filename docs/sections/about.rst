About
=====

HELIOS is an open-source radiative transfer code, which is constructed for studying exoplanetary atmospheres in their full variety. The model atmospheres are one-dimensional and plane-parallel, and the equation of radiative transfer is solved in the hemispheric two-stream approximation with non-isotropic scattering. For given opacities and planetary parameters, HELIOS finds the atmospheric temperature profile in radiative-convective equilibrium and the corresponding planetary emission spectrum.

HELIOS is part of the Exoclimes Simulation Platform (`ESP <http://www.exoclime.org>`_).

The optimal application of HELIOS is in combination with the equilibrium chemistry solver `FASTCHEM <https://github.com/exoclime/FASTCHEM/>`_ and the opacity calculator `HELIOS-K <https://github.com/exoclime/HELIOS-K/>`_. They may be used to compute the equilibrium chemical abundances and the opacities, respectively. The opacity table is constructed with a small k-table generator, included in the HELIOS package.

Do not worry! There are sample files included in the installation package as reference that allow HELIOS to run even without those dependencies.

If you use HELIOS for your own work, please cite its two method papers: `Malik et al. 2017 <http://adsabs.harvard.edu/abs/2017AJ....153...56M>`_ and `Malik et al. 2019 <https://ui.adsabs.harvard.edu/abs/2019AJ....157..170M/>`_.

Any questions, issues or bug reports are appreciated and can be sent to *malik@umd.edu*. 

Thank you for considering HELIOS!

