import io
import json
import os

import matplotlib.pyplot as plt
from coolerClass import Cooler
from json_extender import ExtendedJSONDecoder
from openfermion import get_sparse_operator

from data_manager import ExperimentDataManager
from fauplotstyle.styler import use_style
from fauvqe.models import FermiHubbardModel
from fauvqe.utilities import jw_eigenspectrum_at_particle_number
from fermionic_cooling.plotting.plot_comparison_adiabatic_preprocessing import (
    plot_comparison_fast_sweep,
)
from fermionic_cooling.plotting.plot_each_coupler import plot_each_coupler_perf
from fermionic_cooling.plotting.plot_fastsweep_m_fid import plot_fast_sweep_vs_m
from fermionic_cooling.plotting.plot_thermal_state_decomposition import (
    plot_amplitudes_vs_beta,
)
from fermionic_cooling.plotting.plot_thermalizing_vs_beta_init import (
    afternoon_plot,
    plot_single,
)


def show_if_dry(dry_run: bool):
    if dry_run:
        plt.show()


def get_spectrum(x, y, tunneling, coulomb, n_electrons):
    model = FermiHubbardModel(
        x_dimension=x, y_dimension=y, tunneling=tunneling, coulomb=coulomb
    )
    eig_energies, _ = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(model.fock_hamiltonian),
        particle_number=n_electrons,
    )
    return eig_energies


def load_json(fpath: os.PathLike):
    jobj = json.loads(
        io.open(fpath, encoding="utf8", mode="r").read(),
        cls=ExtendedJSONDecoder,
    )
    return jobj


def load_controlled_cooling_data(fpath):
    jobj = load_json(fpath)
    omegas = jobj["omegas"]
    fidelities = jobj["fidelities"]
    env_energies = jobj["env_energies"]
    return omegas, fidelities, env_energies


def controlled_cooling_load_plot(edm, fpath, fig_filename, sys_eig_energies):
    omegas, fidelities, env_energies = load_controlled_cooling_data(fpath)
    fig = Cooler.plot_controlled_cooling(
        fidelities=fidelities,
        env_energies=env_energies,
        omegas=omegas,
        eigenspectrums=[sys_eig_energies - sys_eig_energies[0]],
    )
    edm.save_figure(fig, fig_filename, add_timestamp=False)


use_style()
dry_run = False
edm = ExperimentDataManager(
    experiment_name="fermionic_cooling_paper_graphs", dry_run=dry_run
)


# thermal_results_1by2
dirnames = afternoon_plot()
fig = plot_single(dirnames)
show_if_dry(dry_run)
edm.save_figure(fig, "thermal_results_1by2", add_timestamp=False)

# nice cooling

sys_eig_energies = get_spectrum(2, 2, 1, 2, [2, 2])

fpath = r"c:\Users\Moi4\Desktop\current\FAU\phd\projects\cooling_fermions\graph_data\fh_new_freecouplers_11h41\run_00000\data\cooling_free_2024_02_27_13_50_01.json"
fig_filename = "nice_cooling"
controlled_cooling_load_plot(edm, fpath, fig_filename, sys_eig_energies)
show_if_dry(dry_run)
# 1e5 depol

fpath = r"c:\Users\Moi4\Desktop\current\FAU\phd\projects\cooling_fermions\graph_data\fh_new_freecouplers_13h58\run_00000\data\cooling_free_2024_02_27_16_18_15.json"
fig_filename = "depol_1e5"
controlled_cooling_load_plot(edm, fpath, fig_filename, sys_eig_energies)
show_if_dry(dry_run)
# 1e5 fermion depol

fpath = r"c:\Users\Moi4\Desktop\current\FAU\phd\projects\cooling_fermions\graph_data\fh_new_freecouplers_17h41\run_00000\data\cooling_free_2024_02_27_19_54_06.json"
fig_filename = "depol_fermion_1e5"
controlled_cooling_load_plot(edm, fpath, fig_filename, sys_eig_energies)
show_if_dry(dry_run)

# controlled therm

fpath = r"c:\Users\Moi4\Desktop\current\FAU\phd\projects\cooling_fermions\graph_data\fh22_0_target_beta_11h04\run_00000\data\cooling_free_2024_02_29_08_51_07.json"
fig_filename = "bigbbrain_thermal"
controlled_cooling_load_plot(edm, fpath, fig_filename, sys_eig_energies)
show_if_dry(dry_run)

# fast sweep v m
dirname = r"c:\Users\Moi4\Desktop\current\FAU\phd\projects\cooling_fermions\graph_data\cooling_with_initial_adiab_sweep_10h10\run_00000\data"
fig = plot_fast_sweep_vs_m(dirname)
show_if_dry(dry_run)
edm.save_figure(fig, "fast_sweep_vs_m", add_timestamp=False)


# each coupler
dirname = r"C:\Users\Moi4\Desktop\current\FAU\phd\projects\cooling_fermions\graph_data\fh22_oneatatime_09h25"
fig = plot_each_coupler_perf(dirname)
show_if_dry(dry_run)
edm.save_figure(fig, "plot_each_coupler_resonance", add_timestamp=False)

# preprocess_adiabsweep

with_adiab = r"c:\Users\Moi4\Desktop\current\FAU\phd\projects\cooling_fermions\graph_data\cooling_with_initial_adiab_sweep_13h33\run_00000\data\cooling_free_adiabatic_2024_02_07_14_04_53.json"
wout_adiab = r"c:\Users\Moi4\Desktop\current\FAU\phd\projects\cooling_fermions\graph_data\cooling_with_initial_adiab_sweep_13h33\run_00000\data\cooling_free_none_2024_02_07_14_37_28.json"
fig_filename = "preprocess_adiabsweep"
fig = plot_comparison_fast_sweep(
    with_adiab=with_adiab, wout_adiab=wout_adiab, sys_eig_energies=sys_eig_energies
)
show_if_dry(dry_run)
edm.save_figure(fig, fig_filename, add_timestamp=False)


if dry_run:
    n_steps = 10
else:
    n_steps = 200

# fh_12_11_components_vs_beta

fig = plot_amplitudes_vs_beta(2, 1, 1, 2, [1, 1], False, n_steps)
fig_filename = "fh_12_11_components_vs_beta"
show_if_dry(dry_run)
edm.save_figure(fig, fig_filename, add_timestamp=False)


# fh22_22_components_vs_beta


fig = plot_amplitudes_vs_beta(2, 2, 1, 2, [2, 2], False, n_steps)
fig_filename = "fh22_22_components_vs_beta"
show_if_dry(dry_run)
edm.save_figure(fig, fig_filename, add_timestamp=False)
