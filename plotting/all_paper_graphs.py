import io
import json
import os

import matplotlib.pyplot as plt
from cooler_class import Cooler
from json_extender import ExtendedJSONDecoder
from openfermion import get_sparse_operator

from data_manager import ExperimentDataManager
from fauplotstyle.styler import use_style
from qutlet.models import FermiHubbardModel
from qutlet.utilities import jw_eigenspectrum_at_particle_number
from fermionic_cooling.plotting.plot_comparison_adiabatic_preprocessing import (
    plot_comparison_fast_sweep,
)
from fermionic_cooling.plotting.plot_fastsweep_m_fid import plot_fast_sweep_vs_m
import numpy as np


from fermionic_cooling.runs.encode_bogos_with_jw import plot_bogo_jw_coefficients


def show_if_dry(dry_run: bool):
    if dry_run:
        plt.show()


def get_spectrum(x, y, tunneling, coulomb, n_electrons):
    model = FermiHubbardModel(
        x_dimension=x,
        y_dimension=y,
        n_electrons=n_electrons,
        tunneling=tunneling,
        coulomb=coulomb,
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


def controlled_cooling_load_plot(
    edm, fpath, fig_filename, sys_eig_energies, tf_minus_val: int = None
):
    omegas, fidelities, env_energies = load_controlled_cooling_data(fpath)
    if tf_minus_val is not None:
        new_env_energies = np.abs(np.array(env_energies[0]) - tf_minus_val)
        env_energies[0] = new_env_energies.astype(list)
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


# # thermal_results_1by2
# dirnames = afternoon_plot()
# fig = plot_single(dirnames)
# show_if_dry(dry_run)
# edm.save_figure(fig, "thermal_results_1by2", add_timestamp=False)

# nice cooling
sys_eig_energies = get_spectrum(2, 2, 1, 2, [2, 2])
jobj = {}
fpath = r"c:\Users\Moi4\Desktop\current\FAU\phd\projects\cooling_fermions\graph_data\fh_new_freecouplers_11h41\run_00000\data\cooling_free_2024_02_27_13_50_01.json"
fig_filename = "nice_cooling"
controlled_cooling_load_plot(edm, fpath, fig_filename, sys_eig_energies)
show_if_dry(dry_run)
jobj[fig_filename] = fpath

# 1e4 depol
fpath = r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2024_03_15\fh_new_freecouplers_15h27\run_00000\data\cooling_free_2024_03_15_15_36_58.json"
fig_filename = "depol_1e4"
controlled_cooling_load_plot(edm, fpath, fig_filename, sys_eig_energies)
show_if_dry(dry_run)
jobj[fig_filename] = fpath


# 1e5 fermion depol
fpath = r"c:\Users\Moi4\Desktop\current\FAU\phd\projects\cooling_fermions\graph_data\fh_new_freecouplers_17h41\run_00000\data\cooling_free_2024_02_27_19_54_06.json"
fig_filename = "depol_fermion_1e5"
controlled_cooling_load_plot(edm, fpath, fig_filename, sys_eig_energies)
show_if_dry(dry_run)
jobj[fig_filename] = fpath


# controlled therm
fpath = r"c:\Users\Moi4\Desktop\current\FAU\phd\projects\cooling_fermions\graph_data\fh22_0_target_beta_11h04\run_00000\data\cooling_free_2024_02_29_08_51_07.json"
fig_filename = "bigbbrain_thermal"
controlled_cooling_load_plot(
    edm, fpath, fig_filename, sys_eig_energies, tf_minus_val=0.5
)
show_if_dry(dry_run)
jobj[fig_filename] = fpath

# fast sweep v m slater
fpath = r"c:\Users\Moi4\Desktop\current\FAU\phd\projects\cooling_fermions\graph_data\cooling_with_initial_adiab_sweep_slater_15h12\run_00000\data"
fig = plot_fast_sweep_vs_m(fpath, 0.49844239875687185)
show_if_dry(dry_run)
edm.save_figure(fig, "fast_sweep_vs_m_slater", add_timestamp=False, fig_shape="regular")
jobj[fig_filename] = fpath

# fast sweep v m coulomb
fpath = r"c:\Users\Moi4\Desktop\current\FAU\phd\projects\cooling_fermions\graph_data\cooling_with_initial_adiab_sweep_08h58\run_00000\data"
fig = plot_fast_sweep_vs_m(fpath, 0.08333327548043287)
show_if_dry(dry_run)
edm.save_figure(
    fig, "fast_sweep_vs_m_coulomb", add_timestamp=False, fig_shape="regular"
)

jobj[fig_filename] = fpath

# # each coupler
# fpath = r"C:\Users\Moi4\Desktop\current\FAU\phd\projects\cooling_fermions\graph_data\fh22_oneatatime_09h25"
# fig = plot_each_coupler_perf(fpath)
# show_if_dry(dry_run)
# edm.save_figure(
#     fig, "plot_each_coupler_resonance", add_timestamp=False, fig_shape="page-wide"
# )
# jobj[fig_filename] = fpath


# single comp
with_adiab = r"c:\Users\Moi4\Desktop\current\FAU\phd\projects\cooling_fermions\graph_data\single_fast_sweep_run\run_00000\data\cooling_free_adiabatic_2024_02_07_14_04_53.json"
wout_adiab = r"c:\Users\Moi4\Desktop\current\FAU\phd\projects\cooling_fermions\graph_data\single_fast_sweep_run\run_00000\data\cooling_free_none_2024_02_07_14_37_28.json"
fig_filename = "fast_sweep_single_comp"
fig = plot_comparison_fast_sweep(
    with_adiab=with_adiab, wout_adiab=wout_adiab, sys_eig_energies=sys_eig_energies
)
show_if_dry(dry_run)
edm.save_figure(fig, fig_filename, add_timestamp=False)

jobj[fig_filename] = {"with_adiab": with_adiab, "wout_adiab": wout_adiab}

# bogos
fpath = r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2024_03_10\jw_encode_bogos_coeff_2_2_2u_2d_15h36\run_00000\data\carissa_00000_2024_03_10_15_38_50.json"
jobj_bogos = load_json(fpath)

total_coefficients = jobj_bogos["total_coefficients"]
max_pauli_strs = jobj_bogos["max_pauli_strs"]

fig = plot_bogo_jw_coefficients(total_coefficients, max_pauli_strs)
fig_filename = "bogo_jw_coefficients"
edm.save_figure(fig, fig_filename, add_timestamp=False)

jobj[fig_filename] = fpath

# if dry_run:
#     n_steps = 10
# else:
#     n_steps = 200

# # fh_12_11_components_vs_beta
# fig = plot_amplitudes_vs_beta(2, 1, 1, 2, [1, 1], False, n_steps)
# fig_filename = "fh_12_11_components_vs_beta"
# show_if_dry(dry_run)
# edm.save_figure(fig, fig_filename, add_timestamp=False)


# # fh22_22_components_vs_beta
# fig = plot_amplitudes_vs_beta(2, 2, 1, 2, [2, 2], False, n_steps)
# fig_filename = "fh22_22_components_vs_beta"
# show_if_dry(dry_run)
# edm.save_figure(fig, fig_filename, add_timestamp=False)


edm.save_dict(jobj)
