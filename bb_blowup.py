import json
from pathlib import Path
import numpy as np
# import pandas as pd

import xtrack as xt
import xpart as xp
import xobjects as xo
import xfields as xf
import pickle

from cpymad.madx import Madx
from xtrack.slicing import Teapot, Strategy
import argparse

REPOSITORY_TOP_LEVEL = Path(__file__).resolve().parent.parent
REFERENCE_FILE = json.load(open(REPOSITORY_TOP_LEVEL/"reference_parameters.json"))

def make_bb_lens(nb, phi, sigma_z, alpha, n_slices, other_beam_q0,
                  sigma_x, sigma_px, sigma_y, sigma_py, beamstrahlung_on=False, compt_x_min=1, binning_mode="unicharge"):

    slicer = xf.TempSlicer(n_slices=n_slices, sigma_z=sigma_z, mode=binning_mode)

    el_beambeam = xf.BeamBeamBiGaussian3D(
            config_for_update = None,
            other_beam_q0=other_beam_q0,
            phi=phi, # half-crossing angle in radians
            alpha=alpha, # crossing plane
            # decide between round or elliptical kick formula
            min_sigma_diff = 1e-28,
            # slice intensity [num. real particles] n_slices inferred from length of this
            slices_other_beam_num_particles = slicer.bin_weights * nb,
            # unboosted strong beam moments
            slices_other_beam_zeta_center = slicer.bin_centers,
            slices_other_beam_Sigma_11    = n_slices*[sigma_x**2], # Beam sizes for the other beam, assuming the same is approximation
            slices_other_beam_Sigma_22    = n_slices*[sigma_px**2],
            slices_other_beam_Sigma_33    = n_slices*[sigma_y**2],
            slices_other_beam_Sigma_44    = n_slices*[sigma_py**2],
            # only if BS on
            slices_other_beam_zeta_bin_width_star_beamstrahlung = None if not beamstrahlung_on else slicer.bin_widths_beamstrahlung / np.cos(phi),  # boosted dz
            # has to be set
            slices_other_beam_Sigma_12    = n_slices*[0],
            slices_other_beam_Sigma_34    = n_slices*[0],
            compt_x_min = compt_x_min,
        )
    el_beambeam.iscollective = False # Disable in twiss

    return el_beambeam

def insert_beambeam_elements(line, bb_def_list, twiss_table, emit):

    print(f"Beam-beam definitions provided, installing beam-beam elements at: {', '.join([bbd['at_element'] for bbd in bb_def_list])}")

    for bb_def in bb_def_list:
        element_name = bb_def['at_element']
 
        # the beam-beam lenses are thin and have no effects on optics so no need to re-compute twiss
        element_twiss_index = list(twiss_table.name).index(element_name)

        # get the line index every time as it changes when elements are installed
        element_line_index = line.element_names.index(element_name)
        sigmas = twiss_table.get_betatron_sigmas(*emit if hasattr(emit, '__iter__') else (emit, emit))

        bb_elem = make_bb_lens(nb=float(bb_def['bunch_intensity']),
                                phi=float(bb_def['crossing_angle']),
                                sigma_z=float(bb_def['sigma_z']),
                                n_slices=int(bb_def['n_slices']),
                                other_beam_q0=int(bb_def['other_beam_q0']),
                                alpha=bb_def['alpha'], # Put it to zero, it is okay for this use case
                                sigma_x =np.sqrt(sigmas['Sigma11'][element_twiss_index]),
                                sigma_px=np.sqrt(sigmas['Sigma22'][element_twiss_index]),
                                sigma_y =np.sqrt(sigmas['Sigma33'][element_twiss_index]),
                                sigma_py=np.sqrt(sigmas['Sigma44'][element_twiss_index]),
                                beamstrahlung_on=bb_def['beamstrahlung'], compt_x_min=bb_def["compt_x_min"], binning_mode=bb_def["binning_mode"])

        line.insert_element(index=element_line_index,
                            element=bb_elem,
                            name=f'beambeam_{element_name}')

def load_file_and_set_beam(mode):

    #  Load thick Mad-X lattice and convert to Xtrack line

    filename = Path(f"../lattices/{mode}/fccee_{mode}_wigglers_thin.json")

    with open(f'{filename}', 'r', encoding='utf-8') as fid:
        loaded_dct = json.load(fid)
    line = xt.Line.from_dict(loaded_dct)
        
    ref_particle = xp.Particles(
        mass0=xp.ELECTRON_MASS_EV,
        q0=1,
        p0c=REFERENCE_FILE[MODE]['ENERGY']*10**9
        )
    line.particle_ref = ref_particle
    line.vars['on_wiggler_v'] = REFERENCE_FILE[MODE]['INIT_GUESS_WIG']
    line.config.XTRACK_USE_EXACT_DRIFTS = True

    return line, ref_particle

def match_vertical_emittance(line, gemitt_y_target):

    line.configure_radiation(model='mean')
    line.compensate_radiation_energy_loss()
    
    opt = line.match(
        solve=True,
        eneloss_and_damping=True,
        compensate_radiation_energy_loss=True,
        targets=[
            xt.Target(eq_gemitt_y=gemitt_y_target, tol=1e-16, optimize_log=True)],
        vary=xt.Vary('on_wiggler_v', step=0.01, limits=(0.01, 0.8))
    )
    print(f"[exec.py] emittance successfully matched with 'on_wiggler_v'={line.vars.val['on_wiggler_v']}, 'init_guess'={REFERENCE_FILE[MODE]['INIT_GUESS_WIG']}")
    opt.target_status()
    return line

def main(mode='z'):

    xline, ref_particle = load_file_and_set_beam(mode=MODE)
    xline.cycle('frf.1', inplace=True)
    for gemitt_y_target in [REFERENCE_FILE[MODE]["EMITTANCE_Y"]]:
        line = xline.copy()
        context = xo.ContextCpu()
        line.build_tracker(_context=context)
        print(f"Matching vertical emittance to {gemitt_y_target}\n")
        line = match_vertical_emittance(line, gemitt_y_target)
        run_tracking(ref_particle=ref_particle, line=line, emit_y=REFERENCE_FILE[MODE]["EMITTANCE_Y_COLL"])

def run_tracking(ref_particle, line, emit_y):

    line.discard_tracker()
    monitor_at = 'frf.1'
    monitor = xt.ParticlesMonitor(
        num_particles=num_particles,
        start_at_turn=0, 
        stop_at_turn=nturns
        )
    line.insert_element(element=monitor, name='frf_monitor', index=monitor_at)

    monitor_at = 'ip.1'
    monitor = xt.ParticlesMonitor(
        num_particles=num_particles,
        start_at_turn=0, 
        stop_at_turn=nturns
        )
    line.insert_element(element=monitor, name='ip_monitor', index=monitor_at)

    tw_rad = line.twiss(eneloss_and_damping=True)
    ex = tw_rad.eq_gemitt_x
    ey = tw_rad.eq_gemitt_y
    ez = tw_rad.eq_gemitt_zeta

    nemitt_x=ex * line.particle_ref.beta0[0] * line.particle_ref.gamma0[0]
    nemitt_y=emit_y * line.particle_ref.beta0[0] * line.particle_ref.gamma0[0]
    sigma_z = REFERENCE_FILE[MODE]['SIGMA_Z_BS']

    print(f"matched ex: {ex:.4e}, ey: {ey:.4e}, ez: {ez:.4e}")

    print(f"Tunes are qx: {tw_rad.qx:.4f}, qy: {tw_rad.qy:.4f}, qs: {tw_rad.qs:.4f}")

    print(f"alpha_c: {tw_rad.momentum_compaction_factor:.4e}, circumference:{tw_rad.circumference:.4f}")

    particles = xp.generate_matched_gaussian_bunch(
        num_particles=num_particles,
        total_intensity_particles=REFERENCE_FILE[MODE]["BUNCH_POPULATION"],
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
        sigma_z=sigma_z,  # depends on flag_beamstrahlung, 2 cases: no bb, bb+bs
        line=line)

    line.particle_ref = ref_particle
    line.build_tracker()
    line.configure_radiation(model='mean')
    line.compensate_radiation_energy_loss()

    line.discard_tracker()
    bb_element_list = ['ip.1', 'ip.3', 'ip.5', 'ip.7'] # marker names
    bb_params = dict(
        bunch_intensity=int(REFERENCE_FILE[MODE]['BUNCH_POPULATION']),
        crossing_angle=15e-3, n_slices=REFERENCE_FILE[MODE]['BB_SLICES'], other_beam_q0=-1, alpha=0,
        sigma_z=sigma_z,
        beamstrahlung = True, compt_x_min = 1e-4, binning_mode = "shatilov"
    )
    bb_def_list = [dict(**{'at_element': name}, **bb_params) for name in bb_element_list]
    insert_beambeam_elements(
        line=line,
        bb_def_list=bb_def_list,
        twiss_table=tw_rad,
        emit=(nemitt_x, nemitt_y)
    )
    line.build_tracker()
    line.configure_radiation(model='quantum', model_beamstrahlung='quantum')

    line.track(particles, num_turns=nturns, turn_by_turn_monitor=True, time=True)
    print(f'[exec.py] Tracked in {line.time_last_track} seconds')
    
    # Save frf data    
    scenter_monitor_idx = np.argwhere(np.array(line.element_names)=="frf_monitor")[0][0]
    coords_dict = line.elements[scenter_monitor_idx].to_dict()["data"]
    pickle_path = Path(REPOSITORY_TOP_LEVEL/f"public/{MODE}/coords_dict_frf_{emit_y}_ey_{MODE}.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(coords_dict, f)
    # Save IP data    
    ip_monitor_idx = np.argwhere(np.array(line.element_names)=="ip_monitor")[0][0]
    coords_dict = line.elements[ip_monitor_idx].to_dict()["data"]
    pickle_path = Path(REPOSITORY_TOP_LEVEL/f"public/{MODE}/coords_dict_ip_{emit_y}_ey_{MODE}.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(coords_dict, f)
    
    print(f"[exec.py] Saved pickle file for {emit_y:.4e}_ey_{MODE} \n")

# Script Mode ------------------------------------------------------------------

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run tracking for one operation mode")
    parser.add_argument("--operation_mode", type=str, required=True, help="Operation mode")
    args = parser.parse_args()
    MODE = args.operation_mode
    nturns = REFERENCE_FILE[MODE]['TRACK_TURNS']
    num_particles = REFERENCE_FILE[MODE]['MACRO_PARTICLES']
    main(mode=MODE)
