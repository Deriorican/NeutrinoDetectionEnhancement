import aa
from collections import defaultdict
import glob
import numpy as np
import os
import pandas as pd
import ROOT
import subprocess
import tarfile

from km3io.definitions import fitparameters as kf
from km3io.definitions import reconstruction as kr
import kadots.reduction.detector_utils as kd
import kadots.utils as ku


AAflux_atmnu = ROOT.Flux_Atmospheric()
ROOT.gSystem.Load("/sps/km3net/users/heijboer/OscProb/libOscProb.so")


server_xrootd = "root://ccxroot.in2p3.fr:1999"
path_data_xrootd = {
    "ORCA11": "/hpss/in2p3.fr/group/km3net/data/KM3NeT_00000132/v7.4/reco/datav7.4.jorcarec.aanet.{run}.root",
    "ORCA18": "/hpss/in2p3.fr/group/km3net/data/KM3NeT_00000148/v8.1/reco/datav8.1.jorcarec.offline.{run}.root",
}
path_mu_xrootd = {
    "ORCA11": "/hpss/in2p3.fr/group/km3net/mc/atm_muon/KM3NeT_00000132/v7.4/reco/",
    "ORCA18": "/hpss/in2p3.fr/group/km3net/mc/atm_muon/KM3NeT_00000148/v8.1/reco/",
}
path_nu_xrootd =  {
    "ORCA11": "/hpss/in2p3.fr/group/km3net/mc/atm_neutrino/KM3NeT_00000132/v7.4/reco/",
    "ORCA18": "/hpss/in2p3.fr/group/km3net/mc/atm_neutrino/KM3NeT_00000148/v8.1/reco/KM3NeT_00000148_{run}.gsg_neutrinos.jppmuon.offline.v8.1.tar",
}

path_mu_irods =  {
    "ORCA11": "/in2p3/km3net/mc/atm_muon/KM3NeT_00000132/v7.4/reco",
    "ORCA18": "/in2p3/km3net/mc/atm_muon/KM3NeT_00000148/v8.1/reco",
}
path_nu_irods = {
    "ORCA11": "/in2p3/km3net/mc/atm_neutrino/KM3NeT_00000132/v7.4/reco",
    "ORCA18": "/in2p3/km3net/mc/atm_neutrino/KM3NeT_00000148/v8.1/reco",
}

path_data_local = "/sps/km3net/users/mlamoure/KM3NeT_processing/data_processing/prod/data/KM3NeT_00000148/v8.1/reco/datav8.1.jorcarec.jsh.aanet.{run}.root"
path_mu_local = "/sps/km3net/users/mlamoure/KM3NeT_processing/data_processing/prod/mc/atm_muon/KM3NeT_00000148/v8.1/reco/mcv8.1.mupage_tuned.sirene.jterbr{run}.jorcarec.jsh.aanet.*.root"
path_nu_local = "/sps/km3net/users/mlamoure/KM3NeT_processing/data_processing/prod/mc/atm_neutrino/KM3NeT_00000148/v8.1/reco/mcv8.1.gsg_{flav}*.jterbr{run}.jorcarec.jsh.aanet.*.root"

path_detx = {
    "ORCA11": "/sps/km3net/repo/data/KM3NeT_00000132/v7.4/detectors/KM3NeT_00000132_{run}.detx",
    "ORCA18": "/sps/km3net/repo/data/KM3NeT_00000148/v8.1/detectors/KM3NeT_00000148_{run}.detx",
}


c = 0.299792458 
n = 1.3800851282
c_medium = c/n


def best_reco_track(evt):
    
    sel = None
    for trk in evt.trks:
        if trk.rec_type != kr.JPP_RECONSTRUCTION_TYPE:
            continue
        if 1 not in trk.rec_stages:
            continue
        if (sel is None) or (trk.rec_stages.size() > sel.rec_stages.size()) or ((trk.rec_stages.size() == sel.rec_stages.size()) and (trk.lik > sel.lik)):
            sel = trk
    return sel


def best_reco_shower(evt):
    
    sel = None
    for trk in evt.trks:
        if trk.rec_type != kr.JPP_RECONSTRUCTION_TYPE:
            continue
        if 101 not in trk.rec_stages:
            continue
        if (sel is None) or (trk.rec_stages.size() > sel.rec_stages.size()) or ((trk.rec_stages.size() == sel.rec_stages.size()) and (trk.lik > sel.lik)):
            sel = trk    
    return sel


def get_files(run, detector="ORCA11", mode="irods"):
    
    suffix = 'offline' if detector=='ORCA18' else 'aanet'
    
    files = defaultdict(list)

    # get data file
    if mode == "local":
        files["data"] += glob.glob(path_data_local.replace("{run}", f"{run:08d}"))
    else:
        files["data"] = [server_xrootd + "/" + path_data_xrootd[detector].replace("{run}", f"{run:08d}")]

    # get muon files
    infiles = subprocess.run(f"ils {path_mu_irods[detector]} | grep jterbr{run:08d}.jorcarec.{suffix} | grep -v .fail",
                             shell=True, check=True, capture_output=True, text=True)
    infiles = infiles.stdout.split('\n')
    infiles = [f.strip() for f in infiles if f.strip() != '']
    for infile in infiles:
        if mode == "irods":
            if not os.path.isfile(f"outs/reduced/raw/{infile}"):
                subprocess.run(f"iget -T {path_mu_irods[detector]}/{infile} outs/reduced/raw/{infile}", shell=True)
            files["muon"].append(f"outs/reduced/raw/{infile}")
        elif mode == "hpss":
            files["muon"].append(f"{server_xrootd}/{path_mu_xrootd[detector]}/{infile}")
        elif mode == "local":
            files["muon"] += glob.glob(path_mu_local.replace("{run}", f"{run:08d}"))

    # get neutrino files
    if mode == "irods" or mode == "hpss":
        tarfile_remote = path_nu_xrootd[detector].replace("{run}", f"{run:08d}")
        tarfile_local = f"{os.getenv('TMPDIR')}/{os.path.basename(tarfile_remote)}"
        if not os.path.isfile(tarfile_local):
            subprocess.run(f"xrdcp {server_xrootd}/{tarfile_remote} {tarfile_local}", shell=True, check=True)
        with tarfile.open(tarfile_local, "r") as tar:
            tar.extractall(path=os.getenv('TMPDIR'))
        os.remove(tarfile_local)
        for flav, fname in zip(["nue", "numu", "nutau", "nuNC"], ["elec-CC", "muon-CC", "tau-CC", "muon-NC"]):
            for sign, pfname in zip(["", "a"], ["", "anti-"]):
                files[sign+flav] = glob.glob(tarfile_local.replace(".tar", ".root").replace("neutrinos", f"{pfname}{fname}*"))
    else:
        for flav, fname in zip(["nue", "numu", "nutau", "nuNC"], ["elec-CC", "muon-CC", "tau-CC", "muon-NC"]):
            for sign, pfname in zip(["", "a"], ["", "anti-"]):
                files[sign+flav] = glob.glob(path_nu_local.replace("{flav}", pfname+fname).replace("{run}", f"{run:08d}"))
    
    files["detector"] = path_detx[detector].replace("{run}", f"{run:08d}")
                    
    return files


def process_one_run(run, detector, force_reprocessing=False, mode="hpss"):

    files = get_files(run, detector, mode)

    # OscProb initialization
    oscprob_PMNS = ROOT.OscProb.PMNS_Fast()
    oscprob_prem = ROOT.OscProb.PremModel()

    # data livetime
    livetime_data = 0
    for filename in files["data"]:
        livetime_data += ROOT.EventFile(filename).header.daq_livetime()

    # get detector information
    det = ROOT.Det(files["detector"])
    detector_center_x = np.average([dom.pos.x for _, dom in det.doms])
    detector_center_y = np.average([dom.pos.y for _, dom in det.doms])
    detector_polygon = kd.detector_containment_polygon(det)  # polygon in XY plan to define starting event
    detector_zrange = kd.detector_containment_zrange(det)  # min/max Z positions of DOMs
    detector_pmt_watchcone = kd.get_pmt_watchcone(det)  # give the number of PMTs in the view range of any other PMT

    def get_R(pos_x, pos_y):
        R2 = (pos_x - detector_center_x)**2 + (pos_y - detector_center_y)**2
        return np.sqrt(R2)

    for name, filenames in files.items():
        
        if name == "detector":
            continue

        # create folder if needed
        outfolder = f"outs/reduced/{detector.lower()}/neutrino" if "nu" in name else f"outs/reduced/{detector.lower()}/{name}"
        os.makedirs(outfolder, exist_ok=True)
        # skip processing if file already exists
        if os.path.isfile(f"{outfolder}/{name}_Run{run:08d}.h5") and not force_reprocessing:
            continue

        if len(filenames) == 0:
            continue

        df = defaultdict(list)
        livetime = 0

        for filename in filenames:

            # skip processing if the file does not open properly
            try:
                file = ROOT.EventFile(filename)
            except:
                continue

            # get muon livetime
            if "muon" in name:
                livetime += file.header.mc_livetime()

            for evt in file:
                det.apply(evt)
                if evt.trks.size() == 0:
                    continue
                trk = best_reco_track(evt)
                shower = best_reco_shower(evt)

                ##########################
                # HITS-RELATED VARIABLES
                ##########################

                # get info from all triggered hits
                hits_t = np.array([h.t for h in evt.hits if h.trig != 0])  # time
                hits_x = np.array([h.pos.x for h in evt.hits if h.trig != 0])  # X position
                hits_y = np.array([h.pos.y for h in evt.hits if h.trig != 0])  # Y position
                hits_z = np.array([h.pos.z for h in evt.hits if h.trig != 0])  # Z position
                hits_dz = np.array([h.dir.z for h in evt.hits if h.trig != 0])  # Z direction
                hits_tot = np.array([h.tot for h in evt.hits if h.trig != 0])  # charge (time-over-threshold)
                hits_sideveto = np.array([detector_pmt_watchcone[h.pmt_id] < 2 for h in evt.hits if h.trig != 0])  # is the hit on the side of the detector
                hits_domid = np.array([h.dom_id for h in evt.hits if h.trig != 0], dtype=int)  # DOM ID
                hits_floorid = np.array([det.doms[int(h.dom_id)].floor_id for h in evt.hits if h.trig != 0], dtype=int)  # floor ID

                # hits charge
                first_hit_z = hits_z[np.argmin(hits_t)]
                max_hit_tot = np.max(hits_tot)
                charge = np.sum(hits_tot)
                charge_above_firsthit = np.sum(hits_tot[hits_z >= first_hit_z])
                x_chargeweighted = np.average(hits_x, weights=hits_tot)
                y_chargeweighted = np.average(hits_y, weights=hits_tot)
                z_chargeweighted = np.average(hits_z, weights=hits_tot)

                quartile = 0.2
                # early/late hits
                t_early_quartile = min(hits_t) + quartile * (max(hits_t) - min(hits_t))
                t_late_quartile = max(hits_t) - quartile * (max(hits_t) - min(hits_t))
                early, late = (hits_t <= t_early_quartile), (hits_t >= t_late_quartile)
                if np.sum(hits_tot[early]) != 0:
                    x_early = np.average(hits_x[early])
                    y_early = np.average(hits_y[early])
                    z_early = np.average(hits_z[early])
                    x_earlyq = np.average(hits_x[early], weights=hits_tot[early])
                    y_earlyq = np.average(hits_y[early], weights=hits_tot[early])
                    z_earlyq = np.average(hits_z[early], weights=hits_tot[early])
                    charge_early = np.sum(hits_tot[early])
                else:
                    x_early = y_early = z_early = x_earlyq = y_earlyq = z_earlyq = charge_early = np.nan
                if np.sum(hits_tot[late]) != 0:
                    x_late = np.average(hits_x[late])
                    y_late = np.average(hits_y[late])
                    z_late = np.average(hits_z[late])
                    x_lateq = np.average(hits_x[late], weights=hits_tot[late])
                    y_lateq = np.average(hits_y[late], weights=hits_tot[late])
                    z_lateq = np.average(hits_z[late], weights=hits_tot[late])
                    charge_late = np.sum(hits_tot[late])
                else:
                    x_late = y_late = z_late = x_lateq = y_lateq = z_lateq = charge_late = np.nan

                # first/last hits
                t_first_quartile, t_last_quartile = np.percentile(hits_t, [100*quartile, 100*(1-quartile)])
                first, last = (hits_t <= t_first_quartile), (hits_t >= t_last_quartile)
                if np.sum(hits_tot[first]) != 0:
                    x_first = np.average(hits_x[first])
                    y_first = np.average(hits_y[first])
                    z_first = np.average(hits_z[first])
                    x_firstq = np.average(hits_x[first], weights=hits_tot[first])
                    y_firstq = np.average(hits_y[first], weights=hits_tot[first])
                    z_firstq = np.average(hits_z[first], weights=hits_tot[first])
                    charge_first = np.sum(hits_tot[first])
                else:
                    x_first = y_first = z_first = x_firstq = y_firstq = z_firstq = charge_first = np.nan
                if np.sum(hits_tot[last]) != 0:
                    x_last = np.average(hits_x[last])
                    y_last = np.average(hits_y[last])
                    z_last = np.average(hits_z[last])
                    x_lastq = np.average(hits_x[last], weights=hits_tot[last])
                    y_lastq = np.average(hits_y[last], weights=hits_tot[last])
                    z_lastq = np.average(hits_z[last], weights=hits_tot[last])
                    charge_last = np.sum(hits_tot[last])
                else:
                    z_last = z_lastq = charge_last = np.nan

                # central hits
                t_avg = np.average(hits_t)
                t_std = np.std(hits_t)
                central = (hits_t > t_avg - t_std) & (hits_t < t_avg - t_std)
                cog_x = np.average(hits_x[central])
                cog_y = np.average(hits_y[central])
                cog_z = np.average(hits_z[central])
                delta_t = np.sqrt(np.power(hits_x[central] - cog_x, 2) + np.power(hits_y[central] -
                                  cog_y, 2) + np.power(hits_z[central] - cog_z, 2)) / c_medium
                cog_t = np.average(hits_t[central] - delta_t)

                # geometry veto (triggered hits)
                top1 = hits_floorid >= 18
                top2 = hits_floorid >= 17
                chargetop1 = np.sum(hits_tot[top1])
                chargetop2 = np.sum(hits_tot[top2])
                n_geometry_veto1_hits = np.count_nonzero(top1)
                n_geometry_veto2_hits = np.count_nonzero(top2)

                # geometry veto (triggered early hits)
                chargetopearly1 = np.sum(hits_tot[top1 & early])
                chargetopearly2 = np.sum(hits_tot[top2 & early])

                # side veto
                chargeside = np.sum(hits_tot[hits_sideveto])
                chargesideearly = np.sum(hits_tot[hits_sideveto & early])

                # loop on hits
                nhits = 0
                nlines, ndoms, ndoms_crk = set(), set(), set()

                for did in hits_domid:
                    nhits += 1
                    nlines.add(det.doms[int(did)].line_id)
                    ndoms.add(did)
                nlines, ndoms = len(nlines), len(ndoms)
                
                ##########################
                # TRACK-RELATED VARIABLES
                ##########################
                
                trk_lik = trk_E = trk_beta0 = trk_beta1 = trk_nhitsJGandalf = trk_npe_mip_total = trk_length = np.nan
                trk_pos_x = trk_pos_y = trk_pos_z = trk_dir_x = trk_dir_y = trk_dir_z = trk_toclosestdom = np.nan
                
                if trk is not None:
                    trk_lik = trk.lik
                    trk_E = trk.E
                    trk_beta0 = trk.fitinf[kf.JGANDALF_BETA0_RAD]
                    trk_beta1 = trk.fitinf[kf.JGANDALF_BETA1_RAD]
                    trk_nhitsJGandalf = trk.fitinf[kf.JGANDALF_NUMBER_OF_HITS]
                    trk_npe_mip_total = trk.fitinf[kf.JSTART_NPE_MIP_TOTAL]
                    trk_length = trk.fitinf[kf.JSTART_LENGTH_METRES]
                    trk_pos_x = trk.pos.x
                    trk_pos_y = trk.pos.y
                    trk_pos_z = trk.pos.z
                    trk_dir_x = trk.dir.x
                    trk_dir_y = trk.dir.y
                    trk_dir_z = trk.dir.z
                    trk_toclosestdom = np.min([(trk.pos - dom.pos).len() for _, dom in det.doms])

                ##########################
                # SHOWER-RELATED VARIABLES
                ##########################

                shower_lik = shower_E = shower_pos_x = shower_pos_y = shower_pos_z = shower_dir_x = shower_dir_y = shower_dir_z = np.nan
                if shower is not None:
                    shower_lik = shower.lik
                    shower_E = shower.E
                    shower_pos_x = shower.pos.x
                    shower_pos_y = shower.pos.y
                    shower_pos_z = shower.pos.z
                    shower_dir_x = shower.dir.x
                    shower_dir_y = shower.dir.y
                    shower_dir_z = shower.dir.z
                    shower_toclosestdom = np.min([(shower.pos - dom.pos).len() for _, dom in det.doms])

                ##########################
                # CHERENKOV-RELATED VARIABLES
                ##########################
                
                nhits_cher = 0
                if trk is not None:
                    for h in evt.hits:
                        if h.trig != 0:
                            continue
                        cher_dt, cher_dist = kd.cherenkov_variables(trk, h)
                        if (np.abs(cher_dt) < 10) and (cher_dist < 100):
                            nhits_cher += 1
                
                ##########################
                # FILL THE DICTIONARY
                ##########################

                # HITS
                df["n_trig_hits"].append(nhits)  # number of PMT hits participating to the trigger
                df["n_trig_doms"].append(ndoms)  # number of DOMs participating to the trigger
                df["n_trig_lines"].append(nlines)  # number of lines participating to the trigger
                df["coc"].append(z_chargeweighted)
                df["tot"].append(charge)  # total charge deposited over the event
                df["max_hit_tot"].append(max_hit_tot)  # max charge in single PMT
                df["z_early"].append(charge_early)  # average Z position of the hits in the early 20% of event duration
                df["z_late"].append(charge_late)  # average Z position of the hits in the late 20% of event duration
                df["z_early_qweighted"].append(z_earlyq)  # average Z position of the hits in the early 20% of event duration (weighted by charge)
                df["z_late_qweighted"].append(z_lateq)  # average Z position of the hits in the late 20% of event duration (weighted by charge)
                df["delta_z_earlylate"].append(z_early - z_late)  # difference between Z position of early and late hits
                df["delta_z_firstlast"].append(z_first - z_last)  # difference between Z position of first 20% and last 20% hits
                df["delta_z_earlylate_qweighted"].append(z_earlyq - z_lateq)  # difference between Z position of early and late hits (weighted by charge)
                df["delta_z_firstlast_qweighted"].append(z_firstq - z_lastq)  # difference between Z position of first 20% and last 20% hits (weighted by charge)
                df["distance_earlylate"].append(np.sqrt((x_early-x_late)**2 + (y_early-y_late)**2 + (z_early-z_late)**2))  # distance between position of early and late hits
                df["distance_earlylate_qweighted"].append(np.sqrt((x_earlyq-x_lateq)**2 + (y_earlyq-y_lateq)**2 + (z_earlyq-z_lateq)**2))  # distance between position of early and late hits (weighted by charge)
                df["distance_firstlast"].append(np.sqrt((x_first-x_last)**2 + (y_first-y_last)**2 + (z_first-z_last)**2))  # distance between position of first and last hits
                df["distance_firstlast_qweighted"].append(np.sqrt((x_firstq-x_lastq)**2 + (y_firstq-y_lastq)**2 + (z_firstq-z_lastq)**2))  # distance between position of first and last hits (weighted by charge)
                df["charge_ratio_abovefirsthit"].append(charge_above_firsthit / charge)  # ratio between the charge in all PMTs above the Z position of the first hit and the total charge
                df["charge_ratio_early"].append(charge_early / charge)  # ratio between the charge of early 20% hits and the total charge
                df["charge_ratio_earlylate"].append(charge_early / charge_late)  # ratio between the charge of early 20% hits and the charge of late 20% hits
                df["charge_ratio_first"].append(charge_first / charge)  # ratio between the charge of first 20% hits and the total charge
                df["charge_ratio_firstlast"].append(charge_first / charge_last)  # ratio between the charge of first 20% hits and the charge of last 20% hits
                df["charge_ratio_veto1"].append(chargetop1 / charge)  # ratio between the total charge in the first floor of PMTs and the total charge
                df["charge_ratio_veto2"].append(chargetop2 / charge)  # ratio between the total charge in the 2 first floor of PMTs and the total charge
                df["charge_ratio_earlyveto1"].append(chargetopearly1 / charge)
                df["charge_ratio_earlyveto2"].append(chargetopearly2 / charge)
                df["charge_ratio_side"].append(chargeside / charge)
                df["charge_ratio_earlyside"].append(chargesideearly / charge)
                
                # CHERENKOV
                df["cher_nhits"].append(nhits_cher)
                
                # TRACK
                df["track_quality"].append(trk_lik)  # fit quality, higher is better
                df["track_energy"].append(trk_E)  # estimated energy in GeV
                df["track_beta0"].append(trk_beta0)  # estimated error on the direction (in radians)
                df["track_beta1"].append(trk_beta1)  # ???
                df["track_nhitsJGandalf"].append(trk_nhitsJGandalf)  # number of hits used in the final reconstruction
                df["track_npe_mip_total"].append(trk_npe_mip_total)
                df["track_length"].append(trk_length)
                df["track_x"].append(trk_pos_x)
                df["track_y"].append(trk_pos_y)
                df["track_z"].append(trk_pos_z)
                df["bestmuon_dx"].append(trk_dir_x)
                df["bestmuon_dy"].append(trk_dir_y)
                df["bestmuon_dz"].append(trk_dir_z)
                df["bestmuon_distance_to_closestdom"].append(trk_toclosestdom)
                
                # SHOWER
                df["shower_quality"].append(shower_lik)
                df["shower_energy"].append(shower_E)
                df["shower_pos_x"].append(shower_pos_x)
                df["shower_pos_y"].append(shower_pos_y)
                df["shower_pos_z"].append(shower_pos_z)
                df["shower_dir_x"].append(shower_dir_x)
                df["shower_dir_y"].append(shower_dir_y)
                df["shower_dir_z"].append(shower_dir_z)
                df["shower_distance_to_closestdom"].append(trk_toclosestdom)

                if name == "data":
                    continue

                # MC-only information
                if name == "muon":
                    mc_trk = evt.mc_trks[0]
                    mc_lep = mc_trk
                else:
                    mc_trk = evt.neutrino()
                    mc_lep = evt.leading_lepton()
                df["mc_dir_x"].append(mc_trk.dir.x)
                df["mc_dir_y"].append(mc_trk.dir.y)
                df["mc_dir_z"].append(mc_trk.dir.z)
                df["mc_pos_x"].append(mc_trk.pos.x)
                df["mc_pos_y"].append(mc_trk.pos.y)
                df["mc_pos_z"].append(mc_trk.pos.z)
                df["mc_energy"].append(mc_trk.E)
                df["mc_starting"].append(kd.is_event_starting(mc_trk.pos, detector_polygon, detector_zrange))
                df["mc_angerr"].append(np.arccos(mc_trk.dir.dot(trk.dir)) if trk else np.inf)
                df["mc_angerr_shower"].append(np.arccos(mc_trk.dir.dot(shower.dir)) if shower else np.inf)
                
                if mc_lep:
                    df["mc_angerr_kin"].append(np.arccos(mc_trk.dir.dot(mc_lep.dir)))
                else:
                    df["mc_angerr_kin"].append(np.nan)
                        
                # MC/neutrino-only information
                if "nu" in name:
                    df["mc_pdg"].append(mc_trk.type)  # 14: numu, -14: anti-numu, 12: nue, 16: nutau, ...
                    df["mc_iscc"].append("CC" in os.path.basename(filename))
                    df["mc_isnc"].append("NC" in os.path.basename(filename))
                    df["mc_w2"].append(evt.w[1])
                    df["mc_ngen"].append(int(file.header.ngen()))
                    ####################
                    # ATMOSPHERIC WEIGHT
                    ####################
                    sign = np.sign(mc_trk.type)
                    # unoscillated weights
                    atm_flux_nue = AAflux_atmnu.dNdEdOmega(int(sign * 12), np.log10(mc_trk.E), mc_trk.dir.z)
                    atm_flux_numu = AAflux_atmnu.dNdEdOmega(int(sign * 14), np.log10(mc_trk.E), mc_trk.dir.z)
                    # OscProb setup
                    oscprob_flav = int((np.abs(mc_trk.type)-12) / 2)
                    oscprob_prem.FillPath(-1. * mc_trk.dir.z)
                    oscprob_PMNS.SetPath(oscprob_prem.GetNuPath())
                    oscprob_PMNS.SetIsNuBar(sign < 0)
                    # compute probabilities + oscillated weights
                    posc_from_nue = oscprob_PMNS.Prob(0, oscprob_flav, mc_trk.E)
                    posc_from_numu = oscprob_PMNS.Prob(1, oscprob_flav, mc_trk.E)
                    atm_wgt = posc_from_nue * atm_flux_nue + posc_from_numu * atm_flux_numu
                    df["weight_bkg"].append(atm_wgt * evt.w[1] * livetime_data / file.header.ngen())
                    ######################
                    # ASTROPHYSICAL WEIGHT
                    ######################
                    sig_wgt = 1e-4 * np.power(mc_trk.E, -2)
                    df["weight_sig"].append(sig_wgt * evt.w[1] * livetime_data / file.header.ngen())

                # MC/muon-only information
                if "muon" in name:
                    df["mc_muonmultiplicity"].append(len(evt.mc_trks)-1)

        # MC weights
        if "muon" in name:
            # no signal weight for muon
            df["weight_sig"] = np.zeros_like(df["n_trig_hits"])
            # simple livetime rescaling
            df["weight_bkg"] = livetime_data / livetime * np.ones_like(df["n_trig_hits"])

        df["run"] = np.full_like(df["n_trig_hits"], run, dtype=int)

        # saving to dataframe
        df = pd.DataFrame(data=df)
        outputfile = f"{outfolder}/{name}_Run{run:08d}.h5"
        print(f"Saving output file {outputfile}")
        df.to_hdf(outputfile, "Evts", "w")
        
    return 


if __name__ == "__main__":
    
    runs = [16545]
    for run in runs:
        process_one_run(run, detector="ORCA18", mode="local", force_reprocessing=True)
