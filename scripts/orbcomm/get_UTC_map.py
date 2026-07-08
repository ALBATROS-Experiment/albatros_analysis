import sys
import os
import numpy as np
import json
import argparse
import matplotlib.pyplot as plt

def get_prediction_error(res, spectra):
    '''Get the error on the predicted UTC discrepancy given residual matrix'''
    A = np.column_stack((spectra, np.ones(len(spectra))))
    N = np.cov(res)
    print(N)

    if res.ndim >1:
        v1 = np.linalg.inv(A.T@np.linalg.inv(N)@A)
    else:
        v1 = N*np.linalg.inv(A.T@A)

    return A@v1@A.T

def get_MAD(data, axis=None):
    '''getting the real median absolute deviation'''
    data_median = np.median(data, axis=axis, keepdims=True)
    abs_deviations = np.abs(data - data_median)
    mad = np.median(abs_deviations, axis=axis)
    return mad


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_start_ts", type=int)
    args = parser.parse_args()
    batch_start_ts = args.batch_start_ts

    #extract some data
    path_discrepancies = f'/scratch/thomasb/batch_{batch_start_ts}/timing_discrepancies'
    path_map = os.path.join(path_discrepancies, 'times_all.json')
    with open(path_map, "r") as f:
            map = json.load(f)

    utc, spectra = [], []
    for fname, pulse in map.items():
        if "offset_fitted" in pulse:
            utc.append(pulse['offset_fitted']/1000 + pulse["pulse_start_ts"])
            spectra.append(pulse["start_specnum"])
    utc = np.array(utc)
    spectra=np.array(spectra)

    #do some math
    UTC_per_spec, UTC_offset = np.polyfit(spectra, utc, 1)
    res = np.asarray(utc)-UTC_per_spec*np.asarray(spectra)-UTC_offset
    res_std = np.std(res)
    res_MAD = get_MAD(res)

    cov = get_prediction_error(res, spectra)
    pred_var = np.diag(cov)
    pred_error = np.sqrt(pred_var)
    mean_err = np.mean(pred_error)

    print('Seconds per spectra', UTC_per_spec)
    print('Initial UTC time:', UTC_offset)
    print(f'Standard deviation of residuals: {res_std} secs')
    print(f"Residual MAD: {res_MAD}")
    print('Mean error on UTC given a spectrum number:', mean_err)

    #plot some stuff
    utc_plotting = utc-batch_start_ts

    fig_fit, ax = plt.subplots()
    plt.scatter(spectra, utc_plotting, c='r', label='Pulse Results')
    plt.plot(spectra, (UTC_per_spec*spectra + UTC_offset)-batch_start_ts, c='b', label='Linear Fit')
    plt.xlabel('Spectrum Number')
    plt.ylabel('UTC Time (s)')
    plt.tight_layout()
    fig_fit.savefig(os.path.join(path_discrepancies, 'fit.png'))
    plt.close(fig_fit)

    fig_res, ax = plt.subplots()
    plt.scatter(utc_plotting, res)
    plt.xlabel('UTC Time (s)')
    plt.ylabel('Residuals (s)')
    plt.hlines(y=0,xmin=np.min(utc_plotting), xmax=np.max(utc_plotting),color='black',linestyle='--')
    plt.tight_layout()
    fig_res.savefig(os.path.join(path_discrepancies, 'fit_residuals.png'))
    plt.close(fig_res)



    #====================================
    # figure with both together (from chat)
    # fig, (ax_fit, ax_res) = plt.subplots(2, 1,figsize=(8, 6),sharex=True,gridspec_kw={'height_ratios': [3, 1]})
    # ax_fit.scatter(spectra, utc_plotting, c='r', label='Pulse Results')
    # ax_fit.plot(spectra,(UTC_per_spec * spectra + UTC_offset) - batch_start_ts,c='b',label='Linear Fit')
    # ax_fit.set_ylabel('UTC Time (s)')
    # ax_fit.legend()

    # ax_res.scatter(spectra, res)
    # ax_res.axhline(0, color='black', linestyle='--')
    # ax_res.set_xlabel('Spectrum Number')
    # ax_res.set_ylabel('Res. (s)')

    # plt.tight_layout()
    # fig.savefig(os.path.join(path_discrepancies, 'fit_and_residuals.png'))
    # plt.close(fig)
    #====================================

    
    #save some stuff
    map['fit'] = {
        'UTC_per_spec': UTC_per_spec, 
        'UTC_offset': UTC_offset,
        'res_std': res_std,
        'res_MAD': res_MAD,
        'mean_err': mean_err
        }
    with open(path_map, "w") as f:
            json.dump(map, f, indent=4)