import numpy as np
import cupy as cp
from scipy.signal import firwin
from scipy.fft import next_fast_len
import datetime,time
import sys
import os
from os import path

sys.path.insert(0, path.expanduser("~"))
from albatros_analysis.src.utils import baseband_utils as bu
from albatros_analysis.src.utils import pfb_utils as pu
from albatros_analysis.src.utils import pycufft
from albatros_analysis.src.correlations import baseband_data_classes as bdc

ddc_kernel = cp.ElementwiseKernel(
    in_params='complex64 ts_in, float64 frac_freq, float64 start_cycles',
    out_params='complex64 ts_out',
    operation='''
        const double TWO_PI = 2 * 3.14159265358979323846;
        
        double cycles = frac_freq * i + start_cycles;
        double mod_cycle = cycles - floor(cycles);
        
        double wrapped_phase = mod_cycle * TWO_PI;
        
        //Cast to FP32 and use fast hardware trig
        float im, re;
        sincosf((float)(-wrapped_phase), &im, &re); // exp (-j 2 pi carrer t)
        
        complex<float> phasor(re, im);
        ts_out = ts_in * phasor;
    ''',
    name='ddc_kernel'
)

cfg = {}
cfg["ipp"] = 5.5e-3
cfg["code_baudrate"] = 200e3  # symbols per sec
cfg["code0"] = "0001_0010_0001_1101"  # complementary code pair, code 0
cfg["code1"] = "0001_0010_1110_0010"  # complementary code pair, code 1
cfg["code_repeat_num"] = 10


def get_code_template(code_type="0"):
    per_code_len = int(6 * 16)  # 6x 16 symbols
    total_len = np.round(np.ceil(cfg["ipp"] * cfg["code_baudrate"])).astype(int) #1100
    print("total len is", total_len)
    code_template = np.ones(per_code_len)

    code0 = code_template.copy().reshape(16, -1)
    code1 = code_template.copy().reshape(16, -1)
    code0_str = cfg["code0"].replace("_", "")
    code1_str = cfg["code1"].replace("_", "")
    # now apply the code0
    for i in range(16):
        sign = int(code0_str[i]) * 2 - 1
        # print("code0", sign)
        code0[i, :] *= sign
        sign = int(code1_str[i]) * 2 - 1
        # print("code1", sign)
        code1[i, :] *= sign
    template = np.zeros(2 * total_len)  # two ipps
    if code_type == "0":
        print("code type requested 0")
        template[:per_code_len] = np.ravel(code0)
    elif code_type == "1":
        print("code type requested 1")
        template[:per_code_len] = np.ravel(code1)
    elif code_type == 'both':
        print("code type requested both")
        template[:per_code_len] = np.ravel(code1)
        template[total_len : total_len + per_code_len] = np.ravel(code0)

    template = np.tile(template, cfg["code_repeat_num"])
    return template


def fir_filter(x, hf, filter_state_1d, buf_len = 4096):
    Nfilt = filter_state_1d.shape[0]
    ncols = buf_len - Nfilt
    nrows = x.shape[0] // ncols
    x = x[: nrows * ncols].reshape(nrows, ncols)

    inp = cp.zeros((nrows, buf_len), dtype="complex64") #buf_len is a fast FFT len, since we'll FFT input
    # print("inp shape is", inp.shape)
    inp[0, :Nfilt] = filter_state_1d 
    inp[:, Nfilt:] = x[:, :]
    inp[1:, :Nfilt] = x[:-1, -Nfilt:]
    filter_state_1d[:] = x[-1, -Nfilt:]

    filt_inp = pycufft.ifft(pycufft.fft(inp, axis=1) * hf, axis=1)
    # print("filt_inp shape is", filt_inp.shape)
    out = cp.zeros((nrows, ncols), dtype="complex64")
    out[:, :] = filt_inp[:, Nfilt:]
    out = np.ravel(out)
    return out


def filter_timestream(
    idxs,
    files,
    pfb_size,
    nchunks,
    channels,
    iono_freqs,
    lblock=4096,
    ntap=4,
    cutsize=16,
    filt_thresh=0.45,
    code_type='both'
):
    """Re-PFB baseband spectra for all antennas x polarizations and x-corr all frequencies

    Parameters
    ----------
    idxs : list
        Starting spectrum number for each antenna.
    files : list
        List of files to process that span start and end timestamps
    pfb_size : int
        size of the PFB block to feed into IPFB, multiple of 4096 is good.
    nchunks : int
        Number of pfb_size blocks to read and process.
    channels : np.ndarray or list
        Channel numbers to feed IPFB [0,2048), should be present in baseband file.
    lblock : int, optional
        Length of a one "original" PFB tap, by default 4096
    ntap : int, optional
        Number of PFB taps (for both inverse and forward PFBs), by default 4
    cutsize : int, optional
        Number of spectra to snip after IPFB to avoid, by default 16.
        Number of samples snipped from the reconstructed timestream = cutsize*lblock.
        IPFB algorithm forces circularity, causing the edges of recons. timestream to be bad.
    filt_thresh : float, optional
        IPFB Wiener filter threshold, by default 0.45
    """
    nant = 1
    npol = 2

    read_size = pfb_size - 2 * cutsize
    timestream_size = read_size * lblock
    nchan = len(channels)

    header = bdc.get_header(files[0][0])
    # print(header)
    bit_mode = header["bit_mode"]
    channel_indices = np.where(np.isin(header["channels"], channels))[
        0
    ]  # channels that are in requested channels
    assert channel_indices[0] % 2 == 0
    assert len(channel_indices) % 2 == 0
    print("Channel indices to be used", channel_indices)
    antenna_objs = []
    for i in range(nant):
        aa = bdc.BasebandFileIterator(
            files[i],
            0,  # fileidx is 0 = start idx is inside the first file
            idxs[i],
            read_size,
            nchunks=nchunks,
            channels=channel_indices,
            type="float",
        )
        antenna_objs.append(aa)
    # print("channels present", aa.obj.channels)
    print(
        "---------------------------------------------\n",
        "Channel indices loaded",
        aa.obj.channel_idxs[0],
        "to",
        aa.obj.channel_idxs[-1],
        "corresponding to channels",
        aa.obj.channels[aa.obj.channel_idxs[0]],
        "to",
        aa.obj.channels[aa.obj.channel_idxs[-1]],
        "rough bandwidth of",
        len(aa.obj.channel_idxs) * 0.061,
        "MHz",
        "\n---------------------------------------------"
    )
    final_channels = aa.obj.channels[aa.obj.channel_idxs].copy()
    ipfb = pu.StreamingIPFB_IQ(
        nant,
        npol,
        final_channels,
        nblock=pfb_size,
        lblock=4096,
        ntap=4,
        window="hamming",
        cut=cutsize,
    )
    print("ipfb channels", ipfb.channels)
    ipfb_start_freq = final_channels[0] * 250e6/4096 #center freq of start chan
    rowidx = 0
    print(ipfb)

    start_specnums = [ant.spec_num_start for ant in antenna_objs]
    ant_specnums = [ant.spec_num_start for ant in antenna_objs]
    ant_ptr = np.zeros(nant, dtype=np.int32)

    ncols = ipfb.lblock
    bw = 200e3
    fs = 250e6 / (4096 / ipfb.lblock)
    print("new samp rate", fs / 1e6, "MHz")


    dsamp = int(fs / bw)
    print(
        "Bandwidth of filter is",
        bw / 1e3,
        "kHz. And downsampling factor is",
        dsamp,
        "exact is",
        fs / bw,
    )
    filter_len = 512
    cutoff = bw / fs
    print("cutoff is", cutoff)
    buf_len = 4096
    h = firwin(filter_len, cutoff=cutoff/2, window=('kaiser',4),fs=1) #cutoff is one sided bandwidth
    h_cupy = cp.zeros((1, buf_len), dtype="complex64")
    h_cupy[0, :filter_len] = cp.asarray(h, dtype="complex64")
    hf = pycufft.fft(h_cupy, axis=1)
    print("Filter length and filter buffer length is", filter_len, hf.shape)
    Nts = (pfb_size - 2 * cutsize) * ipfb.lblock
    ncols = buf_len - filter_len
    nrows = Nts // ncols
    Nts_dc = (nrows * ncols + dsamp - 1) // dsamp  # downsampled length after chopping end bits, essentially ceil
    print("Nts is", Nts, "and downsampled length (after chopping end bits) is", Nts_dc)
    print("rate and sample dt after downsampling is", fs/dsamp, dsamp/fs)

    # we'll have to store the end phase for all frequencies to downconvert continuously
    phase_cycles = cp.zeros(len(iono_freqs), dtype="float64")
    # we'll have to store the last filter state for all frequencies and polarizations to filter continuously
    filter_state = cp.zeros((len(iono_freqs), npol, filter_len), dtype="complex64")

    # code_template0 = get_code_template(code_type="0")
    # code_template1 = get_code_template(code_type="1")
    code_template = get_code_template(code_type=code_type)
    code_templates_gpu = cp.zeros((1, Nts_dc), dtype="complex64")
    code_templates_gpu[0, : len(code_template)] = cp.asarray(code_template, dtype="complex64")
    # code_templates_gpu[1, : len(code_template1)] = cp.asarray(code_template1, dtype="complex64")
    code_spectra = pycufft.fft(code_templates_gpu, axis=1)
    print("code spectra shape", code_spectra.shape)

    filtered_timestreams = cp.zeros((len(iono_freqs), npol, Nts_dc), dtype="complex64")
    
    # Pre-allocate temporary timestreams for downconversion to avoid in-place modification and repeated allocations
    ts_pol0_dc = cp.empty(Nts, dtype="complex64")
    ts_pol1_dc = cp.empty(Nts, dtype="complex64")

    print("Storage for filtered d/c d/s timestreams is", filtered_timestreams.shape, filtered_timestreams.nbytes/1e6, "MB")
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    for chunk_idx, chunks in enumerate(zip(*antenna_objs)):
        # start_event.record()
        ts1 = time.time()
        n = 0
        for ant_idx in range(nant):
            chunk = chunks[ant_idx]
            expected_start_specnum = start_specnums[ant_idx] + (chunk_idx) * read_size
            # print(f"Ant {ant_idx} specnum @ {antenna_objs[ant_idx].spec_num_start}; should be @ {start_specnums[ant_idx] + (chunk_idx+1) * read_size}") #spec_num start has already been incremented since a block was read
            assert (
                antenna_objs[ant_idx].spec_num_start
                == start_specnums[ant_idx] + (chunk_idx + 1) * read_size
            )
            assert (
                chunk["specnums"][0]
                == start_specnums[ant_idx] + (chunk_idx) * read_size
            )
            pol0 = bdc.make_continuous_gpu(
                chunk["pol0"],
                chunk["specnums"] - expected_start_specnum,
                cp.arange(0, nchan),
                read_size,
                nchan,
            )
            pol1 = bdc.make_continuous_gpu(
                chunk["pol1"],
                chunk["specnums"] - expected_start_specnum,
                cp.arange(0, nchan),
                read_size,
                nchan,
            )

            ts_pol0 = ipfb.ipfb(ant_idx, 0, pol0, thresh=filt_thresh)
            ts_pol1 = ipfb.ipfb(ant_idx, 1, pol1, thresh=filt_thresh)
            print("ts shape", ts_pol0.shape, "vs Nts", Nts, "dtype", ts_pol0.dtype)

            # end_event.record()
            # end_event.synchronize()
            # print("tot ipfb time", cp.cuda.get_elapsed_time(start_event, end_event)/1000)

            # begin loop over frequencies
            start_event.record()
            for fi, freq in enumerate(iono_freqs):
                ddc_freq = (freq - ipfb_start_freq)/fs #normalized
                
                # FIX: Use separate buffers for downconversion to avoid modifying the input for the next frequency loop
                ddc_kernel(ts_pol0, ddc_freq, phase_cycles[fi], ts_pol0_dc)
                ddc_kernel(ts_pol1, ddc_freq, phase_cycles[fi], ts_pol1_dc)

                # update the phase for the next chunk. Remember two_pi_t starts from 0.
                # print("init phase cycles", phase_cycles[fi])
                # print("cycles per sample", ddc_freq)
                phase_cycles[fi] += ddc_freq * Nts
                phase_cycles[fi] -= cp.floor(phase_cycles[fi]) # keep it between 0 and 1
                # print("new phase cycles", phase_cycles[fi])

                # filter the downconverted ts, sampling rate 5 us
                # FIX: Use separate filter states for pol0 and pol1
                ts_pol0_filt = fir_filter(ts_pol0_dc, hf, filter_state[fi, 0], buf_len = 4096)[::dsamp]  
                ts_pol1_filt = fir_filter(ts_pol1_dc, hf, filter_state[fi, 1], buf_len = 4096)[::dsamp]
                # print("ts_pol0_filt shape is", ts_pol0_filt.shape, "and ts_pol1_filt shape is", ts_pol1_filt.shape)
                filtered_timestreams[fi, 0, :] = ts_pol0_filt
                filtered_timestreams[fi, 1, :] = ts_pol1_filt
            #perform correlation
            corr = pycufft.ifft( pycufft.fft(filtered_timestreams, axis=2) * cp.conj(code_spectra[None,:, :]), axis=2)
            end_event.record()
            end_event.synchronize()
            print("tot filt and corr time", cp.cuda.get_elapsed_time(start_event, end_event)/1000)
            print("corr shape is", corr.shape)
    return corr


if __name__ == "__main__":
    # I'm not using downconverted IPFB right now, it's under testing.
    # sampling rate is the original 250 MSPS, bw is 200 kHz
    code_type='1'
    #fpath = "/scratch/mohanagr/summer_2025/baseband/mars1"
    #tstart = 1753215895
    fpath = "/scratch/mohanagr/drive3_mars_spring2025/baseband/"
    # tstart = 1746782095 #original
    # tstart = 1746818097
    tstart = 1746784195 # 9 50 Z
    tend = tstart + 15  # seconds of data. Total sweep is like 10 s + some buffer
    tstart_str = datetime.datetime.utcfromtimestamp(tstart).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )
    tend_str = datetime.datetime.utcfromtimestamp(tend).strftime("%Y-%m-%dT%H:%M:%S")
    print("Processing data from", tstart_str, "to", tend_str, "UTC")
    files, idx = bu.get_init_info(tstart, tend, fpath)
    header = bdc.get_header(files[0])
    print("Bit mode of files", header["bit_mode"], "num chans", len(header["channels"]))
    print("Freqs present", header["channels"]*0.061)
    iono_freqs = [3.96632e+06,4.07491e+06,4.187e+06,4.30109e+06,4.41885e+06,4.53983e+06,4.66412e+06,4.79182e+06,4.933e+06,5.0578e+06,5.201e+06,5.33854e+06,5.4847e+06,5.63486e+06,5.78914e+06,5.94763e+06,6.11047e+06,6.278e+06,6.44964e+06,6.62622e+06,6.813e+06,6.99402e+06,7.18551e+06,7.38223e+06,7.58435e+06,7.792e+06,8.00533e+06,8.2245e+06,8.44968e+06,8.68101e+06,8.91869e+06,9.16287e+06,9.41373e+06,9.67146e+06,9.93626e+06,1.02083e+07]
    #iono_freqs = [3.75774e+06,3.86062e+06,3.96632e+06,4.07491e+06,4.187e+06,4.30109e+06,4.41885e+06,4.53983e+06,4.66412e+06,4.79182e+06,4.933e+06,5.0578e+06,5.201e+06,5.33854e+06,5.4847e+06,5.63486e+06,5.78914e+06,5.94763e+06,6.11047e+06,6.278e+06,6.44964e+06,6.62622e+06,6.813e+06,6.99402e+06,7.18551e+06,7.38223e+06,7.58435e+06,7.792e+06,8.00533e+06,8.2245e+06,8.44968e+06,8.68101e+06,8.91869e+06,9.16287e+06,9.41373e+06,9.67146e+06,9.93626e+06,1.02083e+07]
    print("No. of ionosonde freqs to try", len(iono_freqs))
    # sys.exit()

    pfb_size = 1000000
    nchunks = 1

    corr = filter_timestream(
        [idx,],
        [files,],
        pfb_size,
        nchunks,
        np.arange(64,100),
        iono_freqs,
        lblock=4096,
        ntap=4,
        cutsize=16,
        filt_thresh=0.2,
        code_type=code_type
    )
    outdir = f"/scratch/{os.environ.get('USER')}/ionosphere/output"
    fname = "iono_corr_2pols_"+tstart_str+"Z_to_"+tend_str+"Z_"
    os.makedirs(outdir, exist_ok = True)
    print(f"Saving to {path.join(outdir, fname)}")
    np.savez(path.join(outdir, fname), corr = corr, freqs = iono_freqs)
