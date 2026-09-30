# def make_hists_iono_offset(specnum_start, specnum_end, args = default_args):
#     final_channels, antenna_objs = data.get_antenna_objs(args=args)
#     chunks = zip(*antenna_objs).__next__()
#     # Setup IPFB
#     ipfb = sp.setup_ipfb(final_channels, args.ipfb_chunk_size)

#     for ant_idx in range(args.num_ant):
#         chunk = chunks[ant_idx]
#         expected_start_specnum = antenna_objs[ant_idx].spec_num_start
#         # Would have to have + (chunk_idx) * read_size if we were reading in multiple chunks
        
#         pol0 = bdc.make_continuous_gpu(
#             chunk["pol0"],
#             chunk["specnums"] - expected_start_specnum, # Indicies of present spectra
#             xp.arange(0, len(final_channels)),
#             args.read_size,
#             len(final_channels),
#         )

#         pol1 = bdc.make_continuous_gpu(
#             chunk["pol1"],
#             chunk["specnums"] - expected_start_specnum,
#             xp.arange(0, len(final_channels)),
#             args.read_size,
#             len(final_channels),
#         )

#         n_chan = len(final_channels)
#         n_cols = min(6, n_chan)
#         n_rows = (n_chan - 1) // n_cols + 1

#         figsize = (n_cols * 2.5, n_rows * 2)

#         fig, axs = plt.subplots(n_rows, n_cols, sharex = True, sharey = True,
#                                 layout = "constrained", figsize = figsize)
#         flat_axs = np.atleast_1d(axs).flatten()

#         for chan_idx in range(n_chan):
#             freq = final_channels[chan_idx] * args.chan_res_init
#             # Find nearest ionosonde frequency
#             iono_idx = np.argmin(np.abs(args.ionosonde_freqs - freq))
#             iono_freq = args.ionosonde_freqs[iono_idx]

#             start_idx = specnum_start + offset(iono_idx, args.chan_res_init, args = args) - expected_start_specnum
#             end_idx = specnum_end + offset(iono_idx, args.chan_res_init, args = args) - expected_start_specnum
#             # print(chan_idx, start_idx, end_idx)
#             # print(pol0[start_idx:end_idx, chan_idx])

#             pol1_chan = pol1[start_idx:end_idx, chan_idx]
#             # Count each complex value
#             counts = np.zeros((15, 15), dtype=int)
#             for z in pol1_chan:
#                 counts[int(z.imag) + 7, int(z.real) + 7] += 1

#             # Plot
#             flat_axs[chan_idx].imshow(counts, origin="lower", extent=[-7.5, 7.5, -7.5, 7.5])
#             flat_axs[chan_idx].set_title(f"{freq/1e6:.2f} MHz\n(closest freq {iono_freq/1e6:.2f} MHz)")

#         fig.supxlabel("Real")
#         fig.supylabel("Imaginary")
#         # fig.colorbar(label="Count")
        
#         fig.savefig(os.path.join(args.out_dir, "baseband_hist.png"))