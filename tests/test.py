"""
tests for the pyzbc2014 module.
Run various simulations from the Zilany, Bruce, and Carney (2014) model
to verify that the model is working as expected.
The following tests are included:
1) simulate IHC output to a tone stimulus
2) simulate AN output to the IHC output
3) simulate spike times from the AN output
4) plot PSTH and ISI histogram from the spike times
5) simulate population neurogram across CFs
6) simulate rate-level functions for different fiber types

All tests are independent, and can be enabled/disabled by setting the corresponding
boolean variable to True/False.

"""

import numpy as np
from pyzbc2014 import pyzbc2014 as PyZBC2014Model
import matplotlib.pyplot as plt

species = "cat"
cf = 10e3
dbspl = 30.0
fibertype = "hsr"
noisetype = "none"  # 'fresh', or 'none' for faster runs
Fs = 100e3  # Sampling frequency
stimwin = [0.25, 0.25 + 0.5]
t_max = 1.0  # seconds

zbc = PyZBC2014Model()  # create an instance of the model, and load the shared libraries.


def compute_stimulus(
    cf_Hz: float,
    dB_SPL: float = 10.0,
    sample_frequency: float = 100e3,
    t_max: float = 1.0,
    stimulus_window: list = [0.1, 0.3],
) -> (np.ndarray, np.ndarray):
    """compute_stimulus: Compute a stimulus waveform to test the model.

    Parameters
    ----------
    cf_Hz: float
        Frequency of the tone stimulus (Hz)
    dB_SPL: float
        Sound pressure level, dB SPL, for the tone stimulus.
    sample_frequency: float
        Sampling frequency (Hz) for the stimulus waveform
    t_max : float
        maximum time for the stimulus waveform (seconds)
    stimulus_window : list
        start and end times of the stimulus window (seconds)
    Returns
    -------
    np.ndarray, np.ndarray
        time array, stimulus waveform (Pa)
    """
    t = np.arange(0.0, t_max, 1.0 / sample_frequency)
    px = np.zeros_like(t)
    x = np.sin(2 * np.pi * cf_Hz * t)
    for i, xt in enumerate(t):
        if xt >= stimulus_window[0] and xt <= stimulus_window[1]:
            px[i] = 20e-6 * 10 ** (dB_SPL / 20.0) * np.sqrt(2) * x[i]
    return t, px


t, px = compute_stimulus(
    cf_Hz=cf, dB_SPL=dbspl, sample_frequency=Fs, t_max=t_max, stimulus_window=stimwin
)
# plt.plot(t, px)
# plt.show()
# exit()

nreps = 50

# "tests" that will be run
ihc = False  # True
popresp = True
neurogram = False
psth = True
rate_io = False

if ihc:
    nreps = 1
    # Simulate IHC waveform and AN waveform, plot inst. rate result
    t, px = compute_stimulus(
        cf_Hz=cf, dB_SPL=dbspl, sample_frequency=Fs, t_max=t_max, stimulus_window=stimwin
    )
    ihcout = zbc.sim_ihc_zbc2014(px, cf=cf, nrep=nreps, species=species)
    # anout = zbc.sim_anrate_zbc2014(
    #     ihcout, cf=cf, nrep=nreps, fibertype=fibertype, noisetype=noisetype
    # )
    ihc_out = ihcout.reshape(-1, nreps)
    plt.plot(t, ihc_out)
    plt.title(
        f"IHC output for {fibertype:s} at {cf/1e3:.1f} kHz tone at {dbspl:.1f} dB SPL (species: {species})"
    )
    plt.xlabel("Time (seconds)")
    plt.ylabel("IHC output (a.u.)")
    plt.show()

if popresp:
    # Simulate population average-rate response across frequency to pure tone
    nreps = 50
    cfs = np.exp(np.linspace(np.log(0.5e3), np.log(32e3), 41))
    t, px = compute_stimulus(
        cf_Hz=cf, dB_SPL=dbspl, sample_frequency=Fs, t_max=t_max, stimulus_window=stimwin
    )
    rates = np.zeros(cfs.size)
    for i in range(cfs.size):
        ihcout = zbc.sim_ihc_zbc2014(px, cf=cfs[i], nrep=nreps, species=species)
        anout = zbc.sim_anrate_zbc2014(
            ihcout, cf=cfs[i], nrep=nreps, fibertype=fibertype, noisetype=noisetype
        )
        # print('anout shape: ', anout.shape)
        anout = anout.reshape(-1, nreps)
        rates[i] = np.mean(anout)

    plt.plot(cfs, rates)
    plt.xscale("log")
    plt.title(
        f"Average output vs CF, fiber type: {fibertype:s}, {cf/1e3:.1f} kHz tone at {dbspl:.1f} dB SPL (species: {species})"
    )
    plt.xlabel("CF (Hz)")
    plt.ylabel("Average hair cell output")
    plt.show()

if psth:
    nreps = 50
    total_stim = t_max * nreps  # in seconds
    # Plot PSTH for 1 kHz tone at 50 dB SPL
    spikes = []
    isi_times = []
    t, px = compute_stimulus(
        cf_Hz=cf, dB_SPL=dbspl, sample_frequency=Fs, t_max=t_max, stimulus_window=stimwin
    )
    for nr in range(nreps):
        ihcout = zbc.sim_ihc_zbc2014(px, cf=cf, nrep=1, species=species)
        anout = zbc.sim_anrate_zbc2014(
            ihcout, cf=cf, fibertype=fibertype, nrep=1, noisetype=noisetype
        )
        spiketimes = zbc.sim_spike_generator_zbc2014(anout, fs=Fs, totalstim=total_stim, nrep=1)
        spiketimes = spiketimes[spiketimes > 0.0]
        spikes.append(spiketimes)
        # select only the spikes during the stimulus
        stim_spikes = spiketimes[(spiketimes > stimwin[0]) & (spiketimes <= stimwin[1])]
        isi_times.append(np.diff(stim_spikes))
    spks = np.concatenate(spikes)
    isi_times = np.concatenate(isi_times)
    binwidth = 0.001  # 0.5 ms
    numbins = int(np.ceil(t.max() / binwidth))
    rounded_max = np.ceil(t.max())
    xbins = np.arange(0, rounded_max + binwidth, binwidth)
    psthb, bin_edges = np.histogram(spks, bins=numbins)  # xbins) # , range=(0, len(psth)/Fs))
    f, ax = plt.subplots(1, 2, figsize=(8, 5))
    ax[0].stairs(edges=bin_edges, values=psthb / (binwidth * nreps), linewidth=0.5, fill=True)
    ax[0].set_title(f"PSTH for {cf/1e3:.1f} kHz tone at {dbspl:.1f} dB SPL")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Firing rate (sp/s)")
    # plot isi histogram
    # isibins = np.arange(0, 0.05+0.0001, 0.0001)
    isihist, isibin_edges = np.histogram(isi_times, bins=int(0.05 / 0.0001))
    ax[1].stairs(edges=isibin_edges, values=isihist, linewidth=0.5, fill=True)
    ax[1].set_title("Inter-spike interval histogram")
    ax[1].set_xlabel("Interval (ms)")
    ax[1].set_ylabel("Count")
    plt.show()

# Plot population neurogram (high resolution)
if neurogram:
    nreps = 10
    cf = 3e3
    t, px = compute_stimulus(
        cf_Hz=cf, dB_SPL=dbspl, sample_frequency=Fs, t_max=t_max, stimulus_window=stimwin
    )
    cfs = np.exp(np.linspace(np.log(0.5e3), np.log(16e3), 51))
    rates = []
    for i in range(cfs.size):
        ihcout = zbc.sim_ihc_zbc2014(px, cf=cfs[i], nrep=1, species=species)
        anout = zbc.sim_anrate_zbc2014(
            ihcout, cf=cfs[i], nrep=1, fibertype=fibertype, noisetype="none"
        )
        # print(anout.shape)
        rates.append(anout)

    plt.pcolormesh(t, np.log2(cfs / 1e3), np.array(rates))
    plt.title(f"Neurogram for {cf/1e3:.1f} kHz tone at {dbspl:.1f} dB SPL")
    plt.ylabel(f"CF (oct re: {cf/1e3:.1f} kHz)")
    plt.xlabel("Time (s)")
    plt.colorbar(label="Inst. rate (a.u.)")
    plt.show()

if rate_io:
    dbs = np.arange(-10, 101, 5)
    rates = []
    f, ax = plt.subplots(1, 1)
    t_max = 1.0
    nreps = 50
    total_stim = t_max * nreps  # in seconds
    for fibers in ["lsr", "msr", "hsr"]:
        rates = []
        for db in dbs:
            t, px = compute_stimulus(
                cf_Hz=cf, dB_SPL=db, sample_frequency=Fs, t_max=t_max, stimulus_window=stimwin
            )
            stimrate = []
            for nrep in range(nreps):
                if nrep == 0:  # only need to compute IHC once per rep with a given stim and fiber
                    ihcout = zbc.sim_ihc_zbc2014(px, cf=cf, nrep=1, species=species)
                    anout = zbc.sim_anrate_zbc2014(
                        ihcout, cf=cf, nrep=1, fibertype=fibers, noisetype=noisetype
                    )
                spikes = zbc.sim_spike_generator_zbc2014(anout, fs=Fs, totalstim=total_stim, nrep=1)
                spikes = spikes[spikes > 0.0]
                spikes = spikes[(spikes > stimwin[0]) & (spikes < stimwin[1])]
                stimrate.append(np.sum(spikes) / np.diff(stimwin))
            rates.append(np.mean(stimrate))
        ax.plot(dbs, rates)
    ax.set_title(f"Rate vs. Level at {cf/1e3:.1f} kHz (species: {species})")
    ax.set_xlabel("Sound Level (dB SPL)")
    ax.set_ylabel("Average rate (sp/s)")
    ax.legend(["LSR", "MSR", "HSR"])
    plt.show()
