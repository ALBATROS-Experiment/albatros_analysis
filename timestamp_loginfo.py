import numpy as nm
import re, sys, os, datetime

#==========================================================
def logs2ctimes(logdir):
    fnames = os.listdir(logdir)
    fnames.sort()
    log_files = []
    log_ctimes = []
    # Log files have time stamps of form DDMMYYYY_HHMMSS
    s = re.compile(r'(\d{8}\_\d{6})')
    for fname in fnames:
        if not s.search(fname):
            continue
        tstamp = s.search(fname).groups()[0]
        ctime = int((datetime.datetime.strptime(tstamp, '%d%m%Y_%H%M%S') - datetime.datetime(1970, 1, 1)).total_seconds())
        log_files.append(logdir+'/'+fname)
        log_ctimes.append(ctime)
    # Sort by ctimes
    log_ctimes, log_files = zip(*sorted(zip(log_ctimes, log_files)))
    log_files = nm.asarray(log_files)
    log_ctimes = nm.asarray(log_ctimes)
    return log_files, log_ctimes

#==========================================================
if __name__ == '__main__':

    # Usage: python timestamp_loginfo.py [-L <log file directory>] <ctimes>
    
    args = sys.argv[1:]
    ctstamp_list = []
    logdir = '/home/cynthia/working/arctic/data/logs'
    for arg in args:
        if logdir is None:
            logdir = arg
            continue
        if arg == '-L':
            logdir = None
        else:
            ctstamp_list = ctstamp_list + [int(arg)]
    print('Getting log information from', logdir)

    # Get time stamps of config_fpga, baseband, and dump_spectra log files
    config_logfiles, config_logfiles_ctimes = logs2ctimes(logdir+'/albatros_config_fpga')
    baseband_logfiles, baseband_logfiles_ctimes = logs2ctimes(logdir+'/albatros_dump_baseband')
    spectra_logfiles, spectra_logfiles_ctimes = logs2ctimes(logdir+'/albatros_dump_spectra')
    # Get log info corresponding to requested time stamps
    for ctstamp in ctstamp_list:
        print('##########################################################')
        print('Log information for', ctstamp)
        tstamp = datetime.datetime.utcfromtimestamp(ctstamp)
        print('Time stamp (UTC)', tstamp)
        # Find out which config_fpga logfile is the right one
        ind = nm.where(config_logfiles_ctimes - ctstamp < 0)[0][-1]
        print('Config FPGA log file:', config_logfiles[ind])
        # Find closest baseband log file
        i = nm.where(baseband_logfiles_ctimes - ctstamp < 0)[0][-1]
        dt = nm.round((ctstamp - baseband_logfiles_ctimes[i])/60.0, 2)
        print('Nearest baseband log file (time diff =',dt,'minutes):')
        print(baseband_logfiles[i])
        # Find closest dump_spectra log file
        i = nm.where(spectra_logfiles_ctimes - ctstamp < 0)[0][-1]
        dt = nm.round((ctstamp - spectra_logfiles_ctimes[i])/60.0, 2)
        print('Nearest dump_spectra log file (time diff =',dt,'minutes):')
        print(spectra_logfiles[i])
        # Pull out some of the most useful information from the config file
        txt = os.popen('grep "Baseband bits" '+config_logfiles[ind]+' | cut -d " " -f6-').readlines()[0]
        print(txt.strip())
        txt = os.popen('grep "Channels" '+config_logfiles[ind]+' | cut -d " " -f6-').readlines()[0]
        print(txt.strip())
        txt = os.popen('grep "Channel coeffs" '+config_logfiles[ind]+' | cut -d " " -f6-').readlines()[0]
        print(txt.strip())
        txt = os.popen('grep "ADC bits" '+config_logfiles[ind]+' | cut -d " " -f4-').readlines()
        if len(txt) != 0:
            print(txt[0].strip())
        else:
            print('ADC bits used: no information found')
