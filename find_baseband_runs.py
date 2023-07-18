import sys, os, re, datetime
from optparse import OptionParser
import numpy as np

#==========================================================
def chan2freq(chan):
    return np.round(125.*chan/2048., 2)

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
    log_ctimes = np.asarray(log_ctimes)
    log_files = np.asarray(log_files)
    return log_files, log_ctimes

#==========================================================
if __name__ == '__main__':

    """
    Script that trolls log files to find baseband runs within a
    specified time window and with a minimum run length.  Reports
    system state and general health for each baseband run.
    """
    
    parser = OptionParser()
    parser.set_usage('python find_baseband_runs.py [options]')
    parser.set_description(__doc__)
    parser.add_option('-L', '--logdir', dest='logdir',type='str', default='/project/s/sievers/simont/logs',
		      help='Log directory [default: %default]')
    parser.add_option('-t', '--time', dest='min_time', type='int', default=1,
                      help='Minimum length (in minutes) for a run to be reported by this script [default: %default]')
    parser.add_option('-s', '--start', dest='start', type='str', default=None,
                      help='Start date in format YYYYMMDD [default: %default]')
    parser.add_option('-S', '--stop', dest='stop', type='str', default=None,
                      help='Stop date in format YYYYMMDD [default: %default]')
    parser.add_option('-b', '--bits', dest='bits', type='int', default=None,
                      help='Select runs with only specified number of bits [default: %default]')
    parser.add_option('-e', '--events', dest='events_file', type='str',
                      default='/project/s/sievers/simont/logs/events.txt',
                      help='Events file [default: %default]')
    opts, args = parser.parse_args(sys.argv[1:])

    # Get system state
    if opts.events_file != 'None':
        s = re.compile(r'(\d{8}\_\d{6})')
        fp = open(opts.events_file, 'r')
        events_ctimes = []
        events = []
        for line in fp.readlines():
            if not s.search(line):
                continue
            tstamp = s.search(line).groups()[0]
            events_ctimes.append( int((datetime.datetime.strptime(tstamp, '%Y%m%d_%H%M%S') - \
                                       datetime.datetime(1970, 1, 1)).total_seconds()) )
            events.append(line.split('\t')[-1].strip())
        events_ctimes, events = zip(*sorted(zip(events_ctimes, events)))
        events_ctimes = np.asarray(events_ctimes)
        events = np.asarray(events)

    # Get time stamps of config_fpga and baseband log files
    config_logfiles, config_logfiles_ctimes = logs2ctimes(opts.logdir+'/albatros_config_fpga')
    baseband_logfiles, baseband_logfiles_ctimes = logs2ctimes(opts.logdir+'/albatros_dump_baseband')

    # Pick out files within specified start and stop dates, if present
    ctstart = baseband_logfiles_ctimes[0]
    ctstop = baseband_logfiles_ctimes[-1]
    if opts.start is not None:
        ctstart = int((datetime.datetime.strptime(opts.start, '%Y%m%d') - \
                       datetime.datetime(1970, 1, 1)).total_seconds())
    if opts.stop is not None:
        # Add 24 hours to ctstop so that end date is inclusive
        ctstop = int((datetime.datetime.strptime(opts.stop, '%Y%m%d') - \
                      datetime.datetime(1970, 1, 1)).total_seconds()) + 86399
    inds = np.where( (baseband_logfiles_ctimes >= ctstart) & (baseband_logfiles_ctimes <= ctstop) )[0]
    if len(inds) == 0:
        print('No files found within specified start and stop dates')
        exit(0)
    baseband_logfiles = baseband_logfiles[inds]
    baseband_logfiles_ctimes = baseband_logfiles_ctimes[inds]
        
    # Loop over baseband files and pick out run lengths
    s = re.compile(r'^(\d{2}-\d{2}-\d{4}\ \d{2}\:\d{2}\:\d{2})')
    baseband_runtimes = []
    for baseband_logfile in baseband_logfiles:
        fp = open(baseband_logfile, 'r')
        lines = fp.readlines()
        fp.close()
        if len(lines) < 2:
            baseband_runtimes.append(0)
            continue
        if not s.search(lines[0]):
            baseband_runtimes.append(0)
            continue
        tstart = s.search(lines[0]).groups()[0]
        ctstart = int((datetime.datetime.strptime(tstart, '%d-%m-%Y %H:%M:%S') - \
                       datetime.datetime(1970, 1, 1)).total_seconds())
        if not s.search(lines[-1]):
            baseband_runtimes.append(0)
            continue
        tstop = s.search(lines[-1]).groups()[0]
        ctstop = int((datetime.datetime.strptime(tstop, '%d-%m-%Y %H:%M:%S') - \
                      datetime.datetime(1970, 1, 1)).total_seconds())
        baseband_runtimes.append(ctstop-ctstart)
    baseband_runtimes = np.asarray(baseband_runtimes)

    min_sec = opts.min_time*60
    inds = np.where(baseband_runtimes >= min_sec)[0]
    baseband_runtimes = baseband_runtimes[inds]
    baseband_logfiles = baseband_logfiles[inds]
    baseband_logfiles_ctimes = baseband_logfiles_ctimes[inds]

    # Print all the things in reverse chronological order
    for ind in range(len(baseband_logfiles))[-1::-1]:
        # Find the closest system state
        sys_state = 'unknown'
        if opts.events_file != 'None':
            ii = np.where(events_ctimes - baseband_logfiles_ctimes[ind] < 0)[0]
            if len(ii) != 0:
                sys_state = events[ii][-1]
        
        # Find the closest config_fpga log file
        ii = np.where(config_logfiles_ctimes - baseband_logfiles_ctimes[ind] < 0)[0]
        if len(ii) == 0:
            config_logfile = None
        else:
            config_logfile = config_logfiles[ii[-1]]

        # Find number of bits used, see if the human is interested or not
        if opts.bits is not None:
            s = re.compile(r'(\d)$')
            btxt = os.popen('grep "Baseband bits" '+baseband_logfiles[ind]+' | cut -d " " -f6-').readlines()[0]
            if s.search(btxt):
                b = int(s.search(btxt).groups()[0])
                # ...this is super hacky, and could be made way more efficient
                # if we do a top level grep early on.  But I'm tired and lazy.
                if b != opts.bits:
                    continue

        print('=============================================')
        print(baseband_logfiles[ind])
        print('System state:', sys_state)
        print('Start time:', str(datetime.datetime.utcfromtimestamp(baseband_logfiles_ctimes[ind])))
        print('Start ctime:', baseband_logfiles_ctimes[ind])
        print('Run length:', baseband_runtimes[ind]/60, 'minutes')
        btxt = os.popen('grep "Baseband bits" '+baseband_logfiles[ind]+' | cut -d " " -f6-').readlines()[0]
        print(btxt.strip())
        ctxt = os.popen('grep "Channels" '+baseband_logfiles[ind]+' | cut -d " " -f6-').readlines()[0]
        print(ctxt.strip())
        if config_logfile is None:
            print('Channel coeffs: no information found')
        else:
            cctxt = os.popen('grep "Channel coeffs" '+config_logfile+' | cut -d " " -f6-').readlines()[0]
            print(cctxt.strip())
        s = re.compile(r'(\d+)\:(\d+)$')
        if s.search(ctxt):
            c1 = int(s.search(ctxt).groups()[0])
            c2 = int(s.search(ctxt).groups()[1])
            print('Frequencies:', chan2freq(c1), '-', chan2freq(c2), 'MHz')
            print('Freq bandwidth:', chan2freq(c2)-chan2freq(c1), 'MHz')
        if config_logfile is None:
            print('ADC bits used: no information found')
        else:
            atxt = os.popen('grep "ADC bits" '+config_logfile+' | cut -d " " -f4-').readlines()
            if len(atxt) > 0:
                print(atxt[0].strip())
            else:
                print('ADC bits used: no information found')
        if config_logfile is not None:
            ftxt = os.popen('grep "ADC initialisation failed" '+config_logfile+' | wc -l').readlines()[0]
            print('Total ADC initialization failures:', ftxt.strip())
