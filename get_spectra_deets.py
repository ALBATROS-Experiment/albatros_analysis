import datetime, tabulate, os, pytz
import numpy as np
import argparse
import SNAPfiletools as sft
import datetime
import itertools
import re

def get_timezone(timezone):
    """
    Transform given timezone name (e.g. "US/Eastern") into a pytz timezone object.
    Recognizes names of common observation sites (Marion, MARS, Uapishka).
    Return None if given timezone name is not recognized.
    """
    if timezone:
        if timezone in pytz.all_timezones:
            return pytz.timezone(timezone)
        if timezone.lower() == "marion":
            return pytz.timezone("Africa/Nairobi")
        if timezone.lower() == "mars":
            return pytz.timezone("America/Winnipeg")
        if timezone.lower() == "uapishka":
            return pytz.timezone("America/Toronto")
        if timezone.lower() == "utc":
            return pytz.timezone("UTC")

    return None


if __name__ == '__main__':
    # Get some insightful stats from direct spectra when returning after an extended period.
    # Stats: Number of unique days with observations
    # Example usage: python get_spectra_deets.py /path/to/data_auto_cross/ 20181401_000000 20190401_000000

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_dirs", type=str, nargs="+", help="Direct data directory")
    parser.add_argument("time_start", type=str, help="Start time YYYYMMDD_HHMMSS or ctime. Both in UTC.")
    parser.add_argument("time_stop", type=str, help="Stop time YYYYMMDD_HHMMSS or ctime. Both in UTC.")
    parser.add_argument("-tz", "--timezone", type=str, default=None, help="Local timezone of the telescope (must be recognized by pytz). Ex: US/Eastern.")
    args = parser.parse_args()

    timezone = get_timezone(args.timezone) # currently not used

    # figuring out if human time or ctime was passed with pattern matching
    rx_human = re.compile(r'^\d{8}_\d{6}$')
    rx_ctime = re.compile(r'^\d{10}$')
    m1 = rx_human.search(args.time_start)
    m2 = rx_ctime.search(args.time_start)
    if(m1):
        ctime_start = sft.timestamp2ctime(args.time_start)
        ctime_stop = sft.timestamp2ctime(args.time_stop)
    elif(m2):
        ctime_start = int(args.time_start)
        ctime_stop = int(args.time_stop)
    else:
        raise ValueError("INVALID time format entered.")

    data_subdirs = []
    for data_dir in args.data_dirs:
        data_subdirs.extend(sft.time2fnames(ctime_start, ctime_stop, data_dir))

    dates = [datetime.datetime.utcfromtimestamp(int(os.path.basename(data_subdir))).date() for data_subdir in data_subdirs]
    dates = np.sort(dates)

    # Get dates of first and last data
    print("Start date: {}".format(dates[0].strftime("%Y-%m-%d")))
    print("End date: {}".format(dates[-1].strftime("%Y-%m-%d")))

    # Get total time span of data in days
    delta = dates[-1] - dates[0]
    print("Time span (days): {}".format(delta.days))
    
    # Get number of days with data
    unique_dates = np.unique(dates)
    group_by_year = [list(g) for k, g in itertools.groupby(unique_dates, key=lambda d: d.year)]

    print("MM-YYYY: Days with data")
    for y in group_by_year:
        group_by_month = [list(g) for k, g in itertools.groupby(y, key=lambda d: d.month)]
        for m in group_by_month:
            print(m[0].strftime("%m-%Y")+": "+str(len(m))) 
    print("Total: "+str(len(unique_dates)))            

#    d = np.vstack((timestamp_runs, utc_runs, local_runs)).T
#    if timezone:
#        print(tabulate.tabulate(d,headers=["Timestamps","UTC dates","Local dates ({})".format(str(timezone))],stralign="center"))
#    else:
#        print(tabulate.tabulate(d,headers=["Timestamps","UTC dates","My system dates"],stralign="center"))
