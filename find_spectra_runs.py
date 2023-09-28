import datetime, tabulate, os, pytz
import numpy as np
import argparse

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

    return None


if __name__ == '__main__':
    # Just a stupid convenience script for listing ctime-stamped
    # directories and printing out equivalent UTC in human-friendly
    # format.  
    # Example usage: python utc_ls.py ~/auto_cross_data/16528/ -tz MARS
    # For multiple data directories: python utc_ls.py ~/auto_cross_data/1* -tz MARS

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("path", type=str, nargs="+", help="Path to directory that contains Unix timestamp subdirectories. Ex: 16528/ which contains 1652800704/")
    parser.add_argument("-tz", "--timezone", type=str, default=None, help="Local timezone of the telescope (must be recognized by pytz). Ex: US/Eastern.")
    parser.add_argument("-dt", "--file_time", type=float, default=3600, help="Total time (in seconds) of a spectrum file.")
    parser.add_argument("-tol", "--tolerance", type=float, default=120, help="Extra time (in seconds) to allow between the creation of sequential spectra files while searching for continuous runs.")
    parser.add_argument("-min", "--minimum_time", type=float, default=120, help="Minimum time (in seconds) spent on a spectrum file to be considered the start of a run.")
    args = parser.parse_args()

    timezone = get_timezone(args.timezone)

    files = np.array([], dtype=int)
    for path in args.path:
        files = np.append(files, np.sort(np.array(os.listdir(path), dtype=int)))

    time_diff = np.diff(files)
    time_diff_allowed = args.file_time + args.tolerance
    starts = []
    stops = []
    for i in range(len(time_diff)):
        if len(starts) == len(stops):
            if (time_diff[i] <= time_diff_allowed) and (time_diff[i] >= args.minimum_time):
                starts.append(files[i])
        else:
            if time_diff[i] > time_diff_allowed:
                stops.append(files[i])
            elif i == len(time_diff) - 1:
                stops.append(files[i+1])

    timestamp_runs = ["{:d} -- {:d}".format(starts[i], stops[i]) for i in range(len(starts))]

    utc_starts = [datetime.datetime.utcfromtimestamp(i).strftime("%Y-%m-%d %H:%M:%S") for i in starts]
    utc_stops = [datetime.datetime.utcfromtimestamp(i).strftime("%Y-%m-%d %H:%M:%S") for i in stops]
    utc_runs = [utc_starts[i]+" -- "+utc_stops[i] for i in range(len(utc_starts))]

    local_starts = [datetime.datetime.fromtimestamp(i, timezone).strftime("%Y-%m-%d %H:%M:%S") for i in starts]
    local_stops = [datetime.datetime.fromtimestamp(i, timezone).strftime("%Y-%m-%d %H:%M:%S") for i in stops]
    local_runs = [local_starts[i]+" -- "+local_stops[i] for i in range(len(local_starts))]

    d = np.vstack((timestamp_runs, utc_runs, local_runs)).T
    if timezone:
        print(tabulate.tabulate(d,headers=["Timestamps","UTC dates","Local dates ({})".format(str(timezone))],stralign="center"))
    else:
        print(tabulate.tabulate(d,headers=["Timestamps","UTC dates","My system dates"],stralign="center"))
