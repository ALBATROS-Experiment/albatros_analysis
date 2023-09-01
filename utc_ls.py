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


def files_to_human_time(parent_dir, timezone=None):
    '''
    Author: Joelle, May 2022

    Given path to a directory that contains timestamp directories
    (e.g. 16528/ which contains 1652700704/ and etc), gives the 
    human time of each timestamp directory it contains.
    '''

    files = np.sort(np.array(os.listdir(parent_dir),dtype=int))
    utc_dates = [datetime.datetime.utcfromtimestamp(i).strftime("%m-%d-%Y  %H:%M:%S")for i in files]
    local_dates = [datetime.datetime.fromtimestamp(i, timezone).strftime("%m-%d-%Y  %H:%M:%S")for i in files]

    d = np.vstack((files,utc_dates,local_dates)).T
    if timezone:
        return tabulate.tabulate(d,headers=["Timestamp","UTC date","Local date ({})".format(str(timezone))],stralign="center")
    else:
        return tabulate.tabulate(d,headers=["Timestamp","UTC date","My system date"],stralign="center")


if __name__ == '__main__':
    # Just a stupid convenience script for listing ctime-stamped
    # directories and printing out equivalent UTC in human-friendly
    # format.  
    # Example usage: python utc_ls.py ~/auto_cross_data/16528/ -tz MARS

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, nargs="+", help="Path to directory that contains Unix timestamp subdirectories. Ex: 16528/ which contains 1652800704/")
    parser.add_argument("-tz", "--timezone", type=str, default=None, help="Local timezone of the telescope (must be recognized by pytz). Ex: US/Eastern. Default is None.")
    args = parser.parse_args()

    timezone = get_timezone(args.timezone)

    for path in args.path:
        print(files_to_human_time(path, timezone))
