from subprocess import call

import SNAPfiletools as sft

global_time_start = "20190720_000000"
global_time_stop = "20190720_235959"
time_step = 1 #time step in hours


global_ctime_start = sft.timestamp2ctime(global_time_start)
global_ctime_stop = sft.timestamp2ctime(global_time_stop)

ctime_step = time_step *3600

for ctime in range(global_ctime_start, global_ctime_stop, ctime_step):
    somename = sft.time2fnames(ctime, ctime + ctime_step, '/project/s/sievers/mars2019/MARS1/albatros_north_baseband')
    if len(somename) == 0:
        print("Skipped " + str(ctime))
        continue
    call(["python", "coarse_cross.py", "-c", str(ctime), str(ctime + ctime_step)])


