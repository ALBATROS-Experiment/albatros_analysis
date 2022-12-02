import numpy as np
import matplotlib.pyplot as plt
import os
import gc

# TODO: think about moving this to a file maybe?
# Config
maxTemp = 80    # Celsius
minTemp = -20
maxTimeDiff = 10 # seconds
diagnosticDatabase = {"fpga_temp":np.zeros(1), "pi_temp":np.zeros(1), "timeStart":np.zeros(1), "timeStop":np.zeros(1), "fft_overflow1":np.zeros(1),\
            "fft_overflow2":np.zeros(1), "sync_count1":np.zeros(1), "sync_count2":np.zeros(1), "acc_cnt1":np.zeros(1), "acc_cnt2":np.zeros(1),\
            "gpsStartTime":np.zeros(1), "gpsStopStime":np.zeros(1)}

def loadDiagnosticsData(path): # This might seem dangerous but it's fine because each folder contains 35.8kb of data, so very little
    folderCounter = 0
    # TODO: add for loop here to be able to add multiple folders/multiple time periods
    diagnosticDatabase["fpga_temp"] = np.concatenate((diagnosticDatabase["fpga_temp"], np.fromfile(os.path.join(path, "fpga_temp.raw"), dtype="float64"))) 
    diagnosticDatabase["pi_temp"]   = np.concatenate((diagnosticDatabase["pi_temp"], np.fromfile(os.path.join(path, "pi_temp.raw"), dtype="float64")))
    diagnosticDatabase["timeStart"] = np.concatenate((diagnosticDatabase["timeStart"], np.fromfile(os.path.join(path, "time_sys_start.raw"), dtype="float64").astype(np.int32)))
    diagnosticDatabase["timeStop"]  = np.concatenate((diagnosticDatabase["timeStop"], np.fromfile(os.path.join(path, "time_sys_stop.raw"), dtype="float64").astype(np.int32)))
    diagnosticDatabase["fft_overflow1"] = np.concatenate((diagnosticDatabase["fft_overflow1"], np.fromfile(os.path.join(path, "pfb_fft_of1.raw"), dtype="int32")))
    diagnosticDatabase["fft_overflow2"] = np.concatenate((diagnosticDatabase["fft_overflow2"], np.fromfile(os.path.join(path, "pfb_fft_of2.raw"), dtype="int32")))
    diagnosticDatabase["sync_count1"] = np.concatenate((diagnosticDatabase["sync_count1"], np.fromfile(os.path.join(path, "sync_cnt1.raw"), dtype="int32")))
    diagnosticDatabase["sync_count2"] = np.concatenate((diagnosticDatabase["sync_count2"], np.fromfile(os.path.join(path, "sync_cnt2.raw"), dtype="int32")))
    diagnosticDatabase["acc_cnt1"] = np.concatenate((diagnosticDatabase["acc_cnt1"], np.fromfile(os.path.join(path, "acc_cnt1.raw"), dtype="int32")))
    diagnosticDatabase["acc_cnt2"] = np.concatenate((diagnosticDatabase["acc_cnt2"], np.fromfile(os.path.join(path, "acc_cnt2.raw"), dtype="int32")))
    diagnosticDatabase["gpsStartTime"] = np.concatenate((diagnosticDatabase["gpsStartTime"], np.fromfile(os.path.join(path, "time_gps_start.raw"), dtype="int32")))
    diagnosticDatabase["gpsStopStime"] = np.concatenate((diagnosticDatabase["gpsStopStime"], np.fromfile(os.path.join(path, "time_gps_stop.raw"), dtype="int32")))
    if folderCounter == 0:
        for key in list(diagnosticDatabase.keys()):
            diagnosticDatabase[key] = diagnosticDatabase[key][1:] # Strip off the empty array
    folderCounter += 1

def getExtrema(data):
    minExt = {}
    maxExt = {}
    
    keys = np.array(list(data.keys()))
    for key in keys:
        if key in ["gpsStartTime", "gpsStopTime", "timeStart", "timeStop", "acc_cnt1", "acc_cnt2"]: continue # This line could be written better
        minExt[key] = np.min(data[key])
        maxExt[key] = np.max(data[key])

    temp = np.diff(data["timeStart"])
    minExt["timeStartDiff"] = np.min(temp)
    maxExt["timeStartDiff"] = np.max(temp)
    
    temp = np.diff(data["timeStop"])
    minExt["timeStopDiff"] = np.min(temp)
    maxExt["timeStopDiff"] = np.max(temp)
    
    temp = np.diff(data["acc_cnt1"] + data["acc_cnt2"])
    minExt["accCntDiff"] = np.min(temp)
    maxExt["accCntDiff"] = np.max(temp)

    return tuple((minExt, maxExt))

def flagging(diagnosticsExtrema):
    flag = 0b00000000   # This is how I'm going to encode errors. If flag = 0, then all checks passed
                        # 0b(hgfedcba)
    flag = flag | (int(diagnosticsExt[0]["fpga_temp"] < minTemp or diagnosticsExt[0]["pi_temp"] < minTemp or\
        diagnosticsExt[1]["fpga_temp"] > maxTemp or diagnosticsExt[1]["pi_temp"] > maxTemp) << 0) # a = temperature error
    flag = flag | (int(diagnosticsExt[0]["timeStartDiff"] < 0 or diagnosticsExt[0]["timeStopDiff"] < 0 or\
        diagnosticsExt[1]["timeStartDiff"] > maxTimeDiff or diagnosticsExt[1]["timeStopDiff"] > maxTimeDiff) << 1) # b = time diff error
    flag = flag | (int(diagnosticsExt[0]["accCntDiff"] != 2 or diagnosticsExt[1]["accCntDiff"] != 2) << 2) # c = acc_cnt diff error
    flag = flag | (int(diagnosticsExt[0]["fft_overflow1"] != 0 or diagnosticsExt[0]["fft_overflow2"] != 0 or\
        diagnosticsExt[1]["fft_overflow1"] != 0 or diagnosticsExt[1]["fft_overflow2"] != 0) << 3) # d = fft overflow error
    flag = flag | (int(diagnosticsExt[0]["sync_count1"] != 0 or diagnosticsExt[0]["sync_count2"] != 0 or\
        diagnosticsExt[1]["sync_count1"] != 0 or diagnosticsExt[1]["sync_count2"] != 0) << 4) # e = sync_count error
    # TODO: add the check thingy for the lst_binning script
    return flag

def plot(data):
    # For the x axis
    timePlotX = data["timeStop"]-data["timeStop"][0]

    # What we care about is that everything is +1 so we want a straight line to check possible errors
    acc_cnt1 = np.diff(data["acc_cnt1"])
    acc_cnt2 = np.diff(data["acc_cnt2"])

    fig, axis = plt.subplots(2, 3)
    axis[0, 0].plot(timePlotX, data["fpga_temp"], color="green", label="FPGA") # FPGA temp / fpga_temp.raw
    axis[0, 0].plot(timePlotX, data["pi_temp"], color="red", label="RPi") # Raspberry Pi Temperature / pi_temp.raw
    axis[0, 0].set_title("Temperature")
    axis[0, 0].set_ylabel("Temperature (°C)")
    axis[0, 0].set_xlabel("Time")
    axis[0, 0].legend()

    axis[0, 1].plot(timePlotX, data["fft_overflow1"], color="green", label="FFT Overflow 1")
    axis[0, 1].plot(timePlotX, data["fft_overflow2"], color="red", label="FFT Overflow 1")
    axis[0, 1].set_title("FFT Overflow Counter")
    axis[0, 1].set_ylabel("Count")
    axis[0, 1].set_xlabel("Time")
    axis[0, 1].legend()

    axis[1, 1].plot(timePlotX, data["sync_count1"], color="green", label="sync_cnt1")
    axis[1, 1].plot(timePlotX, data["sync_count2"], color="red", label="sync_cnt2")
    axis[1, 1].set_title("Sync_cnt")
    axis[1, 1].set_ylabel("Count")
    axis[1, 1].set_xlabel("Time")
    axis[1, 1].legend()

    axis[1, 0].plot(timePlotX, data["acc_cnt1"], color="green", label="acc_cnt1")
    axis[1, 0].plot(timePlotX, data["acc_cnt2"], color="red", label="acc_cnt2")
    axis[1, 0].set_title("Acc. Count")
    axis[1, 0].set_ylabel("Count")
    axis[1, 0].set_xlabel("Time")
    axis[1, 0].legend()

    axis[0, 2].plot(timePlotX, data["timeStop"]-data["timeStart"], "g.")
    axis[0, 2].set_title("Time difference between sys_stop and sys_start")
    axis[0, 2].set_ylabel("Difference (s)")
    axis[0, 2].set_xlabel("Time (s)")

    axis[1, 2].plot(timePlotX[:-1], np.diff(data["timeStart"]), "g.", label="time_sys_start")
    axis[1, 2].plot(timePlotX[:-1], np.diff(data["timeStop"]), "b.", label="time_sys_stop")
    # TODO: These two lines are currently broken, will have to fix
    # axis[1, 2].plot(gpsStartTime[:-1]-gpsStartTime[0], np.diff(gpsStartTime), color="red", label="time_gps_start")
    # axis[1, 2].plot(gpsStartTime[:-1]-gpsStartTime[0], np.diff(gpsStopStime), color="orange", label="time_gps_stop")
    axis[1, 2].set_title("Time differentials")
    axis[1, 2].set_ylabel("Difference (s)")
    axis[1, 2].set_xlabel("Time (s)")
    axis[1, 2].legend()

    plt.tight_layout()
    plt.show()

# Main block

"""I assumed that every measurement was float64 because it's the only data type that made sense, otherwise you'd get a bunch
of random e-44 garbage. I matched the fft_of shape to the shape of the measurement arrays, hard to test because
I haven't found one yet where there's an overflow (!= 0). Likewise for sync_cnt"""

database = loadDiagnosticsData('/home/samuel/Documents/albatros/uapishka/data_auto_cross/snap1/16279/1627900885')
# getExtrema(database)
plot(diagnosticDatabase)