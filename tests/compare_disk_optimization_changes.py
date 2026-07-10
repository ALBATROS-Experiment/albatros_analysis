import numpy as np
import glob
import os
import re

def load_all_parts(dir_path):
    """
    Finds all visibility part files in a directory, sorts them numerically,
    and loads them into a single pre-allocated big array.
    """
    # Find all part files
    pattern = os.path.join(dir_path, "*_part*.npy")
    part_files = glob.glob(pattern)
    
    if not part_files:
        print(f"No part files found in {dir_path}")
        return None

    # Extract part number and sort numerically
    def get_part_num(f):
        match = re.search(r'_part(\d+)', f)
        return int(match.group(1)) if match else -1
        
    part_files.sort(key=get_part_num)
    num_files = len(part_files)
    
    print(f"Found {num_files} files. Pre-allocating and loading...")

    # Load first file to get dimensions and dtype
    first_arr = np.load(part_files[0])
    print("shape of first file", first_arr.shape)
    nbl, chunk_time, nchan, npol2 = first_arr.shape
    dtype = first_arr.dtype

    last_arr = np.load(part_files[-1])
    total_time = (num_files-1)*chunk_time + last_arr.shape[1]

    # Pre-allocate the big array
    full_array = np.empty((nbl, total_time, nchan, npol2), dtype=dtype)
    full_array[:, :chunk_time, : , :] = first_arr
    full_array[:, -last_arr.shape[1]:, : , :] = last_arr
    del first_arr, last_arr
    # Fill the array

    start=chunk_time
    for i in range(1, num_files-1):
        f = part_files[i]
        arr = np.load(f)
        end = start + arr.shape[1]
        full_array[:, start:end, :, :] = arr
        start = end
        print(f"  Processed {os.path.basename(f)} into slice [{start}:{start+arr.shape[1]}]")
    
    print(f"\nLoading complete.")
    print(f"Resulting shape: {full_array.shape}")
    print(f"Total size: {full_array.nbytes / 1024**3:.2f} GB")

    return full_array

if __name__ == "__main__":
    # Path provided by user
    with_changes = "/scratch/thomasb/mohan/disk_optimize_results/vis_ant=7_pol=2_cha=196:280_20260416T193207"
    without_changes = "/scratch/thomasb/mohan/disk_optimize_results/vis_ant=7_pol=2_cha=196:280_test_nochange_IQ_20260416T233430"
    
    data1 = load_all_parts(with_changes)
    data2 = load_all_parts(without_changes)
    n=data1.shape[1] #in the old code, I was saving the whole final vis_file array
    assert np.array_equal(data1, data2[:,:n,:,:])
    # print(data1.shape, data2.shape)
