import os
import sys

#builds into the same directory as the setup file
path = os.path.realpath(__file__+r"/..")

def build():
    if os.path.exists(path+"/unpacking.c"):
        os.system("gcc -shared -o \""+ path + "/lib_unpacking.so\" -fPIC \"" + path + "/unpacking.c\"")
    else:
        print("Cannot find the file unpacking.c in the directory "+path)
    
    if os.path.exists(path+"/correlations_cpu.c"):
        os.system("gcc -shared -o \""+ path + "/lib_correlations_cpu.so\" -fPIC \"" + path + "/correlations_cpu.c\"")
    else:
        print("Cannot find the file correlations_cpu.c in the directory "+path)
