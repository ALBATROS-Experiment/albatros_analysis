# Build the tools needed for this test
import os
from sys import platform

path = os.path.realpath(__file__ + r"/..")


def build():
    path_so = os.path.join(path, "libalbatrostools.so")
    path_c = os.path.join(path, "albatrostools.c")
    try:
        if platform == "darwin":
            print("Platform is 'darwin'")
            os.system(
                f"gcc -03 -o {path_so} -fPIC -Xpreprocessor -fopenmp --shared {path_c} -lomp"
            )
        else:
            print(f"Platform is unix/linux/window '{platform}'")
            os.system(f"gcc -o {path_so} -fPIC --shared {path_c} -fopenmp")
    except Exception as e:
        print("You need to add ur directory to LD library path")
        print(e)


if __name__ == "__main__":
    build()
