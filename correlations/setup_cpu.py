mac_error_message = """
You may need to install llvm. Try the following:
`brew install llvm`
`brew install libomp`

If that doesn't work make sure llvm and libomp are properly linked.
Try adding the following to your bash profile (or .zshrc or .bashrc):

```
# Add llvm to path
export PATH="$PATH:/opt/homebrew/Cellar/llvm/16.0.6/bin" # lib or bin?

# C include path for omp, installed with `brew install libomp`
export C_INCLUDE_PATH="/opt/homebrew/Cellar/libomp/16.0.6/include:$C_INCLUDE_PATH"
export CPLUS_INCLUDE_PATH="/opt/homebrew/Cellar/libomp/16.0.6/include:$CPLUS_INCLUDE_PATH"
export LIBRARY_PATH="/opt/homebrew/Cellar/libomp/16.0.6/include:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/opt/homebrew/Cellar/libomp/16.0.6/include:$LD_LIBRARY_PATH"

# Try same for llvm lib
export PATH="/opt/homebrew/Cellar/llvm/16.0.6/lib:$PATH"
export C_INCLUDE_PATH="/opt/homebrew/Cellar/llvm/16.0.6/lib:$C_INCLUDE_PATH"
export CPLUS_INCLUDE_PATH="/opt/homebrew/Cellar/llvm/16.0.6/lib:$CPLUS_INCLUDE_PATH"
export LIBRARY_PATH="/opt/homebrew/Cellar/llvm/16.0.6/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/opt/homebrew/Cellar/llvm/16.0.6/lib:$LD_LIBRARY_PATH"
```

But with the appropriate paths for your versions of llvm and libomp.
"""


import os
from sys import platform  # detect whether using darwin or win/linux os

# builds into the same directory as the setup file
path = os.path.realpath(__file__ + r"/..")
print(f"file path {path}")


def build():
    if os.path.exists(path + "/unpacking.c"):
        path_so = os.path.join(path, "lib_unpacking.so")
        path_c = os.path.join(path, "unpacking.c")
        if platform == "darwin":
            # Mac OSx & ARM
            try:
                os.system(
                    f'gcc -shared -o "{path_so}" -fPIC -Xpreprocessor -fopenmp "{path_c}" -lomp'
                )
            except:
                raise Exception(mac_error_message)
        else:
            # Linux/Windows
            os.system(f'gcc -shared -o "{path_so}" -fPIC -fopenmp "{path_c}"')
    else:
        print(f"Cannot find the file unpacking.c in the directory {path}")

    if os.path.exists(os.path.join(path, "correlations_cpu.c")):
        path_so = os.path.join(path, "lib_correlations_cpu.so")
        path_c = os.path.join(path, "correlations_cpu.c")
        if platform == "darwin":
            try:
                os.system(
                    f'gcc -shared -o "{path_so}" -fPIC -Xpreprocessor -fopenmp "{path_c}" -lomp'
                )
            except:
                raise Exception(mac_error_message)
        else:
            # Linux/Windows
            os.system(f'gcc -shared -o "{path_so}" -fPIC -fopenmp "{path_c}"')
    else:
        print(f"Cannot find the file correlations_cpu.c in the directory {path}")


if __name__ == "__main__":
    build()
