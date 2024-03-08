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

from os.path import join, exists, dirname, realpath
from os import system
from sys import platform  # detect whether using darwin or win/linux os
from sys import path as sys_path

path = dirname(realpath(__file__))
print(f"Setup file path {path}")


def build():
    c_paths = [
        join(path, p)
        for p in ["correlations/unpacking.c", "correlations/correlations_cpu.c"]
    ]
    so_paths = [
        join(path, p)
        for p in [
            "correlations/lib_unpacking.so",
            "correlations/lib_correlations_cpu.so",
        ]
    ]

    for p in c_paths:
        if not exists(p):
            print(f"Cannot find the file {p}\n Stopping build!")
            exit(1)
    # proceed to build each path
    for path_c, path_so in zip(c_paths, so_paths):
        if platform == "darwin":
            # Mac OSx & ARM
            try:
                system(
                    f'gcc -shared -o "{path_so}" -fPIC -Xpreprocessor -fopenmp "{path_c}" -lomp'
                )
            except:
                raise Exception(mac_error_message)
        else:
            # Linux/Windows
            system(f'gcc -std=c99 -O3 -march=native -shared -o "{path_so}" -fPIC -fopenmp "{path_c}"')


if __name__ == "__main__":
    build()
