import subprocess
import sys
from pathlib import Path

# Get the script directory
path = Path(__file__).resolve().parent
print(f"File path: {path}")

# Error message for missing dependencies on macOS
mac_error_message = """
You may need to install LLVM and libomp. Try the following:
    brew install llvm libomp

If compilation fails, ensure the following lines are in your .zshrc (or equivalent shell config):

# Add llvm and OpenMP paths
export PATH="$PATH:/opt/homebrew/opt/llvm/bin"
export LDFLAGS="-L/opt/homebrew/opt/llvm/lib -L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/llvm/include -I/opt/homebrew/opt/libomp/include"

# Set OpenMP paths
export C_INCLUDE_PATH="/opt/homebrew/opt/libomp/include:$C_INCLUDE_PATH"
export CPLUS_INCLUDE_PATH="/opt/homebrew/opt/libomp/include:$CPLUS_INCLUDE_PATH"
export LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$LD_LIBRARY_PATH"

After updating your shell configuration, restart your terminal or run:
    source ~/.zshrc
"""


def build():
    """Compiles C files into shared libraries."""
    files_to_compile = ["unpacking.c", "correlations_cpu.c"]

    for filename in files_to_compile:
        path_c = path / filename
        path_so = path / f"lib_{path_c.stem}.so"

        if path_c.exists():
            compile_cmd = (
                [
                    "gcc",
                    "-shared",
                    "-o",
                    str(path_so),
                    "-fPIC",
                    "-Xpreprocessor",
                    "-fopenmp",
                    str(path_c),
                    "-lomp",
                ]
                if sys.platform == "darwin"
                else [
                    "gcc",
                    "-shared",
                    "-o",
                    str(path_so),
                    "-fPIC",
                    "-fopenmp",
                    str(path_c),
                ]
            )

            try:
                subprocess.run(compile_cmd, check=True)
                print(f"Successfully compiled {path_c.name} -> {path_so.name}")
            except subprocess.CalledProcessError:
                print(f"Compilation failed for {filename}!")
                sys.exit(
                    mac_error_message
                    if sys.platform == "darwin"
                    else "Compilation error."
                )

        else:
            print(f"Cannot find {filename} in the directory {path}")


if __name__ == "__main__":
    build()
