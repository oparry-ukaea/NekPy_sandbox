#!/bin/bash

# Check args
if [ $# -ne 1 ]; then
    echo "Usage: $0 [path_to_nektar_build_dir] "
    echo " (Nektar must have been configured with BUILD_PYTHON=True and installed to generate setup.py)"
    exit 1
fi
nektar_dir="$1"

# Check NekPy's setup.py exists 
setup_path="$nektar_dir/setup.py"
if [ ! -f "$setup_path" ]; then
    echo "setup.py not found in $setup_path"
    exit 2
else 
    echo "Found $setup_path"
fi

# Remove existing environment if there is one
env_dir="env"
if [ -d "$env_dir" ]; then
    \rm -rf "$env_dir"; 
fi

# Set the path to NekPy in a temporary requirements file
tmp_requirements="tmp_requirements.txt"
sed "s|NEKPY_PATH|$nektar_dir|" < requirements.txt > "$tmp_requirements"

# Generate and activate the venv
python3 -m venv "$env_dir"
. "$env_dir/bin/activate"
pip install wheel

# Install packages
pip install -r $tmp_requirements

# Remove the temporary requirements file
if [ -f "$tmp_requirements" ]; then
    \rm "$tmp_requirements"
fi

printf "Activate the environment with\n   . $env_dir/bin/activate\n"