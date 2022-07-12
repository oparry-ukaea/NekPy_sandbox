## NekPy_sandbox
Repo for experimenting with various features of NekPy.

### Set up

1. Build and install Nektar as below:
```
git clone https://gitlab.nektar.info/nektar/nektar.git
cd nektar
mkdir build && cd build
cmake -DNEKTAR_BUILD_LIBRARY=ON -DNEKTAR_BUILD_PYTHON=ON -DNEKTAR_BUILD_UTILITIES=ON -DNEKTAR_USE_MPI=ON -DNEKTAR_USE_THREAD_SAFETY=ON ..
make -j4 install
```

2. In the directory containing this readme, generate and activate a Python virtual environment 
```
./make_env.sh [path_to_nektar_build_dir]
. ./env/bin/activate
```

---

The example scripts can then be run with, e.g.:
```
mpirun -n 4 python poisson_eg/poisson.py
```