#!/bin/bash

source /opt/intel/oneapi/setvars.sh
#module load compiler/intel/
#module load numlib/mkl
source ~/anaconda3/bin/activate

f2py3 --fcompiler=gfortran -m eta_gamma_pade -c eta_gamma_pade.f90 --opt="-Ofast"
f2py3 --fcompiler=gfortran -m sparsity -c sparsity.f90 --f90flags="-ffree-line-length-512" --opt="-Ofast"
f2py3 --fcompiler=gfortran -m generate_heom_one_x -c generate_heom_one_x.f90 --f90flags="-ffree-line-length-512" --opt="-Ofast"
gfortran -c -fPIC sparse_liouvillian_storage.f90 -I/opt/intel/oneapi/mkl/latest/include
f2py3 --fcompiler=gfortran \
  -m sparse_propagation \
  -c sparse_propagation.f90 sparse_liouvillian_storage.o \
  --f90flags="-fopenmp -ffree-line-length-512 -fPIC -I/opt/intel/oneapi/mkl/latest/include" \
  --opt="-O3 -fopenmp" \
  -L/opt/intel/oneapi/mkl/latest/lib/intel64 \
  -lmkl_gf_lp64 -lmkl_core -lmkl_gnu_thread -lpthread -lm -ldl -lgomp
