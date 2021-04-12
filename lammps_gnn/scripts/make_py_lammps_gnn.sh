#!/bin/bash
gcc -static `python3.8-config --cflags` -fPIE -c py_lammps_gnn.cpp -o py_lammps_gnn.o  # Compile
gcc py_lammps_gnn.o `python3.8-config --embed --ldflags` -o py_lammps_gnn    # Link
#rm *.o