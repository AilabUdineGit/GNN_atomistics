

#ifndef PY_LAMMPS_GNN_H
#define PY_LAMMPS_GNN_H

#include <Python.h>

namespace PY_LAMMPS_GNN_NS {

void init_python();
//void init_module();

PyObject* new_schnet(double, int, int, int, int, double, double);
PyObject* new_dimenet(double, int, int, int, int, int, int, double, double);

void destroy_schnet();
void destroy_dimenet();

void load_pretrained(PyObject*, char*);

double compute_energy(PyObject*, int, long*, double**, double, double, double, double**);


}

int main();

#endif