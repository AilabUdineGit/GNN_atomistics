

#ifndef PY_LAMMPS_GNN
#define PY_LAMMPS_GNN


namespace PY_LAMMPS_GNN_NS {

void init();
void init_module();
void init_schnet(double, int, int, int, int, double, double);
void init_dimenet();

void destroy_schnet();
void destroy_dimenet();

double compute_energy(int*, double**, int, double**);

}

int main();

#endif