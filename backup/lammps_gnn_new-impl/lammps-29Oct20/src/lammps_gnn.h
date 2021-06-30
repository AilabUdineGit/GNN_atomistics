
#ifndef LAMMPS_GNN_H
#define LAMMPS_GNN_H

#include <Python.h>

namespace LAMMPS_NS { // maybe change NS?

// main abstract class
class GNN {
	public:
		virtual double compute(long*, double**, int, int*, int, double**) = 0;
		virtual double compute_pa(long*, double**, int, int*, int, double**, double*) = 0;
		virtual void load_pretrained(char*) = 0;
};

// GNN with Python implementation
class PyGNN : public GNN {
	public:
		PyGNN();
		double compute(long*, double**, int, int*, int, double**);
		double compute_pa(long*, double**, int, int*, int, double**, double*);
		void load_pretrained(char*);

	protected:
		PyObject* net;
};

class SchNet : public PyGNN {
	public:
		SchNet(double, int, int, int, int, double, double);
		~SchNet();
};

class DimeNet : public PyGNN {
	public:
		DimeNet(double, int, int, int, int, int, int, double, double);
		~DimeNet();
};

}


LAMMPS_NS::GNN* gnn_from_args(int, char**, float*);


#endif