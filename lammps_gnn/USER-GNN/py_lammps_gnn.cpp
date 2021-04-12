

#include "stdio.h"
#include <cstdarg>
#include <Python.h>
#include "py_lammps_gnn.h"

#define PY_LAMMPS_GNN__PY_MODULE_FN "py_lammps_gnn"


void Py_DECREF_v(int , ...);
void Py_DECREF_array(int, PyObject**);


using namespace PY_LAMMPS_GNN_NS;

PyObject* module;
PyObject* net;


//==================================================================================
// INITIALIZATION & CONSTRUCTION
//==================================================================================

void PY_LAMMPS_GNN_NS::init(){
	Py_Initialize();
	init_module();
}

void PY_LAMMPS_GNN_NS::init_module(){
	PyObject* moduleString = PyUnicode_FromString(PY_LAMMPS_GNN__PY_MODULE_FN);
	module = PyImport_Import(moduleString);
	if(module == nullptr){
		PyErr_Print();
		printf("Error: cannot import module %s\n", PY_LAMMPS_GNN__PY_MODULE_FN);
		return;
	}
	//Py_DECREF(moduleString);
}

void PY_LAMMPS_GNN_NS::init_schnet(double _cutoff, int _hidden_channels, int _num_filters, 
		int _num_interactions, int _num_gaussians, double _mean, double _std){

	PyObject* net_class = PyObject_GetAttrString(module, "SchNet");

	PyObject* cutoff = PyFloat_FromDouble(_cutoff);
	PyObject* hidden_channels = PyLong_FromLong(_hidden_channels); 
	PyObject* num_filters = PyLong_FromLong(_num_filters);
	PyObject* num_interactions = PyLong_FromLong(_num_interactions);
	PyObject* num_gaussians =  PyLong_FromLong(_num_gaussians);
	PyObject* mean = PyFloat_FromDouble(_mean);
	PyObject* std = PyFloat_FromDouble(_std);
	PyObject* args = PyTuple_Pack(7, cutoff, hidden_channels, num_filters, 
																num_interactions, num_gaussians, mean, std);

	net = PyObject_CallObject(net_class, args);

	//Py_DECREF_v(8, net_class, cutoff, hidden_channels, num_filters, 
	//						num_interactions, num_gaussians, mean, std);
}

void PY_LAMMPS_GNN_NS::init_dimenet(){

}


//==================================================================================
// DESTRUCTION
//==================================================================================

void PY_LAMMPS_GNN_NS::destroy_schnet(){

}

void PY_LAMMPS_GNN_NS::destroy_dimenet(){

}


//==================================================================================
// COMPUTATION
//==================================================================================

/*
	TODO: should store all sublist refs in a PyObject** array
				to later DECREF them all
*/
double PY_LAMMPS_GNN_NS::compute_energy(int* z, double** x, int n, double** cell){
	
	PyObject* py_z = PyList_New(n);
	PyObject* py_x = PyList_New(n);
	PyObject* py_cell = PyList_New(3);

	PyObject* py_sublist = nullptr;
	PyObject* py_v = nullptr;
	int i, j;

	for(i=0; i<n; i++){
		py_v = PyLong_FromLong(z[i]);
		PyList_SetItem(py_z, i, py_v);
	}

	for(i=0; i<n; i++){
		py_sublist = PyList_New(3);
		for(j=0; j<3; j++){
			py_v = PyFloat_FromDouble(x[i][j]);
			PyList_SetItem(py_sublist, j, py_v);
		}
		PyList_SetItem(py_x, i, py_sublist);
	}

	for(i=0; i<3; i++){
		py_sublist = PyList_New(3);
		for(j=0; j<3; j++){
			py_v = PyFloat_FromDouble(cell[i][j]);
			PyList_SetItem(py_sublist, j, py_v);
		}
		PyList_SetItem(py_cell, i, py_sublist);
	}

	PyObject* py_res = PyObject_CallMethod(net, "compute_energy", "OOO", py_z, py_x, py_cell);
	if(py_res == nullptr){
		PyErr_Print();
		printf("Error in call\n");
	}
	double res = PyFloat_AsDouble(py_res);

	//Py_DECREF_v(6, py_z, py_x, py_cell, py_sublist, py_v, py_res);

	return res;
}

//==================================================================================
// HELPER FUNCTIONS
//==================================================================================

void Py_DECREF_array(int n, PyObject** arr){
	for(int i=0; i<n; i++){
		Py_DECREF(arr[i]);
	}
}

void Py_DECREF_v(int n, ...){
	PyObject* o;
	va_list args;
	va_start(args, n);
	for(int i=0; i<n; i++){
		o = va_arg(args, PyObject*);
		Py_DECREF(o);
	}
 	va_end(args);
}

//==================================================================================
// TESTING
//==================================================================================

int main(){
	printf("ok\n");
	init();
	init_schnet(3.5, 1, 1, 1, 1, 1.0, 0.1);

	int n = 2;
	int i;

	int* z = (int*) malloc(sizeof(int*) * n);
	double** x = (double**) malloc(sizeof(double**) * n);
	for(i=0; i<n; i++){
		z[i] = 26;
		x[i] = (double*) malloc(sizeof(double*) * 3);
	}
	x[0][0] = 1.0;
	x[0][1] = 2.0;
	x[0][2] = 1.0;
	x[1][0] = 0.5;
	x[1][1] = 3.0;
	x[1][2] = 3.0;

	double** cell = (double**) malloc(sizeof(double*)*3);
	for(i=0;i<3;i++){
		cell[i] = (double*) malloc(sizeof(double*)*3);
	}
	cell[0][0] = 3.0;
	cell[0][1] = 0.0;
	cell[0][2] = 0.0;
	cell[1][0] = 0.0;
	cell[1][1] = 3.0;
	cell[1][2] = 0.0;
	cell[2][0] = 1.5;
	cell[2][1] = 1.5;
	cell[2][2] = 1.5;


	double en = compute_energy(z, x, n, cell);
	return 0;
}