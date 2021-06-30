

#include "stdio.h"
#include <cstdarg>

#include <Python.h>
#include "py_lammps_gnn.h"


/* filenames of top level modules containing GNN class definitions */
#define PY_LAMMPS_GNN__SCHNET_MODULE_FN 	"schnet"
#define PY_LAMMPS_GNN__DIMENET_MODULE_FN 	"dimenet"


void Py_DECREF_v(int , ...);
void Py_DECREF_array(int, PyObject**);


using namespace PY_LAMMPS_GNN_NS;

//==================================================================================
// INITIALIZATION, IMPORT
//==================================================================================

void PY_LAMMPS_GNN_NS::init_python(){
	//printf("[lammpsGNN]: initializing Python environment\n");
	Py_Initialize();
}

PyObject* load_module(const char* module_fn){
	//printf("[lammpsGNN]: importing Python module %s\n", module_fn);
	PyObject* moduleString = PyUnicode_FromString(module_fn);
	PyObject* module = PyImport_Import(moduleString);
	if(module == nullptr){
		PyErr_Print();
		printf("[lammpsGNN]: error: failed to import module %s\n", module_fn);
	}
	// TODO: should decref some stuff
	//Py_DECREF(moduleString);
	return module;
}


//==================================================================================
// CONSTRUCTION
//==================================================================================

PyObject* PY_LAMMPS_GNN_NS::new_schnet(double _cutoff, 
	int _hidden_channels, int _num_filters, int _num_interactions, 
	int _num_gaussians, double _mean, double _std){

	PyObject* module = load_module(PY_LAMMPS_GNN__SCHNET_MODULE_FN);
	PyObject* net_class = PyObject_GetAttrString(module, "SchNet");
	if(net_class == nullptr){
		PyErr_Print();
		printf("[lammpsGNN]: error loading class\n");
	}

	PyObject* new_from_lammps = PyObject_GetAttrString(net_class, "new_from_lammps");

	PyObject* cutoff = PyFloat_FromDouble(_cutoff);
	PyObject* hidden_channels = PyLong_FromLong(_hidden_channels); 
	PyObject* num_filters = PyLong_FromLong(_num_filters);
	PyObject* num_interactions = PyLong_FromLong(_num_interactions);
	PyObject* num_gaussians =  PyLong_FromLong(_num_gaussians);
	PyObject* mean = PyFloat_FromDouble(_mean);
	PyObject* std = PyFloat_FromDouble(_std);
	PyObject* args = PyTuple_Pack(7, 
		cutoff, hidden_channels, num_filters, 
		num_interactions, num_gaussians, mean, std);

	PyObject* net = PyObject_CallObject(new_from_lammps, args);
	if(net == nullptr){
		PyErr_Print();
		printf("[lammpsGNN]: failed to instantiate SchNet\n");
	}
	return net;
	// TODO: decref some stuff
	//Py_DECREF_v(8, net_class, cutoff, hidden_channels, num_filters, 
	//						num_interactions, num_gaussians, mean, std);
}

PyObject* PY_LAMMPS_GNN_NS::new_dimenet(double _cutoff, int _hidden_channels, 
	int _out_channels, int _num_blocks, int _num_bilinear, int _num_spherical, 
	int _num_radial, double _mean, double _std){

	PyObject* module = load_module(PY_LAMMPS_GNN__DIMENET_MODULE_FN);
	PyObject* net_class = PyObject_GetAttrString(module, "DimeNet2");
	if(net_class == nullptr){
		PyErr_Print();
		printf("[lammpsGNN]: error loading class\n");
	}

	PyObject* new_from_lammps = PyObject_GetAttrString(net_class, "new_from_lammps");

	PyObject* cutoff = PyFloat_FromDouble(_cutoff);
	PyObject* hidden_channels = PyLong_FromLong(_hidden_channels); 
	PyObject* out_channels = PyLong_FromLong(_out_channels);
	PyObject* num_blocks = PyLong_FromLong(_num_blocks);
	PyObject* num_bilinear =  PyLong_FromLong(_num_bilinear);
	PyObject* num_spherical = PyLong_FromLong(_num_spherical);
	PyObject* num_radial = PyLong_FromLong(_num_radial);
	PyObject* mean = PyFloat_FromDouble(_mean);
	PyObject* std = PyFloat_FromDouble(_std);
	PyObject* args = PyTuple_Pack(9, 
		cutoff, hidden_channels, out_channels, num_blocks, 
		num_bilinear, num_spherical, num_radial, mean, std);
	PyObject* net = PyObject_CallObject(new_from_lammps, args);
	if(net == nullptr){
		PyErr_Print();
		printf("[lammpsGNN]: failed to instantiate DimeNet\n");
	}
	return net;
}

void PY_LAMMPS_GNN_NS::load_pretrained(PyObject* net, char* _file_path){

	PyObject* file_path = PyUnicode_FromString(_file_path);
	if(file_path == nullptr){
		PyErr_Print();
		printf("[lammpsGNN]: error creating string\n");
	}
	PyObject* args = PyTuple_Pack(1, file_path);
	PyObject_CallMethod(net, "load_pretrained", "O", args);
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

double PY_LAMMPS_GNN_NS::compute(PyObject* net, long* z, double** x, int nidx_unique, int* idx_local, int nidx_local, double** f){
	
	PyObject* py_sublist = nullptr;
	PyObject* py_v = nullptr;
	int i, j;

	// make z PyList
	PyObject* py_z = PyList_New(nidx_unique);
	for(i=0; i<nidx_unique; i++){
		py_v = PyLong_FromLong(z[i]);
		PyList_SetItem(py_z, i, py_v);
	}

	// make x PyList
	PyObject* py_x = PyList_New(nidx_unique);
	for(i=0; i<nidx_unique; i++){
		py_sublist = PyList_New(3);
		for(j=0; j<3; j++){
			py_v = PyFloat_FromDouble(x[i][j]);
			PyList_SetItem(py_sublist, j, py_v);
		}
		PyList_SetItem(py_x, i, py_sublist);
	}

	// make idx_local PyList
	PyObject* py_idx_local = PyList_New(nidx_local);
	for(i=0; i<nidx_local; i++){
		py_v = PyLong_FromLong(idx_local[i]);
		PyList_SetItem(py_idx_local, i, py_v);
	}

	PyObject* py_res = PyObject_CallMethod(net, "compute_from_lammps", "OOO", py_z, py_x, py_idx_local);
	if(py_res == nullptr){
		PyErr_Print();
		printf("[lammpsGNN]: error calling compute_from_lammps\n");
	}

	PyObject* py_e = PyTuple_GetItem(py_res, 0);
	double e = PyFloat_AsDouble(py_e);
	
	
	PyObject* py_fm = PyTuple_GetItem(py_res, 1);
	PyObject* py_fv = nullptr;
	PyObject* py_f = nullptr;
	double c_f;
	for(i=0; i<nidx_local; i++){
		py_fv = PyList_GetItem(py_fm, i);
		for(j=0; j<3; j++){
			py_f = PyList_GetItem(py_fv, j);
			c_f = PyFloat_AsDouble(py_f);
			f[i][j] = c_f;
		}
	}
	
	return e;
}

double PY_LAMMPS_GNN_NS::compute_pa(PyObject* net, long* z, double** x, int nidx_unique, int* idx_local, int nidx_local, double** f, double* e_pa){
	
	PyObject* py_sublist = nullptr;
	PyObject* py_v = nullptr;
	int i, j;

	// make z PyList
	PyObject* py_z = PyList_New(nidx_unique);
	for(i=0; i<nidx_unique; i++){
		py_v = PyLong_FromLong(z[i]);
		PyList_SetItem(py_z, i, py_v);
	}

	// make x PyList
	PyObject* py_x = PyList_New(nidx_unique);
	for(i=0; i<nidx_unique; i++){
		py_sublist = PyList_New(3);
		for(j=0; j<3; j++){
			py_v = PyFloat_FromDouble(x[i][j]);
			PyList_SetItem(py_sublist, j, py_v);
		}
		PyList_SetItem(py_x, i, py_sublist);
	}

	// make idx_local PyList
	PyObject* py_idx_local = PyList_New(nidx_local);
	for(i=0; i<nidx_local; i++){
		py_v = PyLong_FromLong(idx_local[i]);
		PyList_SetItem(py_idx_local, i, py_v);
	}

	PyObject* py_res = PyObject_CallMethod(net, "compute_pa_from_lammps", "OOO", py_z, py_x, py_idx_local);
	if(py_res == nullptr){
		PyErr_Print();
		printf("[lammpsGNN]: error calling compute_from_lammps\n");
	}

	PyObject* py_e = PyTuple_GetItem(py_res, 0);
	double e = PyFloat_AsDouble(py_e);
	
	
	PyObject* py_fm = PyTuple_GetItem(py_res, 1);
	PyObject* py_fv = nullptr;
	PyObject* py_f = nullptr;
	double c_f;
	for(i=0; i<nidx_local; i++){
		py_fv = PyList_GetItem(py_fm, i);
		for(j=0; j<3; j++){
			py_f = PyList_GetItem(py_fv, j);
			c_f = PyFloat_AsDouble(py_f);
			f[i][j] = c_f;
		}
	}

	PyObject* py_ev = PyTuple_GetItem(py_res, 2);
	double c_e;
	for(i=0; i<nidx_local; i++){
		py_e = PyList_GetItem(py_ev, i);
		c_e = PyFloat_AsDouble(py_e);
		e_pa[i] = c_e;
	}
	
	return e;
}



//==================================================================================
// HELPERS
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

/*int main(){
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
}*/