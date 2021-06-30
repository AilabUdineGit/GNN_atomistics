
#include "stdio.h"
#include <memory>

#include "lammps_gnn.h"
#include "py_lammps_gnn.h"
#include <Python.h>

using namespace LAMMPS_NS;


GNN* schnet_from_args(int, char**);
GNN* dimenet_from_args(int, char**);

/* 
	DRIVER 
	calls correct function depending on first arg (GNN name)
*/
GNN* gnn_from_args(int narg, char** arg){
	if(narg == 0){
		printf("[lammpsGNN]: You must supply a GNN type along with its parameters.");
		return nullptr;
	}
	/*printf("[lammpsGNN]: creating GNN from inputfile arguments\n");*/

	if(strcmp(arg[0], "SchNet") == 0){
		/*printf("[lammpsGNN]: creating schnet from arguments\n");*/
		return schnet_from_args(narg - 1, arg + 1);
	}

	if(strcmp(arg[0], "DimeNet") == 0){
		return dimenet_from_args(narg - 1, arg + 1	);
	}

	perror("[lammpsGNN]: illegal GNN type. Supported types are: SchNet, Dimenet\n");
	return nullptr;
}

/*
	SPECIALIZED FUNCTIONS
	parse args and call the correct constructor
*/

GNN* schnet_from_args(int narg, char** arg){
	if(narg != 7){
		printf("narg: %d", narg);
		perror("[lammpsGNN]: illegal number of arguments for SchNet");
	}

	double cutoff =  atof(arg[0]);
	int hidden_channels = atoi(arg[1]); 
	int num_filters = atoi(arg[2]); 
	int num_interactions = atoi(arg[3]);
	int num_gaussians = atoi(arg[4]);
	double mean = atof(arg[5]);
	double std = atof(arg[6]);

	GNN* net = new SchNet(cutoff, 
		hidden_channels, 
		num_filters, 
		num_interactions, 
		num_gaussians, 
		mean, std
	);
	return net;
}

GNN* dimenet_from_args(int narg, char** arg){
	if(narg != 9){
		printf("narg: %d", narg);
		perror("[lammpsGNN]: illegal number of arguments for DimeNet");
	}

	double cutoff =  atof(arg[0]);
	int hidden_channels = atoi(arg[1]);
	int out_channels = atoi(arg[2]);
	int num_blocks = atoi(arg[3]);
	int num_bilinear = atoi(arg[4]);;
	int num_spherical = atoi(arg[5]);;
	int num_radial = atoi(arg[6]);
	double mean = atof(arg[7]);
	double std = atof(arg[8]);

	GNN* net = new DimeNet(cutoff, 
		hidden_channels, 
		out_channels, 
		num_blocks, 
		num_bilinear, 
		num_spherical, 
		num_radial, 
		mean, std
	);
	return net;
}
 

// ===========================================================================
// PYGNN METHODS
// ===========================================================================

PyGNN::PyGNN(){
	PY_LAMMPS_GNN_NS::init_python();
}

double PyGNN::compute_energy(int n, long* z, double** x, double cellx, double celly, double cellz, double** f){
	return PY_LAMMPS_GNN_NS::compute_energy(net, n, z, x, cellx, celly, cellz, f);
}

void PyGNN::load_pretrained(char* model_path){
	PY_LAMMPS_GNN_NS::load_pretrained(net, model_path);
}

// ===========================================================================
// SCHNET METHODS
// ===========================================================================

SchNet::SchNet(double cutoff, int hidden_channels, int num_filters, int num_interactions, 
								int num_gaussians, double mean, double std){

	net = PY_LAMMPS_GNN_NS::new_schnet(cutoff, 
		hidden_channels, 
		num_filters,
		num_interactions, 
		num_gaussians, 
		mean, std
	);

	/*printf("[lammpsGNN]: instantiated SchNet with: %f %d %d %d %d %f %f \n", 
		cutoff, hidden_channels, num_filters, 
		num_interactions, num_gaussians, mean, std
	);*/
}

SchNet::~SchNet(){
	//printf("[lammpsGNN]: destroyed SchNet \n");
}


// ===========================================================================
// DIMENET METHODS
// ===========================================================================

DimeNet::DimeNet(double cutoff, int hidden_channels, int out_channels, int num_blocks, 
									int num_bilinear, int num_spherical, int num_radial, double mean, double std){

	net = PY_LAMMPS_GNN_NS::new_dimenet(cutoff, 
		hidden_channels, 
		out_channels, 
		num_blocks, 
		num_bilinear, 
		num_spherical, 
		num_radial, 
		mean, std
	);
}

DimeNet::~DimeNet(){
	/*printf("-- Destroyed DimeNet --\n");
	*/
}