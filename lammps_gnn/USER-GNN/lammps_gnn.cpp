

#include "stdio.h"
#include "lammps_gnn.h"
#include "error.h"
#include <memory>

#include "Python.h"

using namespace LAMMPS_NS;


GNN* schnet_from_args(int, char**);
GNN* dimenet_from_args(int, char**);

GNN* gnn_from_args(int narg, char** arg){
	if(narg == 0){
		perror("You must supply a GNN type along with its parameters.");
	}
	printf("-- creating GNN --\n");

	if(strcmp(arg[0], "SchNet") == 0){
		printf("-- creating schnet --\n");
		return schnet_from_args(narg - 1, arg + 1);
	}

	if(strcmp(arg[0], "DimeNet") == 0){
		return dimenet_from_args(narg - 1, arg + 1	);
	}

	perror("Illegal GNN type. Supported types: SchNet, Dimenet");
	return nullptr;
}

GNN* schnet_from_args(int narg, char** arg){
	if(narg != 7){
		printf("narg: %d", narg);
		perror("Illegal number of arguments for SchNet.");
	}

	double _cutoff =  atof(arg[0]);
	int _hidden_channels = atoi(arg[1]); 
	int _num_filters = atoi(arg[2]); 
	int _num_interactions = atoi(arg[3]);
	int _num_gaussians = atoi(arg[4]);
	double _mean = atof(arg[5]);
	double _std = atof(arg[6]);

	GNN* net = new SchNet(_cutoff, _hidden_channels, _num_filters, _num_interactions, 
								_num_gaussians, _mean, _std);
	return net;
}

GNN* dimenet_from_args(int narg, char** arg){
	if(narg != 9){
		printf("narg: %d", narg);
		perror("Illegal number of arguments for DimeNet.");
	}

	double _cutoff =  atof(arg[0]);
	int _hidden_channels = atoi(arg[1]);
	int _out_channels = atoi(arg[2]);
	int _num_blocks = atoi(arg[3]);
	int _num_bilinear = atoi(arg[4]);;
	int _num_spherical = atoi(arg[5]);;
	int _num_radial = atoi(arg[6]);
	double _mean = atof(arg[7]);
	double _std = atof(arg[8]);

	GNN* net = new DimeNet(_cutoff, _hidden_channels, _out_channels, _num_blocks, 
									_num_bilinear, _num_spherical, _num_radial, _mean, _std);
	return net;
}
 
// ===========================================================================
// SCHNET METHODS
// ===========================================================================

SchNet::SchNet(double _cutoff, int _hidden_channels, int _num_filters, int _num_interactions, 
								int _num_gaussians, double _mean, double _std){
	cutoff = _cutoff; 
	hidden_channels = _hidden_channels;
	num_filters = _num_filters;
	num_interactions = _num_interactions;
	num_gaussians = _num_gaussians;
	mean = _mean;
	std = _std;	

	printf("-- Instantiated SchNet with: %f %d %d %d %d %f %f --\n", cutoff, hidden_channels, num_filters, 
					num_interactions, num_gaussians, mean, std);
}

SchNet::~SchNet(){
	printf("-- Destroyed SchNet --\n");
}

double SchNet::compute_energy(int n, double** x){
	printf("-- SchNet::compute_energy --\n");
	for(int i=0; i<n; i++)
		printf("-- %f %f %f --\n", x[i][0], x[i][1], x[i][2]);
	return 10.0;
} 

// ===========================================================================
// DIMENET METHODS
// ===========================================================================

DimeNet::DimeNet(double _cutoff, int _hidden_channels, int _out_channels, int _num_blocks, 
									int _num_bilinear, int _num_spherical, int _num_radial, double _mean, double _std){
	cutoff = _cutoff;
	hidden_channels = _hidden_channels;
	out_channels = _out_channels;
	num_blocks = _num_blocks; 
	num_bilinear = _num_bilinear;
	num_spherical = _num_spherical;
	num_radial = _num_radial;
	mean = _mean;
	std = _std;

	printf("-- Instantiated DimeNet with: %f %d %d %d %d %d %d %f %f --\n", cutoff, hidden_channels,
		out_channels, num_blocks, num_bilinear, num_spherical, num_radial, mean, std);
}

DimeNet::~DimeNet(){
	printf("-- Destroyed DimeNet --\n");
}

double DimeNet::compute_energy(int n, double** x){
	printf("-- DimeNet::compute_energy --\n");
	for(int i=0; i<n; i++)
		printf("-- %f %f %f --\n", x[i][0], x[i][1], x[i][2]);
	return 20.0;
}
