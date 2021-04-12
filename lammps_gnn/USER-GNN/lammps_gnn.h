
#ifndef LAMMPS_GNN
#define LAMMPS_GNN



namespace LAMMPS_NS {

class GNN {
	public:
		GNN(){};
		~GNN(){};
		virtual double compute_energy(int, double**) = 0;
};


class SchNet : public GNN {
	public:
		SchNet(double, int, int, int, int, double, double);
    ~SchNet(); 
    double compute_energy(int, double**);

  protected:
    double cutoff;
    int hidden_channels;
    int num_filters;
    int num_interactions;
    int num_gaussians;
    double mean;
    double std;
};

class DimeNet : public GNN {
  public:
    DimeNet(double, int, int, int, int, int, int, double, double);
    ~DimeNet();
    double compute_energy(int, double**);

  protected:
    double cutoff;
    int hidden_channels;
    int out_channels;
    int num_blocks;
    int num_bilinear;
    int num_spherical;
    int num_radial;
    double mean;
    double std;
};

}


LAMMPS_NS::GNN* gnn_from_args(int, char**);


#endif