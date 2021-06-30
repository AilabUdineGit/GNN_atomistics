  /* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Paul Crozier (SNL)
------------------------------------------------------------------------- */

#include <cmath>
#include <cstring>
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include "respa.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"
#include "stdio.h"

#include "domain.h"
#include "pair_gnn.h"
#include "lammps_gnn.h"


using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

PairGNN::PairGNN(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1; //?
  loaded = false;
}

/* ---------------------------------------------------------------------- */

PairGNN::~PairGNN()
{ 
  /*if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut);
    memory->destroy(epsilon);
    memory->destroy(sigma);
    memory->destroy(lj1);
    memory->destroy(lj2);
    memory->destroy(lj3);
    memory->destroy(lj4);
    memory->destroy(offset);
  }*/
}

/* ---------------------------------------------------------------------- */

void PairGNN::compute(int eflag, int vflag)
{
  if(!loaded){
    gnn->load_pretrained(model_path);
    loaded = true;
  } 

  ev_init(eflag,vflag);

  /*
  printf("-- eflag: %d, vflag: %d --\n", eflag, vflag);
  printf("-- evflag: %d --\n", evflag);
  printf("-- eflag_global: %d, eflag_atom: %d --\n", eflag_global, eflag_atom);
  printf("-- vflag_global: %d, vflag_atom: %d --\n", vflag_global, vflag_atom);
  */

  int n = atom->nlocal;
  double **f = atom->f;
  int *type = atom->type;

  long* z = new long[n];
  for(int i=0; i<n; i++)
    z[i] = map[type[i]];

  double e = gnn->compute_energy(n, 
    z,
    atom->x, 
    domain->boxhi[0] - domain->boxlo[0],
    domain->boxhi[1] - domain->boxlo[1],
    domain->boxhi[2] - domain->boxlo[2],
    f
  );

  if (eflag_atom) 
    error->all(FLERR, "per atom energy is not yet implemented for pair_style gnn");

  if (eflag) eng_vdwl= e;

  if (vflag_fdotr) virial_fdotr_compute();

  float xtmp, ytmp, ztmp;
  for (int i = 0; i < n; i++) {
    xtmp = atom->x[i][0];
    ytmp = atom->x[i][1];
    ztmp = atom->x[i][2];
    printf("%f %f %f -- %f %f %f\n", xtmp, ytmp, ztmp, f[i][0], f[i][1], f[i][2]);
  }

}


/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairGNN::allocate()
{
  allocated = 1;

  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 1;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(map,n+1,"pair:map");
  memory->create(model_path, 128,"pair:model_path");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */
// pair_style lorenzo **arg
void PairGNN::settings(int narg, char **arg)
{
  gnn = gnn_from_args(narg, arg);
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */
// pair_coeff **arg
void PairGNN::coeff(int narg, char **arg)
{
  if (narg < 3)
    error->all(FLERR, "Incorrect args for pair coefficients");

  if ((strcmp(arg[0], "*") && strcmp(arg[1], "*"))) 
    error->all(FLERR, "Incorrect args for pair coefficients\nYou should use \"pair_coeff * * <filename>\"");

  if (!allocated) allocate();

  int n = atom->ntypes;
  if (narg != 3 + 2 * n)
    error->all(FLERR, "Incorrect args for pair coefficients, some atom types are not mapped");

  strcpy(model_path, arg[2]);

  for (int i = 3; i < narg; i+=2) {
    map[atoi(arg[i])] = atoi(arg[i+1]);
  }
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairGNN::init_style()
{

}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairGNN::init_one(int i, int j)
{
  return 0.0;
}