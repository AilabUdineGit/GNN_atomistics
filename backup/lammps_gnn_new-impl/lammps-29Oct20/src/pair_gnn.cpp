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

#include <stdlib.h>
#include "domain.h"
#include "pair_gnn.h"
#include "lammps_gnn.h"


using namespace LAMMPS_NS;
using namespace MathConst;

int compareInt(const void* a, const void* b)
{
  return ( *(int*)a - *(int*)b );
}

/* ---------------------------------------------------------------------- */

PairGNN::PairGNN(LAMMPS *lmp) : Pair(lmp)
{
	printf("PairGNN\n");
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  loaded = false;
  //ghostneigh = 1;
  //nextra = 3;
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
  if(comm->nprocs != 1){
    printf("I am proc %d\n", comm->me);
  }
  */

  int i, j, k, ii, jj, jnum;
  int *jlist;

  int *type = atom->type; // array of atom types
  int inum = list->inum;  // number of owned/local atoms
  int* ilist = list->ilist;  // local atoms indices
  int* numneigh = list->numneigh; // length of neighbor list of each local atom
  int** firstneigh = list->firstneigh;  // neighbor lists of each local atom

  int nlocal = inum; // number of local atoms
  int nidx = inum;   // upper bound of number of distinct atoms (local + neighbors) (there might be duplicates)
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    nidx += numneigh[i];
  }

  int* idx = nullptr;  // indices of local + neighbor atoms with possible duplicates
  memory->create(idx, nidx, "idx");
  k = 0;
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    idx[k++] = i;
    jnum = numneigh[i];
    jlist = firstneigh[i];
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      idx[k++] = j;
    }
  }
  qsort(idx, nidx, sizeof(int), compareInt);  

  int* idx_unique = nullptr; // distinct indices of local + neighbor atoms (== local + ghost)
  memory->create(idx_unique, nidx, "idx_unique");
  j = 0;
  for(i=0; i<nidx; i++){
    if(i == 0 || idx[i] != idx[i-1]){
      idx_unique[j++] = idx[i];
    }
  }
  int nidx_unique = j;
  memory->grow(idx_unique, nidx_unique, "idx_unique"); // shrink array

  int* idx_local = nullptr;  // indices of local atoms relative to idx_unique
  /* 
    i.e. idx_local[ii] = j s.t. idx_unique[j] = i = ilist[ii]
  */
  memory->create(idx_local, nlocal, "idx_local");
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    for(j=0; j<nidx_unique; j++){
      if(idx_unique[j] == i){
        idx_local[ii] = j;
        j = nidx_unique; // found, break
      }
    }
  }
  int nidx_local = nlocal;

  double** f = nullptr; // force array to accomodate result for local atoms
  memory->create(f, nidx_local, 3, "f");
  long* z = nullptr;    // atomic number of each of the local + ghost atoms relative to idx_unique
  // i.e. z[i] = z[idx_unique[i]]
  memory->create(z, nidx_unique, "z");
  for(i=0; i<nidx_unique; i++){
    z[i] = map[type[idx_unique[i]]];
  }
  double** x = nullptr; // positions of each of the local + ghost atoms relative to idx_unique
  // i.e. x[i] = atom->x[idx_unique[i]]
  memory->create(x, nidx_unique, 3, "x");
  for(i=0; i<nidx_unique; i++){
    x[i][0] = atom->x[idx_unique[i]][0];
    x[i][1] = atom->x[idx_unique[i]][1];
    x[i][2] = atom->x[idx_unique[i]][2];
  }

  double e = 0.0;
  if (eflag_atom){
    double* e_pa = nullptr;
    memory->create(e_pa, nidx_local, "e_pa"); // per-atom energy array to accomodate result for local atoms 
    e = gnn->compute_pa(z, x, nidx_unique, idx_local, nidx_local, f, e_pa);
    for(i=0; i<nidx_local; i++){
      eatom[idx_unique[idx_local[i]]] += e_pa[i];
    }
  } else {
    e = gnn->compute(z, x, nidx_unique, idx_local, nidx_local, f);
  }

  if (eflag_global){
    eng_vdwl += e;
  }

  for(i=0; i<nidx_local; i++){
    atom->f[idx_unique[idx_local[i]]][0] += f[i][0];
    atom->f[idx_unique[idx_local[i]]][1] += f[i][1];
    atom->f[idx_unique[idx_local[i]]][2] += f[i][2];
  }

  if (vflag_fdotr){
    virial_fdotr_compute();
  }


  //e = gnn->compute(z, x, imax, idx, k, local_idx, nlocal, f);
  /*
  double** f = nullptr;
  memory->create(f, imax, imax, "f");
  long* z = nullptr;
  memory->create(z, imax, "z");
  for(i=0; i<imax; i++){
    z[i] = map[type[i]];
  }
  double** x = nullptr;
  memory->create(x, imax, 3, "x");
  for(i=0; i<imax; i++){
    x[i][0] = atom->x[i][0];
    x[i][1] = atom->x[i][1];
    x[i][2] = atom->x[i][2];
  }
  // store indices of local 
  int* local_idx = nullptr;
  memory->create(local_idx, nlocal, "local_idx");
  for (ii = 0; ii < inum; ii++) {
    local_idx[i] = ilist[ii];
  }
  // store indices of local + neighbors
  int* idx = nullptr;
  memory->create(idx, ntotal, "idx");
  k = 0;  
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    idx[k++] = i;
    jnum = numneigh[i];
    jlist = firstneigh[i];
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      idx[k++] = j;
    }
  }
  */


  /*
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    printf("owned atom %d with id %d \n", ii, i);
    xtmp = atom->x[i][0];
    ytmp = atom->x[i][1];
    ztmp = atom->x[i][2];
    printf("at position x[i][0:3]: %f %f %f\n", xtmp, ytmp, ztmp);
    itype = type[i];
    //printf("itype = type[i]: %d\n", itype);
    jlist = firstneigh[i];
    jnum = numneigh[i];
    printf("with %d neighbors: \n", jnum);
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      printf("atom firstneigh[i][%d] with id %d\n", jj, j);
      printf("x[j][0:3]: %f %f %f\n", atom->x[j][0], atom->x[j][1], atom->x[j][2]);
    }
    printf("---------------\n");
  }
  */
  
  /*
  for(i=0; i<ntotal; i++){
    printf("atom %d of type %d at [%f, %f, %f]\n", i, type[i], atom->x[i][0], atom->x[i][1], atom->x[i][2]);
  }
  */

  // END DEBUGGING STUFF




  // compute and update forces and pa energies if needed
  /*
  

  // update forces
  for(i=0; i<nlocal; i++){
    atom->f[i][0] += f[i][0];
    atom->f[i][1] += f[i][1];
    atom->f[i][2] += f[i][2];
  }

  // update global energy or virial if needed
  */
  /*for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    printf("#############################################\n");
    printf("neighbors of owned atom %d with id %d \n", ii, i);
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    printf("at position x[i][0:3]: %f %f %f\n", xtmp, ytmp, ztmp);
    itype = type[i];
    //printf("itype = type[i]: %d\n", itype);
    jlist = firstneigh[i];
    jnum = numneigh[i];
    natoms += jnum;
    printf("num neighbors: %d\n", jnum);
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      printf("atom firstneigh[i][%d] with id %d\n", jj, j);
      printf("x[j][0:3]: %f %f %f\n", x[j][0], x[j][1], x[j][2]);
    }
  }*/
  /*
  printf("-- eflag: %d, vflag: %d --\n", eflag, vflag);
  printf("-- evflag: %d --\n", evflag);
  printf("-- eflag_global: %d, eflag_atom: %d --\n", eflag_global, eflag_atom);
  printf("-- vflag_global: %d, vflag_atom: %d --\n", vflag_global, vflag_atom);
  */
}


/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairGNN::allocate()
{
  printf("ALLOCATE\n");
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
  printf("SETTINGS\n");
  gnn = gnn_from_args(narg, arg, &realcut);
  //realcut = cut * 1.5;
  printf("realcut: %f", realcut);
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */
// pair_coeff **arg
void PairGNN::coeff(int narg, char **arg)
{
  printf("COEFF\n");
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
  printf("INIT STYLE\n");
  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 1;
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
  //neighbor->requests[irequest]->newton = 2; // ???
  neighbor->requests[irequest]->ghost = 1;
  neighbor->requests[irequest]->cut = 1;
  neighbor->requests[irequest]->cutoff = realcut;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairGNN::init_one(int i, int j)
{
  printf("INIT ONE\n");
  return realcut;
}