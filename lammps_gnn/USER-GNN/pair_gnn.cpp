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

#include "pair_gnn.h"
#include "lammps_gnn.h"

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


using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

PairGNN::PairGNN(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
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
  printf("-- COMPUTE --\n");
  double gnn_energy = gnn->compute_energy(atom->nlocal, atom->x);
  eng_vdwl= gnn_energy;
  /*
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double rsq,r2inv,r6inv,forcelj,factor_lj;
  int *ilist,*jlist,*numneigh,**firstneigh;
  
  evdwl = 0.0; // ?
  ev_init(eflag,vflag); // ?

  printf("-- eflag: %d, vflag: %d --\n", eflag, vflag);
  // eflag: energy?
  // vflag: virial?


  double **x = atom->x; // posizioni 
  double **f = atom->f; // forze

  int *type = atom->type; // tipi
  int nlocal = atom->nlocal; // ?
  double *special_lj = force->special_lj; //?
  int newton_pair = force->newton_pair; //?

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;


  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    //printf("-- Positions: %f %f %f --\n", xtmp, ytmp, ztmp);
  }

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0]; // pos x, y, z
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i]; // tipo
    jlist = firstneigh[i]; // neighbors
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) { // loop over neighbors 
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)]; 
      j &= NEIGHMASK;

      delx = xtmp - x[j][0]; // distance x, y, z
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz; // distance^2
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) { // distance^2 < cutoff^2
        r2inv = 1.0/rsq; // 1/d**2
        r6inv = r2inv*r2inv*r2inv; // 1/d**6
        forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]); // (sigma1)/d**12 - (sigma2)/d**6  
        fpair = factor_lj*forcelj*r2inv; // epsilon * ((sigma1)/d**12 - (sigma2)/d**6) * 1/d**2

        f[i][0] += delx*fpair;

        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }

        if (eflag) {
          evdwl = r6inv*(lj3[itype][jtype]*r6inv-lj4[itype][jtype]) -
            offset[itype][jtype];
          evdwl *= factor_lj;
          printf("-- wvdwl %f --\n", evdwl);
        }

        if (evflag) ev_tally(i,j,nlocal,newton_pair,
                             evdwl,0.0,fpair,delx,dely,delz);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    printf("--- i: %d, fx fy fz %f %f %f ---\n", i, f[i][0], f[i][1], f[i][2]);
  }
  */
}


/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairGNN::allocate()
{
  printf("-- ALLOCATE --\n");
  allocated = 1;

  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 1;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  /**
  int n = atom->ntypes; // types of atoms
  printf("-- n: %d --", n)
  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  memory->create(epsilon,n+1,n+1,"pair:epsilon");
  memory->create(sigma,n+1,n+1,"pair:sigma");
  memory->create(lj1,n+1,n+1,"pair:lj1");
  memory->create(lj2,n+1,n+1,"pair:lj2");
  memory->create(lj3,n+1,n+1,"pair:lj3");
  memory->create(lj4,n+1,n+1,"pair:lj4");
  memory->create(offset,n+1,n+1,"pair:offset");
  **/
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */
// pair_style lorenzo **arg
void PairGNN::settings(int narg, char **arg)
{
  printf("-- SETTINGS --\n");

  gnn = gnn_from_args(narg, arg);
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */
// pair_coeff **arg
void PairGNN::coeff(int narg, char **arg)
{
  printf("-- COEFF --\n");

  if (narg != 3)
    error->all(FLERR, "Incorrect args for pair coefficients");

  if ((strcmp(arg[0], "*") && strcmp(arg[1], "*"))) 
    error->all(FLERR, "Incorrect args for pair coefficients\nYou should use \"pair_coeff * * <filename>\"");

  if (!allocated) allocate();
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairGNN::init_style()
{
  printf("-- INIT STYLE\n");

  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairGNN::init_one(int i, int j)
{
  printf("-- INIT ONE (%d, %d) \n", i, j);
  return cutsq[0][0];
}