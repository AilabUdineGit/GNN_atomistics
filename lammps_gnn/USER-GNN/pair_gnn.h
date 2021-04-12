/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(gnn,PairGNN)

#else

#ifndef LMP_PAIR_GNN_H
#define LMP_PAIR_GNN_H

#include "pair.h"
#include "lammps_gnn.h"


namespace LAMMPS_NS {

class PairGNN : public Pair {

 public:
  PairGNN(class LAMMPS *);
  virtual ~PairGNN();
  virtual void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  void init_style();
  double init_one(int, int);

 protected:
  GNN* gnn;

  virtual void allocate();
};

}

#endif
#endif