#@title  { vertical-output: true, form-width: "25%" }
from torch_geometric.nn import DimeNet
from torch_geometric.nn.acts import swish
from math import sqrt, pi as PI

import numpy as np
import torch
from torch.nn import Linear, Embedding
from torch_scatter import scatter
from torch_sparse import SparseTensor
from torch_geometric.nn import radius_graph
from torch_geometric.data import download_url
from torch_geometric.data.makedirs import makedirs

from torch_geometric.nn.models.dimenet import Envelope
from torch_geometric.nn.models.dimenet import BesselBasisLayer
from torch_geometric.nn.models.dimenet import SphericalBasisLayer
from torch_geometric.nn.models.dimenet import ResidualLayer
from torch_geometric.nn.models.dimenet import InteractionBlock
from torch_geometric.nn.models.dimenet import OutputBlock

from torch_geometric.nn.models.dimenet_utils import bessel_basis, real_sph_harm
import ase
from ase.neighborlist import neighbor_list 
from ase import Atoms

import ppro
from torch_geometric.data import DataLoader

try:
    import sympy as sym
except ImportError:
    sym = None

import os
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
except ImportError:
    tf = None

# TODO: move this somewhere else
device = DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float64


def pbc_edges(cutoff, z, x, cell, batch, compute_sc=False):

  NH1 = torch.tensor([], dtype=torch.long, device=DEVICE)
  NH2 = torch.tensor([], dtype=torch.long, device=DEVICE)
  S = torch.tensor([], dtype=torch.long, device=DEVICE)
  D = torch.tensor([], dtype=DTYPE, device=DEVICE)
  SC = torch.tensor([], dtype=DTYPE, device=DEVICE) if compute_sc else None

  if batch is not None:
    # count number of elements for each batch
    batch_ids = list(set(batch.cpu().tolist()))
    batch_sizes = [ (batch == id).sum().item() for id in batch_ids ]

    for i in range(len(batch_sizes)):
      offset = sum(batch_sizes[:i]) # to obtain correct atom indices
      
      atoms = Atoms(charges = (z[offset:offset + batch_sizes[i]]).cpu(), 
        positions = x[offset:offset + batch_sizes[i]].detach().numpy(), 
        cell = (cell[3*i:3*(i+1)]).cpu(),
        pbc=True
      ) 
      
      nh1, nh2, s = neighbor_list("ijS", atoms, cutoff, self_interaction=False) 
      nh1 = torch.tensor(nh1, dtype=torch.long, device=DEVICE)
      nh2 = torch.tensor(nh2, dtype=torch.long, device=DEVICE)
      nh1 = nh1 + offset
      nh2 = nh2 + offset
      s = torch.tensor(s, dtype=DTYPE, device=DEVICE)
      d = x[nh2] - x[nh1] + torch.matmul(s, cell[3*i:3*(i+1)])
      
      if compute_sc:
        cell_flat = torch.flatten(cell[3*i:3*(i+1)])
        sc = torch.tile(cell_flat, (len(d), 1))
        sc[:, 0:3] = (sc[:, 0:3].T * s[:, 0]).T
        sc[:, 3:6] = (sc[:, 3:6].T * s[:, 1]).T
        sc[:, 6:9] = (sc[:, 6:9].T * s[:, 2]).T
        SC = torch.cat((SC, sc), 0)      

      NH1 = torch.cat((NH1, nh1), 0)
      NH2 = torch.cat((NH2, nh2), 0)
      S = torch.cat((S, s), 0)
      D = torch.cat((D, d), 0)

  else: # no batch
    atoms = Atoms(charges = z.cpu(), positions = x.detach().numpy(), cell = cell.cpu(), pbc=True)
    nh1, nh2, s = neighbor_list("ijS", atoms, cutoff, self_interaction=False)
    nh1 = torch.tensor(nh1, dtype=torch.long, device=DEVICE)
    nh2 = torch.tensor(nh2, dtype=torch.long, device=DEVICE)
    s = torch.tensor(s, dtype=DTYPE, device=DEVICE)
    d = x[nh2] - x[nh1] + torch.matmul(s, cell)
    
    if compute_sc:
      cell_flat = torch.flatten(cell)
      sc = torch.tile(cell_flat, (len(d), 1))
      sc[:, 0:3] = (sc[:, 0:3].T * s[:, 0]).T
      sc[:, 3:6] = (sc[:, 3:6].T * s[:, 1]).T
      sc[:, 6:9] = (sc[:, 6:9].T * s[:, 2]).T  
      SC = sc

    NH1, NH2, S, D = nh1, nh2, s, d

  D = D.norm(dim=-1)
  return  NH1, NH2, D, S, SC 

class EmbeddingBlock(torch.nn.Module):
  def __init__(self, num_radial, hidden_channels, act=swish):
    super(EmbeddingBlock, self).__init__()
    self.act = act

    self.emb = Embedding(95, hidden_channels)
    self.lin_rbf = Linear(num_radial, hidden_channels, bias=False)
    self.lin = Linear(3 * hidden_channels, hidden_channels)

    self.reset_parameters()

  def reset_parameters(self):
    self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
    self.lin_rbf.reset_parameters()
    self.lin.reset_parameters()

  def forward(self, x, rbf, i, j):
    x = self.emb(x)
    #rbf = self.act(self.lin_rbf(rbf)) # FIX: this should not have an activation function
    rbf = self.lin_rbf(rbf)
    return self.act(self.lin(torch.cat([x[i], x[j], rbf], dim=-1)))



class DimeNet(torch.nn.Module):
  
  def __init__(self, hidden_channels, out_channels, num_blocks, num_bilinear,
                 num_spherical, num_radial, cutoff=5.0, envelope_exponent=5,
                 num_before_skip=1, num_after_skip=2, num_output_layers=3,
                 act=swish, mean=None, std=None):
    super(DimeNet, self).__init__()

    self.cutoff = cutoff

    #set mean and standard deviation of energies
    self.mean = mean 
    self.std = std

    # padding used for PBCs
    self.padding = torch.nn.ConstantPad2d((0,6,0,0), 0)

    if sym is None:
        raise ImportError('Package `sympy` could not be found.')

    self.num_blocks = num_blocks

    self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
    self.sbf = SphericalBasisLayer(num_spherical, num_radial, cutoff,
                                    envelope_exponent)

    self.emb = EmbeddingBlock(num_radial, hidden_channels, act)

    self.output_blocks = torch.nn.ModuleList([
        OutputBlock(num_radial, hidden_channels, out_channels,
                    num_output_layers, act) for _ in range(num_blocks + 1)
    ])

    self.interaction_blocks = torch.nn.ModuleList([
        InteractionBlock(hidden_channels, num_bilinear, num_spherical,
                          num_radial, num_before_skip, num_after_skip, act)
        for _ in range(num_blocks)
    ])

    self.reset_parameters()

  def reset_parameters(self):
    self.rbf.reset_parameters()
    self.emb.reset_parameters()
    for out in self.output_blocks:
      out.reset_parameters()
    for interaction in self.interaction_blocks:
      interaction.reset_parameters()

  def triplets_original(self, edge_index, num_nodes):
    row, col = edge_index  # j->i

    value = torch.arange(row.size(0), device=row.device)
    adj_t = SparseTensor(row=col, col=row, value=value,
                         sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t[row]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices (k->j->i) for triplets.
    idx_i = col.repeat_interleave(num_triplets)
    idx_j = row.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
    mask = (idx_i != idx_k)  # Remove i == k triplets.
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

    # Edge indices (k-j, j->i) for triplets.
    idx_kj = adj_t_row.storage.value()[mask]
    idx_ji = adj_t_row.storage.row()[mask]

    return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji


  def triplets(self, edge_index, num_nodes, shift_cells=None, shift=None):
    row, col = edge_index  # j->i

    value = torch.arange(row.size(0), device=row.device)
    adj_t = SparseTensor(row=col, col=row, value=value,
                         sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t[row]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    idx_i = col.repeat_interleave(num_triplets)
    idx_j = row.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()

    if shift_cells is not None: # Update also the shift vectors
      shift_cells_i = shift_cells.repeat_interleave(num_triplets, dim=0)
      shift_i = shift.repeat_interleave(num_triplets, dim=0)
      shift_cells_k = -shift_cells[adj_t_row.storage.value()]
      shift_k = -shift[adj_t_row.storage.value()]

    mask = torch.all((torch.cat((torch.unsqueeze(idx_i, 1), shift_i), dim=1) ==\
                      torch.cat((torch.unsqueeze(idx_k, 1), shift_k), dim=1)), dim=1)

    idx_i, idx_j, idx_k = idx_i[~mask], idx_j[~mask], idx_k[~mask]
    if shift_cells is not None: # Remove also from the shift vector
      shift_cells_i = shift_cells_i[~mask]
      shift_cells_k = shift_cells_k[~mask]
      shift_i = shift_i[~mask]
      shift_k = shift_k[~mask]

    idx_kj = adj_t_row.storage.value()[~mask]
    idx_ji = adj_t_row.storage.row()[~mask]

    return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji, shift_cells_i, shift_i, shift_cells_k, shift_k


  def forward(self, z, pos, cell=None, batch=None):
        
    edge_index = []
    dist = []
    shift_cells = None
    if cell is not None: # implement PBC
      r1, r2, dist, shift, shift_cells = pbc_edges(self.cutoff, z, pos, cell, batch, compute_sc=True) 
      edge_index = [r1, r2]

        
      i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji, shift_cells_i, shift_i, shift_cells_k, shift_k = self.triplets(
          edge_index, num_nodes=z.size(0), shift_cells=shift_cells, shift=shift)        
    else: # old method without PBC
      edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
      i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets_original(
          edge_index, num_nodes=z.size(0))        
      dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

    # Define atoms position 
    pos_i = pos[idx_i]
    pos_j = pos[idx_j]
    pos_k = pos[idx_k]

    if cell is not None: # Fix coordinates for PBCs

      pos_i = pos_i + shift_cells_i[:, 0:3] + shift_cells_i[:, 3:6] + shift_cells_i[:, 6:9]
      pos_k = pos_k + shift_cells_k[:, 0:3] + shift_cells_k[:, 3:6] + shift_cells_k[:, 6:9]
      sc_ij = torch.all(~torch.all(pos_i == pos_j, dim=1)) 
      sc_kj = torch.all(~torch.all(pos_k == pos_j, dim=1))
      #if not (sc_ij and sc_kj):
      #   raise NameError('Found same position for different atoms!')

    # Calculate angles - with some Fixes to indexes compared to the orig. version
    pos_ji, pos_kj = pos_j - pos_i, pos_k - pos_j
    a = (pos_ji * pos_kj).sum(dim=-1)
    b = torch.cross(pos_ji, pos_kj).norm(dim=-1) 
    angle = torch.atan2(b, a)      

    rbf = self.rbf(dist)
    sbf = self.sbf(dist, angle, idx_kj)

    # Embedding block.
    x = self.emb(z, rbf, i, j)
    P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))

    # Interaction blocks.
    for interaction_block, output_block in zip(self.interaction_blocks,
                                               self.output_blocks[1:]):
      x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
      P += output_block(x, rbf, i)

    # Energy de-standardization
    if self.std is not None and self.mean is not None:
      P = P * self.std + self.mean

    res = P.sum(dim=0) if batch is None else scatter(P, batch, dim=0)
    return res

  @staticmethod
  def new_from_lammps(cutoff, hidden_channels, out_channels, 
    num_blocks, num_bilinear, num_spherical, num_radial, mean, std):

    dimenet = DimeNet(cutoff=cutoff, 
      hidden_channels=hidden_channels,
      out_channels=out_channels,
      num_blocks=num_blocks,
      num_bilinear=num_bilinear,
      num_spherical=num_spherical,
      num_radial=num_radial,
      mean=mean,
      std=std) 
    dimenet.eval()
    if DTYPE == torch.float64: # TODO: change this
      dimenet = dimenet.double()
    return dimenet

  def load_pretrained(self, file_path):
    state = torch.load(file_path, map_location=DEVICE)
    self.load_state_dict(state)

  def compute_energy_from_lammps(self, z, x, cell):
    
    atoms = ase.Atoms(symbols="Fe"*len(x), 
      positions=x,
      cell=cell,
      pbc=True
    )
    data_object = ppro.data_object(atoms, z=z)
    loader = DataLoader([data_object], batch_size=1, shuffle=False)
    for data in loader:
      data.x.requires_grad = True
      print("$") 
      print("delete_this version")
      print(f"{data.cell.tolist()}")
      print(f"{data.x.tolist()}")
      et = self.forward(data.charges, data.x, cell=data.cell, batch=data.batch)
      print(f"{et.item()}")
      ft = -1 * torch.autograd.grad(et, data.x, grad_outputs=torch.ones_like(et), create_graph=True, retain_graph=True)[0]
      print(f"{ft.tolist()}")
      print("$") 
    e = et.item()
    f = ft.tolist()
    with open("f_gnn.txt", "a") as myfile:
      c = data_object.cell.tolist()
      myfile.write(f"{c[0][0]*c[1][1]*c[2][2]/2} {f}\n")
    return e, f
