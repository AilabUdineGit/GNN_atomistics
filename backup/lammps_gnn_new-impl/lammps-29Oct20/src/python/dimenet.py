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


qm9_target_dict = {
    0: 'mu',
    1: 'alpha',
    2: 'homo',
    3: 'lumo',
    5: 'r2',
    6: 'zpve',
    7: 'U0',
    8: 'U',
    9: 'H',
    10: 'G',
    11: 'Cv',
}


device = DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float64



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


class DimeNet2(DimeNet):

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

  def triplets(self, edge_index, num_nodes):
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
    mask = idx_i != idx_k  # Remove i == k triplets.
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

    # Edge indices (k-j, j->i) for triplets.
    idx_kj = adj_t_row.storage.value()[mask]
    idx_ji = adj_t_row.storage.row()[mask]

    return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji

  def forward(self, z, pos, batch=None, pa=False):
    
    edge_index = radius_graph(pos, r=self.cutoff, batch=batch)

    i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(edge_index, num_nodes=z.size(0))

    # Calculate distances.
    dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
      
    # Define atoms position 
    pos_i = pos[idx_i]
    pos_j = pos[idx_j] # central atom
    pos_k = pos[idx_k]

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
        P += output_block(x, rbf, i, num_nodes=pos.size(0))

    # Energy de-standardization
    if self.std is not None and self.mean is not None:
      P = P * self.std + self.mean

    res = P.sum(dim=0) if batch is None else scatter(P, batch, dim=0)
        
    if pa:
      return res, P
    return res

  @staticmethod
  def new_from_lammps(cutoff, hidden_channels, out_channels, 
    num_blocks, num_bilinear, num_spherical, num_radial, mean, std):

    dimenet = DimeNet2(cutoff=cutoff, 
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

  def compute_from_lammps(self, z, x, idx_local):
    
    data_object = ppro.data_object_from_arrays(z, x)
    
    loader = DataLoader([data_object], batch_size=1, shuffle=False)
    for data in loader:
      data.x.requires_grad = True 
      et, pat = self.forward(data.z, data.x, data.batch, pa=True)
      ft = -1 * torch.autograd.grad(et, data.x, grad_outputs=torch.ones_like(et), create_graph=False, retain_graph=True)[0]
    idx_ghost = torch.tensor(sorted(list( set(list(range(len(z)))) - set(idx_local) )), dtype=torch.long)
    print(f"idx_local: {idx_local}")
    print(f"idx_ghost: {idx_ghost}")
    print(f"Ghost:\n{data.x[idx_ghost]}\n{ft[idx_ghost]}")
    e = pat[idx_local].sum().item()
    f = ft[idx_local].tolist()
    return e, f

  def compute_pa_from_lammps(self, z, x, idx_local):

    data_object = ppro.data_object_from_arrays(z, x)

    loader = DataLoader([data_object], batch_size=1, shuffle=False)
    for data in loader:
      data.x.requires_grad = True 
      et, pat = self.forward(data.z, data.x, data.batch, pa=True)
      ft = -1 * torch.autograd.grad(et, data.x, grad_outputs=torch.ones_like(et), create_graph=True, retain_graph=True)[0]
    e = pat[idx_local].sum().item()
    e_pa = pat[idx_local].tolist()
    f = ft[idx_local].tolist()
    return e, f, e_pa
