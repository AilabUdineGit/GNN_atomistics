
from math import pi as PI

import ase
from ase import Atoms
from ase.neighborlist import neighbor_list 
import torch
import torch.nn.functional as F
from torch.nn import Embedding, Sequential, Linear, ModuleList
import numpy as np

from torch_scatter import scatter
from torch_geometric.nn import radius_graph, MessagePassing


import ppro
from torch_geometric.data import DataLoader


# TODO: move this somewhere else
device = DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float64


def pbc_edges(cutoff, z, pos, cell, batch, pbcs=True):
  if cell is None:
    return

  tmp_z = z.cpu()
  tmp_pos = pos.detach().cpu().numpy()
  tmp_cell = cell.cpu()
  nh1_tmp = np.array([]) # will contain all connection from node i
  nh2_tmp = np.array([]) # .. to node j
  D = torch.tensor([], dtype=DTYPE, device=DEVICE)
  #Do = torch.tensor([], dtype=DTYPE, device=device)
  if batch is not None: #batch input
    tmp_batch = np.array(batch.cpu())
    batch_size = []
    found_b = []
    for b in tmp_batch: # create an array with each element being the dim of the corresponding index batch
      if b not in found_b:
        found_b.append(b)
        batch_size.append((tmp_batch == b).sum())

    for i in range(len(batch_size)): # 
      prev_sum = sum(batch_size[:i])
      current_z = tmp_z[prev_sum:batch_size[i]+prev_sum]
      #print(tmp_cell[3*i:3*(i+1)])
      atms = Atoms(charges=current_z, positions=tmp_pos[prev_sum:batch_size[i]+prev_sum], cell=tmp_cell[3*i:3*(i+1)], pbc=pbcs) # create the atomic structure
      nh1, nh2, d, S = neighbor_list("ijDS", atms, cutoff, self_interaction=False) # get the connections for the atomic structure
      nh1 = nh1 + prev_sum # adds the number of previous elements to the atom index
      nh2 = nh2 + prev_sum
      S = torch.tensor(S, dtype=DTYPE, device=DEVICE)
      #d = torch.tensor(d, dtype=DTYPE, device=device)
      #Do = torch.cat((Do, d), 0)
      d = pos[nh2] - pos[nh1] + torch.matmul(S, cell[3*i:3*(i+1)])
      D = torch.cat((D, d), 0)
      nh1_tmp = np.concatenate((nh1_tmp, np.array(nh1)))
      nh2_tmp = np.concatenate((nh2_tmp, np.array(nh2)))

  else: #single cell
    atms = Atoms(charges=tmp_z, positions=tmp_pos, cell=tmp_cell) # create the atomic structure
    nh1, nh2 = neighbor_list("ij", atms, self.cutoff, self_interaction=False) # get the connections for the atomic structure
    nh1_tmp = np.concatenate((nh1_tmp, np.array(nh1)))
    nh2_tmp = np.concatenate((nh2_tmp, np.array(nh2)))

  D = D.norm(dim=-1)
  #Do = Do.norm(dim=-1)
  #assert torch.all(torch.eq(D, Do)).item()
  return [torch.tensor(nh1_tmp, dtype = torch.long).to(device), torch.tensor(nh2_tmp, dtype = torch.long).to(device)], D



class InteractionBlock(torch.nn.Module):
  def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff):
    super(InteractionBlock, self).__init__()
    self.mlp = Sequential(
        Linear(num_gaussians, num_filters),
        ShiftedSoftplus(),
        Linear(num_filters, num_filters),
    )
    self.conv = CFConv(hidden_channels, hidden_channels, num_filters,
                       self.mlp, cutoff)
    self.act = ShiftedSoftplus()
    self.lin = Linear(hidden_channels, hidden_channels)

    self.reset_parameters()

  def reset_parameters(self):
    torch.nn.init.xavier_uniform_(self.mlp[0].weight)
    self.mlp[0].bias.data.fill_(0)
    torch.nn.init.xavier_uniform_(self.mlp[2].weight)
    self.mlp[0].bias.data.fill_(0)
    self.conv.reset_parameters()
    torch.nn.init.xavier_uniform_(self.lin.weight)
    self.lin.bias.data.fill_(0)

  def forward(self, x, edge_index, edge_weight, edge_attr):
    x = self.conv(x, edge_index, edge_weight, edge_attr)
    x = self.act(x)
    x = self.lin(x)
    return x


class CFConv(MessagePassing):
  def __init__(self, in_channels, out_channels, num_filters, nn, cutoff):
    super(CFConv, self).__init__(aggr='add')
    self.lin1 = Linear(in_channels, num_filters, bias=False)
    self.lin2 = Linear(num_filters, out_channels)
    self.nn = nn
    self.cutoff = cutoff

    self.reset_parameters()

  def reset_parameters(self):
    torch.nn.init.xavier_uniform_(self.lin1.weight)
    torch.nn.init.xavier_uniform_(self.lin2.weight)
    self.lin2.bias.data.fill_(0)

  def forward(self, x, edge_index, edge_weight, edge_attr):
    C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
    W = self.nn(edge_attr) * C.view(-1, 1)

    x = self.lin1(x)
    x = self.propagate(edge_index, x=x, W=W)
    x = self.lin2(x)
    return x

  def message(self, x_j, W):
    return x_j * W


class GaussianSmearing(torch.nn.Module):
  def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
    super(GaussianSmearing, self).__init__()
    offset = torch.linspace(start, stop, num_gaussians)
    self.coeff = (-0.5 / (offset[1] - offset[0]).item()**2)
    self.register_buffer('offset', offset)

  def forward(self, dist):
    dist = dist.view(-1, 1) - self.offset.view(1, -1)
    res = torch.exp(self.coeff * torch.pow(dist, 2))
    return res

class ShiftedSoftplus(torch.nn.Module):
  def __init__(self):
    super(ShiftedSoftplus, self).__init__()
    self.shift = torch.log(torch.tensor(2.0)).item()

  def forward(self, x):
    return F.softplus(x) - self.shift


class SchNet(torch.nn.Module):

  def __init__(self, hidden_channels=128, num_filters=128,
               num_interactions=6, num_gaussians=50, cutoff=10.0,
               readout='add', dipole=False, mean=None, std=None,
               atomref=None, pbc=False):
    super(SchNet, self).__init__()

    assert readout in ['add', 'sum', 'mean']

    self.hidden_channels = hidden_channels
    self.num_filters = num_filters
    self.num_interactions = num_interactions
    self.num_gaussians = num_gaussians
    self.cutoff = cutoff
    self.readout = readout
    self.dipole = dipole
    self.readout = 'add' if self.dipole else self.readout
    self.mean = mean
    self.std = std
    self.scale = None

    atomic_mass = torch.from_numpy(ase.data.atomic_masses)
    self.register_buffer('atomic_mass', atomic_mass)

    self.embedding = Embedding(100, hidden_channels)
    self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

    self.interactions = ModuleList()
    for _ in range(num_interactions):
        block = InteractionBlock(hidden_channels, num_gaussians,
                                 num_filters, cutoff)
        self.interactions.append(block)

    self.lin1 = Linear(hidden_channels, hidden_channels // 2)
    self.act = ShiftedSoftplus()
    self.lin2 = Linear(hidden_channels // 2, 1)

    self.register_buffer('initial_atomref', atomref)
    self.atomref = None
    if atomref is not None:
        self.atomref = Embedding(100, 1)
        self.atomref.weight.data.copy_(atomref)

    self.pbc = pbc
    
    self.reset_parameters()

  def reset_parameters(self):
    self.embedding.reset_parameters()
    for interaction in self.interactions:
        interaction.reset_parameters()
    torch.nn.init.xavier_uniform_(self.lin1.weight)
    self.lin1.bias.data.fill_(0)
    torch.nn.init.xavier_uniform_(self.lin2.weight)
    self.lin2.bias.data.fill_(0)
    if self.atomref is not None:
        self.atomref.weight.data.copy_(self.initial_atomref)


  def forward(self, z, pos, cell, batch=None, pa=False):
    assert z.dim() == 1 and z.dtype == torch.long
    batch = torch.zeros_like(z) if batch is None else batch

    h = self.embedding(z)

    if self.pbc:
      edge_index, d = pbc_edges(self.cutoff, z, pos, cell, batch)
      edge_index = torch.stack(edge_index)
      row, col = edge_index
      edge_weight = d.to(dtype=DTYPE)
    else:
      edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
      row, col = edge_index
      edge_weight = (pos[row] - pos[col]).norm(dim=-1)
    
    edge_attr = self.distance_expansion(edge_weight)

    for interaction in self.interactions:
        h = h + interaction(h, edge_index, edge_weight, edge_attr)

    h = self.lin1(h)
    h = self.act(h)
    h = self.lin2(h)

    if self.dipole:
        mass = self.atomic_mass[z].view(-1, 1)
        c = scatter(mass * pos, batch, dim=0) / scatter(mass, batch, dim=0)
        h = h * (pos - c[batch])

    if not self.dipole and self.mean is not None and self.std is not None:
        h = h * self.std + self.mean

    if not self.dipole and self.atomref is not None:
        h = h + self.atomref(z)

    out = scatter(h, batch, dim=0, reduce=self.readout)

    if self.dipole:
        out = torch.norm(out, dim=-1, keepdim=True)

    if self.scale is not None:
        out = self.scale * out

    if pa:
      return out, h
    return out


  def __repr__(self):
    return (f'{self.__class__.__name__}('
            f'hidden_channels={self.hidden_channels}, '
            f'num_filters={self.num_filters}, '
            f'num_interactions={self.num_interactions}, '
            f'num_gaussians={self.num_gaussians}, '
            f'cutoff={self.cutoff})')



  @staticmethod
  def new_from_lammps(cutoff, num_hidden, num_filters, num_interactions,
    num_gaussians, mean, std):

    schnet = SchNet(cutoff=cutoff, 
      hidden_channels=num_hidden,
      num_interactions=num_interactions,
      num_gaussians=num_gaussians,
      mean=mean,
      std=std,
      pbc=False) 
    schnet.eval()
    if DTYPE == torch.float64: # TODO: change this
      schnet = schnet.double()
    return schnet

  def load_pretrained(self, file_path):
    print(f"[lammps_gnn] loading pretrained model from {file_path}")
    state = torch.load(file_path, map_location=DEVICE)
    self.load_state_dict(state)

  def compute_from_lammps(self, z, x, idx_local):
    
    data_object = ppro.data_object_from_arrays(z, x)

    loader = DataLoader([data_object], batch_size=1, shuffle=False)
    for data in loader:
      data.x.requires_grad = True 
      et, pat = self.forward(data.z, data.x, data.batch, pa=True)
      ft = -1 * torch.autograd.grad(et, data.x, grad_outputs=torch.ones_like(et), create_graph=True, retain_graph=True)[0]
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
