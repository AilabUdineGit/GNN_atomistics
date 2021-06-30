
import ase
import torch
from torch_geometric.data import Data


# TODO: move this somewhere else
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float64

def data_object(atoms: ase.Atoms, z=None):
  
  n = atoms.get_global_number_of_atoms()

  #cell = torch.tensor(atoms.cell, dtype=DTYPE).to(DEVICE)
  cell = 0.0  
  positions = torch.tensor(atoms.get_positions(), dtype=DTYPE).to(DEVICE)
  
  if z == None:
    charges = [26] * len(positions)
  else:
    charges = z
  charges = torch.tensor(charges, dtype=torch.long).to(DEVICE)

  if "energy" in atoms.info:
    y = torch.tensor(atoms.info["energy"], dtype=DTYPE).to(DEVICE)
  else:
    y = torch.tensor(0.0, dtype=DTYPE).to(DEVICE)
  #f = torch.tensor(atoms.get_array("force", copy=True), dtype=DTYPE).to(device)
  f = None

  return Data(charges=charges, x=positions, y=y, cell=cell, n=n, f=f)

def data_object_from_arrays(z, x):

  z = torch.tensor(z, dtype=torch.long, device=DEVICE)
  x = torch.tensor(x, dtype=DTYPE, device=DEVICE)

  return Data(z=z, x=x, y=0.0)