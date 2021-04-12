

import torch
import random
import math
import torch_geometric


# dummy to be replaced
class SchNet():

    def __init__(self, cutoff, hidden_channels, num_filters, 
        num_interactions, num_gaussians, mean, std):
        
        print(f"{cutoff} {hidden_channels} {num_filters} {num_interactions}", end="")
        print(f"{num_gaussians} {mean} {std}")
        alive = True

    def compute_energy(self, z, x, cell):
        print("Computing energy...")
        print(f"z = {z}")
        for i, p in enumerate(x):
            print(f"pos[{i}] = {p}")
        print(f"cell = {cell}")
        return 0.0


def main():
	pass



if __name__ == "__main__":
	main()
