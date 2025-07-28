# %% [markdown]
# ## Predicting absorption coefficient with GNNOptic from crystal structures

# %% [markdown]
# ### Getting started

# %%
# model
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import torch_scatter
import e3nn
from e3nn import o3
from typing import Dict, Union
import os


# crystal structure data
from ase import Atom, Atoms
from ase.neighborlist import neighbor_list
from ase.visualize.plot import plot_atoms
palette = ['#43AA8B', '#F8961E', '#F94144']
sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

# data pre-processing and visualization
import numpy as np
import  matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
from scipy.interpolate import interp1d
import random

# utilities
import time
from mendeleev import element
from tqdm import tqdm
from utils.utils_data import (load_data, train_valid_test_split, plot_example, plot_predictions, weighted_mean, r2_score)
from utils.utils_model import Network, visualize_layers, train
from utils.new_loss import New_Loss
from utils.utils_model import evaluate

bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
default_dtype = torch.float64
torch.set_default_dtype(default_dtype)

# Create a colormap based on the number of unique symbols
datasets = ['g', 'y', 'r']
colors = dict(zip(datasets, palette))
cmap = mpl.colors.LinearSegmentedColormap.from_list('cmap', [palette[k] for k in [0,2,1]])

# Check device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print('torch device:' , device)

# %% [markdown]
# ### Data provenance
# We train our model using the database of High-throughput optical absorption spectra for inorganic semiconductors [[Yang et al. 2018]](https://arxiv.org/pdf/2209.02918.pdf) from the Materials Project. Independent-particle approximation (IPA) and containing 944 crystalline solids with number of atoms in cell < 10.

# %%
data_file = '/home/mariyaah/data1/mpcontribs/ferroelectrics_final_final.pkl'
df , species = load_data(data_file)
df.head()

# %%
len(df)

# %%
from pymatgen.core import Structure
from ase import Atoms
import pandas as pd

def pymatgen_to_ase(structure):
    return Atoms(
        symbols=[site.specie.symbol for site in structure],
        positions=structure.cart_coords.copy(),
        cell=structure.lattice.matrix.copy(),
        pbc=True
    )

df['structure'] = df['structure'].apply(pymatgen_to_ase)


# %% [markdown]
# ### Data structures
# Crystal structures are represented as [ASE](https://wiki.fysik.dtu.dk/ase/ase/atoms.html?highlight=atoms#the-atoms-object) (Atomic Simulation Environment) `Atoms` objects, which store the atomic species and positions of each atom in the unit cell, as well as the lattice vectors of the unit cell.

# %%
# plot an example structure
i = 16 # structure index in dataframe
struct = df.iloc[i]['structure']
symbols = np.unique(list(struct.symbols))
z = dict(zip(symbols, range(len(symbols))))

# Create the plot
fig, ax = plt.subplots(figsize=(6,6))
norm = plt.Normalize(vmin=0, vmax=len(symbols)-1)
color = [mpl.colors.to_hex(k) for k in cmap(norm([z[j] for j in list(struct.symbols)]))]
plot_atoms(struct, ax, radii=0.25, colors=color, rotation=('45x,45y,0z'))

# Add legend with atom labels
legend_elements = []
for symbol, color in zip(set(symbols), [cmap(norm(i)) for i in range(len(set(symbols)))]):
    legend_elements.append(Line2D([0], [0], marker='o', color='w', label=symbol, markerfacecolor=color, markersize=10))
ax.legend(handles=legend_elements, loc='upper right')

# Set labels and title
ax.set_xlabel(r'$x_1\ (\AA)$')
ax.set_ylabel(r'$x_2\ (\AA)$')
ax.set_title(df.iloc[i]['formula'].translate(sub), fontname='DejaVu Sans')

# data statistics (#Atoms/cell)
sites = [len(s.get_positions()) for s in list(df['structure'])]
fig, ax = plt.subplots(figsize=(5,4))
ax.hist(sites, bins=max(sites), fc='#2ab0ff', ec='#e0e0e0', lw=1, width=0.8)
ax.set_xlim(0.0, 11)
x_ticks = np.arange(0, 12, 2) 
ax.set_xticks(x_ticks)
ax.set_ylabel('Number of examples')
ax.set_xlabel('Number of atoms per unit-cell')
fig.patch.set_facecolor('white')
plt.tight_layout()
fig.savefig('data-site.pdf')

# bandgap = np.array(df['bandgap'])
# fig, ax = plt.subplots(figsize=(5,4))
# ax.hist(bandgap, bins=50, fc='#2ab0ff', ec='#e0e0e0')
# ax.set_xlim(-0.5, 5.5)
# x_ticks = np.arange(0, 6, 1) 
# ax.set_xticks(x_ticks)
# ax.set_xlabel('Energy band gap (eV)')
# ax.set_ylabel('Number of examples')
# fig.patch.set_facecolor('white')
# plt.tight_layout()
# #fig.savefig('data-bandgap.pdf')

# lattice parameter statistics
def get_lattice_parameters(data):
    a = []
    len_data = len(data)
    for i in range(len_data):
        d = data.iloc[i]
        a.append(d.structure.cell.cellpar()[:3])
    return np.stack(a)
a = get_lattice_parameters(df)

fig, ax = plt.subplots(figsize=(5,4))
b = 0.
bins = 50
for d, c, n in zip(['a', 'b', 'c'], colors.values(), [a[:,0], a[:,1], a[:,2]]):
    color = [int(c.lstrip('#')[i:i+2], 16)/255. for i in (0,2,4)]
    y, bins, _, = ax.hist(n, bins=bins, fc=color+[0.7], ec=color, bottom=b, label=d)
    b += y
ax.set_xlabel('Lattice parameter ($\AA$)')
ax.set_ylabel('Number of examples')
ax.legend(frameon=False)
plt.tight_layout()
fig.savefig('data-lattice.pdf')
print('average lattice parameter (a/b/c):', a[:,0].mean(), '/', a[:,1].mean(), '/', a[:,2].mean())

# Show the plot
plt.show()

# %% [markdown]
# ### Reconstruction data

# %%
# energy_min = 0.0 #Unit of energy in eV
# energy_max = 50.0 #Unit of energy in eV
# nstep = 251 #Number of the energy points

# new_x = np.linspace(energy_min, energy_max, nstep)
# def interpolate(row, column):
#     interp = interp1d(row['energies'], row[column], kind='linear', fill_value=0, bounds_error=False)
#     new_y = interp(new_x)
#     return new_y

# # Apply the custom function to create a new column
# df['energies_interp'] = df.apply(lambda x: new_x, axis=1)
# df['imag_dielectric_interp'] = df.apply(lambda row: interpolate(row, 'imag_dielectric'), axis=1)
# df['real_dielectric_interp'] = df.apply(lambda row: interpolate(row, 'real_dielectric'), axis=1)
# df['absorption_coefficient_interp'] = df.apply(lambda row: interpolate(row, 'absorption_coefficient'), axis=1)


# Show the plot for compare the original data and interpolated data
# plt.figure(figsize=(6, 7))

# plt.subplot(3, 1, 1)
# plt.plot(df['energies'][i], df['imag_dielectric'][i], label='Original data')
# plt.scatter(df['energies_interp'][i], df['imag_dielectric_interp'][i], s=12, marker ='o', alpha=1, color='C1', label='Interpolated data')
# x_ticks = np.arange(0, 80, 10) 
# plt.xticks(x_ticks)
# plt.xlabel('')
# plt.ylabel(r'Im($\varepsilon$)')
# plt.legend(frameon=False)

# plt.subplot(3, 1, 2)
# plt.plot(df['energies'][i], df['real_dielectric'][i], label='Original data')
# plt.scatter(df['energies_interp'][i], df['real_dielectric_interp'][i], s=12, marker ='o', alpha=1, color='C2', label='Interpolated data')
# x_ticks = np.arange(0, 80, 10) 
# plt.xticks(x_ticks)
# plt.xlabel('')
# plt.ylabel(r'Re($\varepsilon$)')
# plt.legend(frameon=False)

# plt.subplot(3, 1, 3)
# plt.plot(df['energies'][i], df['absorption_coefficient'][i]/1.0E6, label='Original data')
# plt.scatter(df['energies_interp'][i], df['absorption_coefficient_interp'][i]/1.0E6, s=12, marker ='o', alpha=1, color='C3', label='Interpolated data')
# x_ticks = np.arange(0, 80, 10) 
# plt.xticks(x_ticks)
# plt.xlabel('Photon energy (eV)')
# plt.ylabel(r'$\alpha$ ($\times 10^6$ 1/cm)')
# plt.legend(frameon=False)

# plt.tight_layout()
# #plt.savefig('data-interpolation.pdf')
# plt.show()

# %% [markdown]
# ### Feature representation

# %%
# one-hot encoding atom type
type_encoding = {}
specie_mass = []
specie_dipole = []
specie_radius = []



# Additional features
specie_electronegativity = []
specie_electron_affinity = []
specie_ionization_energy = []
specie_atomic_volume = []
specie_nvalence = []
specie_block = []
specie_mendeleev_number = []
specie_vdw_radius = []
specie_atomic_radius = []
specie_atomic_radius_empirical = []
specie_c6 = []
specie_hardness = []
specie_softness = []
specie_thermal_conductivity = []
specie_specific_heat = []
specie_fusion_heat = []
specie_evaporation_heat = []
specie_group = []
specie_period = []


for Z in tqdm(range(1, 119), bar_format=bar_format):              #Change 109 to 119 to increase the number of elements
    specie = Atom(Z)
    type_encoding[specie.symbol] = Z - 1

    e = element(specie.symbol)
    
    # Core features
    Z_mass = specie.mass  # atomic mass (always present)
    Z_dipole = e.dipole_polarizability if e.dipole_polarizability is not None else 67.0
    Z_radius = e.covalent_radius_pyykko if e.covalent_radius_pyykko is not None else 70.0

    # Electronic features
    Z_en = e.en_pauling if e.en_pauling is not None else 1.5
    Z_ea = e.electron_affinity if e.electron_affinity is not None else 0.0
    Z_ie = e.ionenergies.get(1, 5.0) if e.ionenergies else 5.0
    Z_nval = e.nvalence() if e.nvalence is not None else 1
    Z_block = ord(e.block[0]) - ord('s') if e.block is not None else 0
    Z_mend = e.mendeleev_number if e.mendeleev_number is not None else 100

    # Spatial/size features
    Z_vdw = e.vdw_radius if e.vdw_radius is not None else 180.0
    Z_rad_rahm = e.atomic_radius_rahm if e.atomic_radius_rahm is not None else 100.0
    Z_rad_emp = e.atomic_radius if e.atomic_radius is not None else 100.0
    Z_vol = e.atomic_volume if e.atomic_volume is not None else 10.0

    # Bonding and chemical interaction
    Z_c6 = e.c6 if e.c6 is not None else 20.0

    # Thermodynamic features
    Z_tc = e.thermal_conductivity if e.thermal_conductivity is not None else 10.0
    Z_sh = e.specific_heat_capacity if e.specific_heat_capacity is not None else 0.2
    Z_fh = e.fusion_heat if e.fusion_heat is not None else 10.0
    Z_eh = e.evaporation_heat if e.evaporation_heat is not None else 30.0

    # Periodic position
    Z_group = e.group if e.group is not None else 0
    Z_period = e.period if e.period is not None else 0

    # Append all features
    specie_mass.append(Z_mass)
    specie_dipole.append(Z_dipole)
    specie_radius.append(Z_radius)
    specie_electronegativity.append(Z_en)
    specie_electron_affinity.append(Z_ea)
    specie_ionization_energy.append(Z_ie)
    specie_atomic_volume.append(Z_vol)
    specie_nvalence.append(Z_nval)
    specie_block.append(Z_block)
    specie_mendeleev_number.append(Z_mend)
    specie_vdw_radius.append(Z_vdw)
    specie_atomic_radius.append(Z_rad_rahm)
    specie_atomic_radius_empirical.append(Z_rad_emp)
    specie_c6.append(Z_c6)
    specie_thermal_conductivity.append(Z_tc)
    specie_specific_heat.append(Z_sh)
    specie_fusion_heat.append(Z_fh)
    specie_evaporation_heat.append(Z_eh)
    specie_period.append(Z_period)

# Convert all to torch diagonal matrices
type_onehot = torch.eye(len(type_encoding))
mass_onehot = torch.diag(torch.Tensor(specie_mass))
dipole_onehot = torch.diag(torch.Tensor(specie_dipole))
radius_onehot = torch.diag(torch.Tensor(specie_radius))
en_onehot = torch.diag(torch.Tensor(specie_electronegativity))
ea_onehot = torch.diag(torch.Tensor(specie_electron_affinity))
ie_onehot = torch.diag(torch.Tensor(specie_ionization_energy))
vol_onehot = torch.diag(torch.Tensor(specie_atomic_volume))
nvalence_onehot = torch.diag(torch.Tensor(specie_nvalence))
block_onehot = torch.diag(torch.Tensor(specie_block))
mend_number_onehot = torch.diag(torch.Tensor(specie_mendeleev_number))
vdw_radius_onehot = torch.diag(torch.Tensor(specie_vdw_radius))
atomic_radius_onehot = torch.diag(torch.Tensor(specie_atomic_radius))
atomic_radius_empirical_onehot = torch.diag(torch.Tensor(specie_atomic_radius_empirical))
c6_onehot = torch.diag(torch.Tensor(specie_c6))
thermal_conductivity_onehot = torch.diag(torch.Tensor(specie_thermal_conductivity))
specific_heat_onehot = torch.diag(torch.Tensor(specie_specific_heat))
fusion_heat_onehot = torch.diag(torch.Tensor(specie_fusion_heat))
evaporation_heat_onehot = torch.diag(torch.Tensor(specie_evaporation_heat))
period_onehot = torch.diag(torch.Tensor(specie_period))
#print(mass_onehot)

# %%
# Find the scale value
# tmp = np.array([df.iloc[i]['polarization'] for i in range(len(df))])
# scale_data = np.median(np.max(tmp, axis=1))
# print(scale_data)

# build data
# def build_data(entry, type_encoding, type_onehot, r_max=5.):
#     symbols = list(entry.structure.symbols).copy()
#     positions = torch.from_numpy(entry.structure.positions.copy())
#     lattice = torch.from_numpy(entry.structure.cell.array.copy()).unsqueeze(0)

#     # edge_src and edge_dst are the indices of the central and neighboring atom, respectively
#     # edge_shift indicates whether the neighbors are in different images or copies of the unit cell
#     edge_src, edge_dst, edge_shift = neighbor_list("ijS", a=entry.structure, cutoff=r_max, self_interaction=True)
    
#     # compute the relative distances and unit cell shifts from periodic boundaries
#     edge_batch = positions.new_zeros(positions.shape[0], dtype=torch.long)[torch.from_numpy(edge_src)]
#     edge_vec = (positions[torch.from_numpy(edge_dst)]
#                 - positions[torch.from_numpy(edge_src)]
#                 + torch.einsum('ni,nij->nj', torch.tensor(edge_shift, dtype=default_dtype), lattice[edge_batch]))

#     # compute edge lengths (rounded only for plotting purposes)
#     edge_len = np.around(edge_vec.norm(dim=1).numpy(), decimals=2)
    
#     data = tg.data.Data(
#         pos=positions, lattice=lattice, symbol=symbols,
#         x_mass=mass_onehot[[type_encoding[specie] for specie in symbols]],       # atomic mass (node feature)
#         x_dipole=dipole_onehot[[type_encoding[specie] for specie in symbols]],   # atomic dipole polarizability (node feature)
#         x_radius=radius_onehot[[type_encoding[specie] for specie in symbols]],   # atomic covalent radius (node feature)
#         z=type_onehot[[type_encoding[specie] for specie in symbols]],            # atom type (node attribute)
#         edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0),
#         edge_shift=torch.tensor(edge_shift, dtype=default_dtype),
#         edge_vec=edge_vec, edge_len=edge_len,
#         y = np.log10(torch.tensor([entry.polarization], dtype=torch.float64).unsqueeze(0)+1)

#         # y=torch.from_numpy(entry.absorption_coefficient_interp/scale_data).unsqueeze(0)
#     )
    
    # return data

def build_data(entry, type_encoding, type_onehot, r_max=5.):
    symbols = list(entry.structure.symbols).copy()
    positions = torch.from_numpy(entry.structure.positions.copy())
    lattice = torch.from_numpy(entry.structure.cell.array.copy()).unsqueeze(0)

    # Neighbor list (periodic)
    edge_src, edge_dst, edge_shift = neighbor_list("ijS", a=entry.structure, cutoff=r_max, self_interaction=True)

    # Edge vector and length
    edge_batch = positions.new_zeros(positions.shape[0], dtype=torch.long)[torch.from_numpy(edge_src)]
    edge_vec = (positions[torch.from_numpy(edge_dst)]
                - positions[torch.from_numpy(edge_src)]
                + torch.einsum('ni,nij->nj', torch.Tensor(edge_shift), lattice[edge_batch]))
    edge_len = np.around(edge_vec.norm(dim=1).numpy(), decimals=2)

    # Index list based on atom types
    idx = [type_encoding[specie] for specie in symbols]

    data = tg.data.Data(
        pos=positions,
        lattice=lattice,
        symbol=symbols,

        # Core atomic features
        z=type_onehot[idx],
        x_mass=mass_onehot[idx],
        x_dipole=dipole_onehot[idx],
        x_radius=radius_onehot[idx],

        # Extended atomic features
        x_en=en_onehot[idx],
        x_ea=ea_onehot[idx],
        x_ie=ie_onehot[idx],
        x_vol=vol_onehot[idx],
        x_nvalence=nvalence_onehot[idx],
        x_block=block_onehot[idx],
        x_mendeleev=mend_number_onehot[idx],
        x_vdw=vdw_radius_onehot[idx],
        x_atomic_rad=atomic_radius_onehot[idx],
        x_atomic_rad_emp=atomic_radius_empirical_onehot[idx],
        x_c6=c6_onehot[idx],
        x_tc=thermal_conductivity_onehot[idx],
        x_sh=specific_heat_onehot[idx],
        x_fh=fusion_heat_onehot[idx],
        x_eh=evaporation_heat_onehot[idx],
        x_period=period_onehot[idx],

        # Edge features
        edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0),
        edge_shift=torch.tensor(edge_shift),
        edge_vec=edge_vec,
        edge_len=edge_len,

        # Target label (also float32)
        y=torch.log10(torch.Tensor([[entry.polarization]]) + 1)
    )

    return data





r_max = 6. # cutoff radius
df['data'] = df.progress_apply(lambda x: build_data(x, type_encoding, type_onehot, r_max), axis=1)


# %%
#i = 16 # structure index in dataframe
#plt.plot(df['data'][i].y.reshape(-1))
# plot_example(df, i=i, label_edges=True)

# %% [markdown]
# ### Training, validation, and testing datasets
# Split the data into training, validation, and testing datasets with balanced representation of different elements in each set.

# %%
run_time = time.strftime('%y%m%d_%H%M%S', time.localtime())
# train/valid/test split
idx_train, idx_valid = train_valid_test_split(df, species, valid_size=0.1, test_size=0, seed=12, plot=True)
#Save train loss values sets
np.savetxt('./model/idx_train_'+ run_time +'.txt', idx_train, fmt='%i', delimiter='\t')
np.savetxt('./model/idx_valid_'+ run_time +'.txt', idx_valid, fmt='%i', delimiter='\t')
# np.savetxt('./model/idx_test_'+ run_time +'.txt', idx_test, fmt='%i', delimiter='\t')
plt.savefig('data-split.pdf')

# # %%
# print("df length:", len(df))
# print("Max idx_train:", max(idx_train))
# print("Max idx_valid:", max(idx_valid))
# print("Max idx_test:", max(idx_test))
with open(f'model/mariya/df_{run_time}.pickle', 'wb') as f:
    pkl.dump(df, f)


# %% [markdown]
# For use with the trained model provided, the indices of the training, validation, and test sets are loaded below. These indices were generated with a specific seed using the above `train_valid_test_split` function.

# %%
# load train/valid/test indices
with open('./model/idx_train_'+run_time+'.txt', 'r') as f: 
    idx_train = [int(i.split('\n')[0]) for i in f.readlines() if int(i.split('\n')[0])< len(df)]
with open('./model/idx_valid_'+run_time+'.txt', 'r') as f: 
    idx_valid = [int(i.split('\n')[0]) for i in f.readlines() if int(i.split('\n')[0])< len(df)]
# with open('./model/idx_test_'+run_time+'.txt', 'r') as f: 
#     idx_test = [int(i.split('\n')[0]) for i in f.readlines() if int(i.split('\n')[0])< len(df)]

# format dataloaders
batch_size = 1
dataloader_train = tg.loader.DataLoader(df.iloc[idx_train]['data'].values, batch_size=batch_size, shuffle=True)
dataloader_valid = tg.loader.DataLoader(df.iloc[idx_valid]['data'].values, batch_size=batch_size)
# dataloader_test = tg.loader.DataLoader(df.iloc[idx_test]['data'].values, batch_size=batch_size)

# %%
# calculate average number of neighbors
def get_neighbors(df, idx):
    n = []
    for entry in df.iloc[idx].itertuples():
        N = entry.data.pos.shape[0]
        for i in range(N):
            n.append(len((entry.data.edge_index[0] == i).nonzero()))
    return np.array(n)

n_train = get_neighbors(df, idx_train)
n_valid = get_neighbors(df, idx_valid)
# # n_test = get_neighbors(df, idx_test)

# fig, ax = plt.subplots(1,1, figsize=(5,4))
# b = 0.
# bins = 50
# for (d, c), n in zip(colors.items(), [n_train, n_valid, n_test]):
#     color = [int(c.lstrip('#')[i:i+2], 16)/255. for i in (0,2,4)]
#     y, bins, _, = ax.hist(n, bins=bins, fc=color+[0.7], ec=color, bottom=b, label=d)
#     b += y
# ax.set_xlabel('Number of neighbors')
# ax.set_ylabel('Number of examples')
# ax.legend(frameon=False)

# print('average number of neighbors (train/valid/test):', n_train.mean(), '/', n_valid.mean(), '/', n_test.mean())

# %% [markdown]
# ### Network architecture
# We build a model based on the `Network` described in the `e3nn` [Documentation](https://docs.e3nn.org/en/latest/api/nn/models/gate_points_2101.html), modified to incorporate the periodic boundaries we imposed on the crystal graphs. The network applies equivariant convolutions to each atomic node and finally takes an average over all nodes, normalizing the output.

# %%
class MixingLinear(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(MixingLinear, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.weight = nn.Parameter(torch.Tensor(self.out_feature, self.in_feature))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x):
        weight = torch.abs(self.weight)/(torch.sum(torch.abs(self.weight), dim=1, keepdim=True)+1e-10)
        return F.linear(x, weight)

# %%
class PeriodicNetwork(Network):
    def __init__(self, in_dim, em_dim, **kwargs):            
        # override the `reduce_output` keyword to instead perform an averge over atom contributions    
        self.pool = False
        if kwargs['reduce_output'] == True:
            kwargs['reduce_output'] = False
            self.pool = True
            
        super().__init__(**kwargs)

        # embed the one-hot encoding
        self.em_dim = em_dim
        self.em_type = nn.Linear(in_dim, em_dim)
        self.em_mass = nn.Linear(in_dim, em_dim)
        self.em_dipole = nn.Linear(in_dim, em_dim)
        self.em_radius = nn.Linear(in_dim, em_dim)
        self.em_en = nn.Linear(in_dim, em_dim)
        self.em_ea = nn.Linear(in_dim, em_dim)
        self.em_ie = nn.Linear(in_dim, em_dim)
        self.em_vol = nn.Linear(in_dim, em_dim)
        self.em_nvalence = nn.Linear(in_dim, em_dim)
        self.em_block = nn.Linear(in_dim, em_dim)
        self.em_mendeleev = nn.Linear(in_dim, em_dim)
        self.em_vdw = nn.Linear(in_dim, em_dim)
        self.em_atomic_rad = nn.Linear(in_dim, em_dim)
        self.em_atomic_rad_emp = nn.Linear(in_dim, em_dim)
        self.em_c6 = nn.Linear(in_dim, em_dim)
        self.em_tc = nn.Linear(in_dim, em_dim)
        self.em_sh = nn.Linear(in_dim, em_dim)
        self.em_fh = nn.Linear(in_dim, em_dim)
        self.em_eh = nn.Linear(in_dim, em_dim)
        self.em_period = nn.Linear(in_dim, em_dim)
        self.em_mixing = MixingLinear(19, 1)         #Linear layer for mixing the atom features (mass, dipole, radius)

    def forward(self, data: Union[tg.data.Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
    # === Embed individual atomic features ===
        data.z = data.z.to(dtype=self.em_type.weight.dtype)
        data.z = F.relu(self.em_type(data.z))
        data.x_mass = F.relu(self.em_mass(data.x_mass))
        data.x_dipole = F.relu(self.em_dipole(data.x_dipole))
        data.x_radius = F.relu(self.em_radius(data.x_radius))
        data.x_en = F.relu(self.em_en(data.x_en))
        data.x_ea = F.relu(self.em_ea(data.x_ea))
        data.x_ie = F.relu(self.em_ie(data.x_ie))
        data.x_vol = F.relu(self.em_vol(data.x_vol))
        data.x_nvalence = F.relu(self.em_nvalence(data.x_nvalence))
        data.x_block = F.relu(self.em_block(data.x_block))
        data.x_mendeleev = F.relu(self.em_mendeleev(data.x_mendeleev))
        data.x_vdw = F.relu(self.em_vdw(data.x_vdw))
        data.x_atomic_rad = F.relu(self.em_atomic_rad(data.x_atomic_rad))
        data.x_atomic_rad_emp = F.relu(self.em_atomic_rad_emp(data.x_atomic_rad_emp))
        data.x_c6 = F.relu(self.em_c6(data.x_c6))
        data.x_tc = F.relu(self.em_tc(data.x_tc))
        data.x_sh = F.relu(self.em_sh(data.x_sh))
        data.x_fh = F.relu(self.em_fh(data.x_fh))
        data.x_eh = F.relu(self.em_eh(data.x_eh))
        data.x_period = F.relu(self.em_period(data.x_period))

        # === Stack and mix all features ===
        tmp = torch.stack([
            data.x_mass,
            data.x_dipole,
            data.x_radius,
            data.x_en,
            data.x_ea,
            data.x_ie,
            data.x_vol,
            data.x_nvalence,
            data.x_block,
            data.x_mendeleev,
            data.x_vdw,
            data.x_atomic_rad,
            data.x_atomic_rad_emp,
            data.x_c6,
            data.x_tc,
            data.x_sh,
            data.x_fh,
            data.x_eh,
            data.x_period,
        ], dim=0)  # shape: (num_features, N, em_dim)

        tmp2 = torch.permute(tmp, (1, 2, 0))  # shape: (N, em_dim, num_features)
        data.x = torch.permute(self.em_mixing(tmp2), (2, 0, 1)).reshape(-1, self.em_dim)  # final shape: (N, em_dim)

        # === Superclass forward ===
        output = super().forward(data)
        output = torch.relu(output)  # enforce non-negativity

        # === Optional pooling ===
        if self.pool:
            output = torch_scatter.scatter_mean(output, data.batch, dim=0)

        return output

# %%
# out_dim = len(df.iloc[0]['energies_interp'])      # about 200 points
out_dim = 1


# random search
em_dim = 96
layers = 3
mul = 64
lmax = 2
max_radius = 6.




print(f"em_dim = {em_dim}")
print(f"layers = {layers}")
print(f"mul = {mul}")
print(f"lmax = {lmax}")
print(f"max_radius = {max_radius}")

model = PeriodicNetwork(
    in_dim=118,                            # dimension of one-hot encoding of atom type
    em_dim=em_dim,                         # dimension of atom-type embedding
    irreps_in=str(em_dim)+"x0e",           # em_dim scalars (L=0 and even parity) on each atom to represent atom type
    irreps_out=str(out_dim)+"x0e",         # out_dim scalars (L=0 and even parity) to output
    irreps_node_attr=str(em_dim)+"x0e",    # em_dim scalars (L=0 and even parity) on each atom to represent atom type
    layers=layers,                              # number of nonlinearities (number of convolutions = layers + 1)
    mul=mul,                                # multiplicity of irreducible representations
    lmax=lmax,                                # maximum order of spherical harmonics
    max_radius=r_max,                      # cutoff radius for convolution
    num_neighbors=n_train.mean(),          # scaling factor based on the typical number of neighbors
    reduce_output=True                     # whether or not to aggregate features of all atoms at the end
)

print(model)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
# visualize tensor products of the model
# visualize_layers(model)

# %% [markdown]
# ### Training
# The model is trained using a mean-squared error loss function with an Adam optimizer.

# %%
opt = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.96)

#loss_fn = torch.nn.HuberLoss()
loss_fn = torch.nn.MSELoss()
loss_fn_mae = torch.nn.L1Loss()
new_loss = New_Loss()

# %%
# run_name = 'model_alpha_' + run_time
run_name = 'model_polarization_' + run_time

model.pool = True
train(model, opt, dataloader_train, dataloader_valid, loss_fn, loss_fn_mae, run_name, 
       max_iter=100, scheduler=scheduler, device=device)


# %%

new_loss_avg = evaluate(model, dataloader_valid, new_loss, loss_fn_mae, device)
print(f"New Loss Average = {new_loss_avg}")

# %%
# load pre-trained model and plot its training history
#run_name = 'model_alpha_mass_' + run_time
history = torch.load('./model/' + run_name + '.torch', map_location=device)['history']
steps = [d['step'] + 1 for d in history]
loss_train = [d['train']['loss'] for d in history]
loss_valid = [d['valid']['loss'] for d in history]

# np.savetxt(run_name+'_MSE_loss.txt', np.column_stack((steps, loss_train, loss_valid)), fmt='%.8f', delimiter='\t')

# fig, ax = plt.subplots(figsize=(4,4))
# ax.plot(steps, loss_train, 'o-', label="Training", color='C0')
# ax.plot(steps, loss_valid, 'o-', label="Validation", color='C3')
# ax.set_xlabel('Epochs')
# ax.set_ylabel('Loss')
# ax.legend(frameon=False)
# plt.tight_layout()
# fig.savefig(run_name + '_loss.pdf')

# # %% [markdown]
# # ### Results
# # We evaluate our model by visualizing the predicted and true optical spectra in each error quartile.

# # %%
# # predict on all data
# model.load_state_dict(torch.load('./model/'+run_name + '.torch', map_location=device)['state'])
# model.pool = True

# dataloader = tg.loader.DataLoader(df['data'].values, batch_size=64)
# df['mse'] = 0.
# df['y_pred'] = np.empty((len(df), 0)).tolist()

# model.to(device)
# model.eval()

# # weight contribution of each feature
# weight = torch.abs(model.em_mixing.weight)/(torch.sum(torch.abs(model.em_mixing.weight), dim=1, keepdim=True)+1e-10)
# print(f"Weight = {weight}")

# with torch.no_grad():
#     i0 = 0
#     for i, d in tqdm(enumerate(dataloader), total=len(dataloader), bar_format=bar_format):
#         d.to(device)
#         output = model(d)
#         loss = F.mse_loss(output, d.y, reduction='none').mean(dim=-1).cpu().numpy()
#         df.loc[i0:i0 + len(d.y) - 1, 'y_pred'] = [[k] for k in output.cpu().numpy()]
#         df.loc[i0:i0 + len(d.y) - 1, 'mse'] = loss
#         i0 += len(d.y)
        
# # df['y_pred'] = df['y_pred'].map(lambda x: x[0])*scale_data
# df['y_pred'] = df['y_pred'].map(lambda x: x[0])

# np.savetxt(run_name+'_y_pred.txt', df['y_pred'].values, fmt='%.8f', delimiter='\t')


# # %%
# df_test = df.iloc[idx_test]
# np.savetxt(run_name + '_true_vs_pred.txt',
#            np.column_stack((df_test["polarization"].values, df_test["y_pred"].values)),
#            fmt='%.8f', delimiter='\t', header='True_Polarization\tPredicted')



# fig, ax = plt.subplots(figsize=(4, 4))
# ax.plot(df_test["polarization"], 10**(df_test["y_pred"]) - 1, '.', color='C0', label='Transformed Prediction')
# ax.plot(df_test["polarization"], df_test["polarization"], color='C3', label='Ideal (y = x)')
# ax.set_xlabel('True Polarization')
# ax.set_ylabel('10^Predicted - 1')
# ax.legend(frameon=False)
# plt.tight_layout()
# fig.savefig(run_name + '_correlation.pdf')


# from sklearn.metrics import r2_score

# r2_value = r2_score(df_test["polarization"], 10**(df_test['y_pred'])-1)
# print(f"R squared = {r2_value}")
# print(f"Run Name = {run_name}")

import os

output_dir = "/home/mariyaah/data1/GNNOpt/RANDOM_SEARCH_DATA"
os.makedirs(output_dir, exist_ok=True)
path = lambda name: os.path.join(output_dir, name)

np.savetxt(path(run_name + '_MSE_loss.txt'), np.column_stack((steps, loss_train, loss_valid)), fmt='%.8f', delimiter='\t')

fig, ax = plt.subplots(figsize=(4,4))
ax.plot(steps, loss_train, 'o-', label="Training", color='C0')
ax.plot(steps, loss_valid, 'o-', label="Validation", color='C3')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.legend(frameon=False)
plt.tight_layout()
fig.savefig(path(run_name + '_loss.pdf'))

# predict on all data
model.load_state_dict(torch.load('./model/' + run_name + '.torch', map_location=device)['state'])
model.pool = True

dataloader = tg.loader.DataLoader(df['data'].values, batch_size=16)
df['mse'] = 0.
df['y_pred'] = np.empty((len(df), 0)).tolist()

model.to(device)
model.eval()

# weight contribution of each feature
weight = torch.abs(model.em_mixing.weight)/(torch.sum(torch.abs(model.em_mixing.weight), dim=1, keepdim=True)+1e-10)
print(f"Weight = {weight}")

with torch.no_grad():
    i0 = 0
    for i, d in tqdm(enumerate(dataloader), total=len(dataloader), bar_format=bar_format):
        d.to(device)
        output = model(d)
        loss = F.mse_loss(output, d.y, reduction='none').mean(dim=-1).cpu().numpy()
        df.loc[i0:i0 + len(d.y) - 1, 'y_pred'] = [[k] for k in output.cpu().numpy()]
        df.loc[i0:i0 + len(d.y) - 1, 'mse'] = loss
        i0 += len(d.y)

df['y_pred'] = df['y_pred'].map(lambda x: x[0])
np.savetxt(path(run_name + '_y_pred.txt'), df['y_pred'].values, fmt='%.8f', delimiter='\t')

# df_test = df.iloc[idx_test]
# np.savetxt(path(run_name + '_true_vs_pred_test.txt'),
#            np.column_stack((df_test["polarization"].values, df_test["y_pred"].values)),
#            fmt='%.8f', delimiter='\t', header='True_Polarization\tPredicted')

# fig, ax = plt.subplots(figsize=(4, 4))
# ax.plot(df_test["polarization"], 10**(df_test["y_pred"]) - 1, '.', color='C0', label='Transformed Prediction')
# ax.plot(df_test["polarization"], df_test["polarization"], color='C3', label='Ideal (y = x)')
# ax.set_xlabel('True Polarization')
# ax.set_ylabel('10^Predicted - 1')
# ax.legend(frameon=False)
# plt.tight_layout()
# fig.savefig(path(run_name + '_correlation_test.pdf'))

df_valid = df.iloc[idx_valid]

np.savetxt(path(run_name + '_true_vs_pred_valid.txt'),
           np.column_stack((df_valid["polarization"].values, df_valid["y_pred"].values)),
           fmt='%.8f', delimiter='\t', header='True_Polarization\tPredicted')

fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(df_valid["polarization"], 10**(df_valid["y_pred"]) - 1, '.', color='C0', label='Transformed Prediction')
ax.plot(df_valid["polarization"], df_valid["polarization"], color='C3', label='Ideal (y = x)')
ax.set_xlabel('True Polarization')
ax.set_ylabel('10^Predicted - 1')
ax.legend(frameon=False)
plt.tight_layout()
fig.savefig(path(run_name + '_correlation_valid.pdf'))

df_train = df.iloc[idx_train]
np.savetxt(path(run_name + '_true_vs_pred_train.txt'),
           np.column_stack((df_train["polarization"].values, df_train["y_pred"].values)),
           fmt='%.8f', delimiter='\t', header='True_Polarization\tPredicted')

fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(df_train["polarization"], 10**(df_train["y_pred"]) - 1, '.', color='C0', label='Transformed Prediction')
ax.plot(df_train["polarization"], df_train["polarization"], color='C3', label='Ideal (y = x)')
ax.set_xlabel('True Polarization')
ax.set_ylabel('10^Predicted - 1')
ax.legend(frameon=False)
plt.tight_layout()
fig.savefig(path(run_name + '_correlation_train.pdf'))

fig_hist, ax_hist = plt.subplots(figsize=(4, 4))
ax_hist.hist(df_valid["y_pred"], bins=5, edgecolor='black')
ax_hist.set_xlabel(f'Predicted Polarization')
ax_hist.set_ylabel('Frequency')
ax_hist.set_title(f'Histogram of Predicted Polarization')
plt.tight_layout()
fig_hist.savefig(path(run_name + f'_histogram_y_pred_valid.pdf'))
fig2_hist, ax2_hist = plt.subplots(figsize=(4, 4))
ax2_hist.hist(df_valid["polarization"], bins=5, edgecolor='black')
ax2_hist.set_xlabel(f'Polarization')
ax2_hist.set_ylabel('Frequency')
ax2_hist.set_title(f'Histogram of Polarization')
plt.tight_layout()
fig2_hist.savefig(path(run_name + f'_histogram_real_valid.pdf'))


from sklearn.metrics import r2_score
r2_value = r2_score(df_valid["polarization"], 10**(df_valid['y_pred']) - 1)

print(f"R squared = {r2_value}")

print(f"Run Name = {run_name}")





# %%
# plot_predictions(df, idx_train, column='absorption_coefficient_interp', header=run_name, title='Training')

# # %%
# plot_predictions(df, idx_valid, column='absorption_coefficient_interp', header=run_name, title='Validation')

# # %%
# plot_predictions(df, idx_test, column='absorption_coefficient_interp', header=run_name, title='Testing')

# # %%
# column = 'absorption_coefficient_interp'

# df_tr = df.iloc[idx_train][['formula','energies_interp', column, 'y_pred', 'mse']]
# dx_tr = np.array([df_tr.iloc[i]['energies_interp'] for i in range(len(df_tr))])
# gt_tr = np.array([df_tr.iloc[i][column] for i in range(len(df_tr))])
# pr_tr = np.array([df_tr.iloc[i]['y_pred'] for i in range(len(df_tr))])
# # Weighted mean of ground true and predicted values.
# wgt_tr = weighted_mean(dx_tr, gt_tr)/1.0E6
# wpr_tr = weighted_mean(dx_tr, pr_tr)/1.0E6
# # Calculate R^2 value
# r2_tr = r2_score(wgt_tr, wpr_tr)

# df_va = df.iloc[idx_valid][['formula','energies_interp', column, 'y_pred', 'mse']]
# dx_va = np.array([df_va.iloc[i]['energies_interp'] for i in range(len(df_va))])
# gt_va = np.array([df_va.iloc[i][column] for i in range(len(df_va))])
# pr_va = np.array([df_va.iloc[i]['y_pred'] for i in range(len(df_va))])
# # Weighted mean of ground true and predicted values.
# wgt_va = weighted_mean(dx_va, gt_va)/1.0E6
# wpr_va = weighted_mean(dx_va, pr_va)/1.0E6
# # Calculate R^2 value
# r2_va = r2_score(wgt_va, wpr_va)

# df_te = df.iloc[idx_test][['formula','energies_interp', column, 'y_pred', 'mse']]
# dx_te = np.array([df_te.iloc[i]['energies_interp'] for i in range(len(df_te))])
# gt_te = np.array([df_te.iloc[i][column] for i in range(len(df_te))])
# pr_te = np.array([df_te.iloc[i]['y_pred'] for i in range(len(df_te))])
# # Weighted mean of ground true and predicted values.
# wgt_te = weighted_mean(dx_te, gt_te)/1.0E6
# wpr_te = weighted_mean(dx_te, pr_te)/1.0E6
# # Calculate R^2 value
# r2_te = r2_score(wgt_te, wpr_te)

# print('R^2 (Train) = {:.4f}'.format(r2_tr), 'R^2 (Valid) = {:.4f}'.format(r2_va), 'R^2 (Test) = {:.4f}'.format(r2_te))

# # Plot the correction between ground true and predicted values.
# fig, ax = plt.subplots(figsize=(4,4))

# # Add a diagonal line for reference
# min_val = min(np.min(wgt_tr), np.min(wpr_tr), np.min(wgt_va), np.min(wpr_va), np.min(wgt_te), np.min(wpr_te))
# max_val = max(np.max(wgt_tr), np.max(wpr_tr), np.max(wgt_va), np.max(wpr_va), np.max(wgt_te), np.max(wpr_te))
# width = max_val - min_val
# ax.plot([min_val-1, max_val+1], [min_val-1, max_val+1], 'k--', label='Perfect Correlation')

# ax.scatter(wgt_tr, wpr_tr, s=15, marker ="o", alpha=0.7, color='C0', label='Train')
# ax.scatter(wgt_va, wpr_va, s=12, marker ="s", alpha=0.7, color='C1', label='Valid')
# ax.scatter(wgt_te, wpr_te, s=15, marker ="v", alpha=0.7, color='C2', label='Test')

# ax.set_xlabel(r'True $\overline{\alpha}*$ ($\times 10^6$ cm$^{-1}$)')
# ax.set_ylabel(r'Predicted $\overline{\alpha}$ ($\times 10^6$ cm$^{-1}$)')

# #ax.set_xlim(min_val-0.01*width, max_val+0.01*width)
# #ax.set_ylim(min_val-0.01*width, max_val+0.01*width)
# ax.set_xlim(0.0, 1.3)
# ax.set_ylim(0.0, 1.3)
# # Set the tick distance
# x_ticks = np.arange(0, 1.3, 0.3) 
# ax.set_xticks(x_ticks)
# y_ticks = np.arange(0, 1.3, 0.3) 
# ax.set_yticks(y_ticks)
# ax.set_aspect('equal')
# #plt.text(0.7, 0.36, r'$R^2$ (Train) = {:.4f}'.format(r2_tr), fontsize = 12, color = 'C0')
# #plt.text(0.7, 0.24, r'$R^2$ (Valid) = {:.4f}'.format(r2_va), fontsize = 12, color = 'C1')
# #plt.text(0.7, 0.12, r'$R^2$ (Test) = {:.4f}'.format(r2_te), fontsize = 12, color = 'C2')

# plt.tight_layout()
# fig.savefig(run_name + '_correlation.pdf')

# plt.show()

# # %%
# #Relative error between true and predicted values.
# error_tr = np.abs(wgt_tr - wpr_tr) / wgt_tr
# error_va = np.abs(wgt_va - wpr_va) / wgt_va
# error_te = np.abs(wgt_te - wpr_te) / wgt_te

# density = np.sum(error_te < 0.10)/len(error_te)
# print('Relative error density (Test) < 10%:', density)

# #print(error_te)

# # Plot relative error density
# fig, ax = plt.subplots(figsize=(4,3))
# #plt.hist(error_te, bins=200, density=True, alpha=0.7, color='blue')
# #plt.hist(error_va, bins=200, density=True, alpha=0.7, color='blue')

# #sns.kdeplot(error_tr, fill=True, color='C0', cumulative=True, common_norm=True, label='Relative Error')
# #sns.kdeplot(error_va, fill=True, color='C1', cumulative=True, common_norm=True, label='Relative Error')
# sns.kdeplot(error_te, fill=True, color='C2', cumulative=True, common_norm=True, label='Relative Error')
# #plt.title('Relative Error Density')
# plt.axvline(x=0.10,color='gray', linestyle='--')
# plt.axhline(y=density, color='C3', linestyle='--')
# plt.ylim(0, 1)
# plt.xlim(0, 0.4)
# plt.xlabel('Relative Error', fontsize = 18)
# plt.ylabel('Density', fontsize = 18)
# plt.tight_layout()
# #fig.savefig(run_name + '_error.pdf', transparent=True)
# #plt.grid(True)
# plt.show()


