# slvm-data-generation
### Description
Personal convenience repository to hold data generation scripts for all of my PhD works. Lot of this will be uncredited and is not to be presented as my own work, just <i>my</i> modifications for my works. As well, there is not an effort to make the code reuseable between datasets or sophisticated. The goal here is to have isolated, dedicated scripts to go to a dataset and run it as when I had ran it before.

### Data Generation Conventions
To run a single-dynamic dataset (e.g., one single gravvity affecting a bouncing ball), in each data folder, run <code>python3 name_of_dataset.py</code>. This will generate a dedicated output folder with the same dataset name containing 3 .npz files - train, val, test. These contain images sequences and ground-truth states (if they exist/are known for a system).

To run a multi-dynamic dataset (e.g., multiple parameter configurations of one shared Hamiltonian equation), in each data folder, run <code>python3 name_of_datasetX.py</code> where <code>X</code> is the number of dynamics to run. This varies by dataset, but most Hamiltonian Dynamics only have a max of 3. This will generate a dedicated output folder with the same dataset name, now containing <code>X</code> number of subfolders designated as <code>dataset_i</code> where <code>i</code> is the label of the given parameter configuration. Each subfolder then contains a similar 3 .npz files - train, val, test. These contain images sequences and ground-truth states (if they exist/are known for a system).

### Dataloaders
Provided in each dataset style are some example dataloaders I use in my works, one for standard dynamics learning and one for meta-learning. 

#### Hamiltonian Dynamics
The standard Hamiltonian <code>dataloader.py</code> is much like any other PyTorch dataloader, just loading in the static .npz file, converting it into Tensors, and giving an example get_ function.

The meta-learning dataloader <code>dataloader_meta_learning.py</code> takes a multi-dynamic dataset, which is a folder of subfolders each pertaining to a separate task, and combines them together into the loaded in dataset. It builds a mapping between task labels and sample indices and contains a built in split_ function that makes new context-query pairs at each episode, which must be called in the training code at each episode start (whether that is per-batch or per-epoch).

#### Switching Dynamics
The standard Switching <code>dataloader.py</code> is much like any other PyTorch dataloader, just loading in the static .npz file, converting it into Tensors, and giving an example get_ function.