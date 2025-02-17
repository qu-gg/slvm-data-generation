# slvm-data-generation
### Description
Personal convenience repository to hold data generation scripts for all of my PhD works. This is not presented as my own purely original work, just <i>my</i> modifications for datasets used my works. As well, there is not an effort to make the code reuseable between datasets or sophisticated. The goal here is to have isolated, dedicated scripts to go to a dataset and run it as when I had ran it before.

### Credit
<ul>
<li> <b>Bouncing Ball</b> and <b>Bouncing Ball [Gravity]</b>: Sourced from the <a href="https://github.com/simonkamronn/kvae">KVAE</a> repo.
<li> <b>Double Pendulum</b>, <b>Mass Spring</b>, <b>Pendulum</b>, <b>Two Body</b>: Sourced from the <a href="https://github.com/google-deepmind/dm_hamiltonian_dynamics_suite/tree/master?tab=readme-ov-file">DM Hamiltonian Suite</a>.
<li> <b>NASCAR</b>: Sourced from <a href="https://github.com/slinderman/recurrent-slds">R-SLDS</a>.
<li> <b>Lorenz</b> and <b>Double Pendulum</b>: Sourced from <a href="https://github.com/ostadabbas/DSARF/tree/master/">DSARF</a>.
</ul>

### Mixed-Physics ICLR 2023 vs. 2025
My ICLR 2023 (meta-sLVM) and 2025 (CoSFan) works consider slightly different datasets of mixed Hamiltonian dynamics but were considered under the same name as they represented a similar concept. Each version has its own included dataset generation folder. ICLR 2023 considered 3 systems - Gravity, Mass Spring, and Pendulum - with in-system heterogeneity stemming from differing friction values. ICLR 2025 considered 4 systems - Gravity, Pendulum, Double Pendulum, and Two Body - with in-system heterogeneity stemming from differing gravity magnitudes or spring constants.

### Data Generation Conventions
To run a single-dynamic dataset (e.g., one single gravvity affecting a bouncing ball), in each data folder, run <code>python3 name_of_dataset.py</code>. This will generate a dedicated output folder with the same dataset name containing 3 .npz files - train, val, test. These contain images sequences and ground-truth states (if they exist/are known for a system).

To run a multi-dynamic dataset (e.g., multiple parameter configurations of one shared Hamiltonian equation), in each data folder, run <code>python3 name_of_datasetX.py</code> where <code>X</code> is the number of dynamics to run. This varies by dataset, but most Hamiltonian Dynamics only have a max of 3. This will generate a dedicated output folder with the same dataset name, now containing <code>X</code> number of subfolders designated as <code>dataset_i</code> where <code>i</code> is the label of the given parameter configuration. Each subfolder then contains a similar 3 .npz files - train, val, test. These contain images sequences and ground-truth states (if they exist/are known for a system).

### Dataloaders
Provided in each dataset style are some example dataloaders I use in my works, one for standard dynamics learning and one for meta-learning. 

#### Hamiltonian Dynamics
The standard Hamiltonian <code>dataloader.py</code> is much like any other PyTorch dataloader, just loading in the static .npz file, converting it into Tensors, and giving an example get_ function.

The meta-learning dataloader <code>dataloader_meta_learning.py</code> takes a multi-dynamic dataset, which is a folder of subfolders each pertaining to a separate task, and combines them together into the loaded in dataset. It builds a mapping between task labels and sample indices and contains a built in split_ function that makes new context-query pairs at each episode, which must be called in the training code at each episode start (whether that is per-batch or per-epoch).

#### Switching Dynamics
The standard Switching <code>dataloader.py</code> is much like any other PyTorch dataloader, just loading in the static .npz file, converting it into Tensors, and giving an example get_ function