# CASTER-DTA
Developing modified GNN-based drug-target affinity prediction model(s) that incorporates 3D protein structural information.

## Introduction

This code accompanies the paper "CASTER-DTA: Equivariant Graph Neural Networks for Predicting Drug-Target Affinity", which can be found as a preprint at the following link: https://www.biorxiv.org/content/10.1101/2024.11.25.625281v1

Authors: Rachit Kumar, Joseph D. Romano, Marylyn D. Ritchie


This is the version as of submission of the code that was used to develop and train equivariant graph neural networks (GNNs) for predicting drug-target affinity as well as the code needed to perform the downstream analyses thereof.


## Setting up

### Base requirements

You should first clone the repository and enter it:

```
git clone https://github.com/rachitk/caster-dta
cd caster-dta
```

To install dependencies, make sure you are using Python 3.10 or 3.11; you can then run the following command (we recommend you run this in a virtual environment or Conda environment):

```
pip install -r requirements.txt
```

Please see the `requirements.txt` file to find a list of dependencies that will be installed.

Please note that torch-scatter can be installed by uncommenting its corresponding line in `requirements.txt` or by simply running `pip install torch_scatter`; however, this particular package installation can be quite slow. As such, we strongly recommend that you install torch-scatter from wheel instead, and you can find the link to it in https://data.pyg.org/whl/ for each torch+CUDA+OS combination. 

The `requirements.txt` pins torch to 2.2.2 to try to make this easier. You will likely want to look in https://data.pyg.org/whl/torch-2.2.2%2Bcpu.html or https://data.pyg.org/whl/torch-2.2.2%2Bcu121.html, though you may need to look in other directories for other CUDA versions.

After finding the link for the correct torch-scatter version, you can install it like so (though you should replace the path to the wheel as you found it above):

```
pip install https://data.pyg.org/whl/torch-2.2.0%2Bcpu/torch_scatter-2.1.2%2Bpt22cpu-cp310-cp310-linux_x86_64.whl
```


### Setting up Colabfold

We provide some instructions on how to set up ColabFold in `./ext-packages/setup_info.txt`. Do note that our implementation of running ColabFold does require that you have Singularity installed on your machine. Instructions for this can be found here: https://docs.sylabs.io/guides/3.0/user-guide/installation.html

ColabFold is used to run AlphaFold2 locally for folding any proteins that are not found in PDB. Do note that this can take some time depending on what GPUs you have access to, and if you would rather simply use proteins that are present in PDB only, you can modify a flag in `train_model.py` as described below; in that case, setting up ColabFold is not required.



### Visualization additional requirement

Note that for visualization, you will also need to install PyMOL. We recommend that you install the open-source version, which you can do through Conda, but the licensed version will also work if you obtain a valid license. 

We are unfortunately not able to provide instructions on how to do this as the process can vary substantially based on the operating system due to the licensing if you go that route, though do note PyMOL is only needed if running the `visualize_*` scripts.




## Training

### Davis, KIBA, and Metz datasets

We provide the Davis and KIBA datasets that were used in the paper. These datasets were downloaded from DeepDTA and are expected to be stored in `./data/deepdta_data/davis` and `./data/deepdta_data/kiba`, respectively.

We also provide a copy of the Metz dataset used in the paper. This dataset was downloaded from a different paper known as PADME and is expected to be stored in `./data/other_data/metz`. 

### BindingDB dataset

We provide in the `./data/other_data/bindingdb` folder a link in the README to download the BindingDB full dataset that we used, as we are unable to distribute it on Github due to its size. 

The README in this folder provides a link to the ZIP file, which you will have to extract into `BindingDB_all.tsv`. An MD5 checksum of the ZIP file is provided in the README as well. If you are having trouble accessing this data, we are happy to reupload it elsewhere if you contact us.

### Command for Training

An example command to simply run and train a CASTER-DTA(2,2) model using these datasets can be found below:

```
python train_model.py --dataset davis --seed 9 --out-folder davis_seed9
```

A more complex command that allows saving of the data and logging the results to a directory called `output` with the results stored in a subfolder based on the current date can be found below (this will also output the results to standard output).

```
env DTA_OUTFOLDER="output/"$(date +"%Y%m%d__%H_%M_%S")"" bash -c 'mkdir "${DTA_OUTFOLDER}" && python train_model.py --dataset davis --seed 9 --out-folder $DTA_OUTFOLDER | tee -a "${DTA_OUTFOLDER}/log.txt"'
```

In this manner, the model used for the downstream tasks was trained using (effectively) the following command - the output path is referenced in all of the downstream tasks scripts:

```
python train_model.py --dataset bindingdb --seed 9 --out-folder pretrained_model_downstream
```

#### Considerations

The `train_model.py` file provided trains CASTER-DTA(2,2) with the dataset details as described in the paper.

If you would like to change how the 3D protein structures are generated (or acquired), you can modify `load_dataset_kwargs` on lines 94-103. You can, for example, disable searching Protein Databank (`skip_pdb_dl`), enable PDB files with complexed ligands (`allow_complexed_pdb`), or disable making computational folds with AlphaFold2 (`create_comp`).

If you would like to change the model using one of the provided options in construction, you can modify the arguments that define the GNN on lines 289-344, with particular note of lines 298 and 314 to change the number of convolutions for each to produce the other CASTER-DTA variants such as CASTER-DTA(1,1).

You can also change the dataset arguments to try other edge thresholds and such (as described in our paper's ablation studies) by modifying lines 112-127. 

Each training command will save a variety of files to the output folder, some of which are important if wanting to reload the model for the downstream analyses:

- `dataset_kwargs.json`: a copy of the dataset arguments (for easier reproduction and to allow loading of a new dataset to match these)
- `dataset_rescale_params.json`: a copy of the parameters for rescaling the output of the dataset (to allow for unscaling results if rerun/reloaded)
- `model_kwargs.json`: a copy of the model construction parameters, which allows for the model to be rebuilt exactly as trained.
- `bestvalmodel_*.pt`: a file that contains the final model parameters to be reloaded into a trained model. The best one can automatically be reloaded by searching for the best performing ones. (A train equivalent is also available, if interested)

Other files (not critical, but useful) that are saved include:

- `train_model.py`: a copy of the training script as is when the command was run (allows for easier reproduction of results if the file is changed)
- `train_command.txt`: a copy of the command used to train the model (cleaned up) for easier reproduction if needed.
- `model_standardprint.txt`: a simple print of the model, useful for verifying the architecture matches expectations.
- `model_summary.txt`: a prettier version of the model architecture in text form, useful for verifying the architecture matches expectations.
- Several .pt files for epoch checkpoints which can be reloaded if interested, but are not necessary after training is done.


## Downstream Tasks

### Pretrained Model

The model used for all downstream tasks was CASTER-DTA(2,2) trained on BindingDB using seed 9. We provide a copy of this model (and required files for reloading it) in `pretrained_model_downstream`, which is where the downstream task scripts expect it. In this case, we provide only the parameters for the model actually used (the one with the best validation-performance during training), which is in the file `bestvalmodel_bindingdb_val0.6889_epoch01011.pt`. 

### Downstream task scripts

There are three downstream tasks for this paper with their own scripts for executing to get results and visualizing:

| **Task**           | **Test Script**           | **Visualization Script**         |
|--------------------|---------------------------|----------------------------------|
| BioLIP Binding     | `test_biolip_binding.py`  | `visualize_biolip_results.py`    |
| PharmGKB Variation | `test_dta_variation.py`   | `visualize_variation_results.py` |
| Protein Assessment | `test_protein_binders.py` | `visualize_binder_results.py`    |


### BioLIP data

The BioLIP data nonredundant set and ligand data must be downloaded and placed in `./data/biolip_data/BioLiP_nr.txt.gz` and `./data/biolip_data/ligand.tsv.gz`, respectively. Instructions on how to do so can be found on the BioLIP website: https://zhanggroup.org/BioLiP/download/readme.txt

### PharmGKB data

The PharmGKB variants were acquired as in the paper. No external resources are required for this analysis (other than possibly needing to setup ColabFold as described above) as the proteins will otherwise be downloaded or folded as needed.


### Protein binding data

For this analysis, the DrugBank complete dataset must be downloaded and placed in `./data/full database.xml`. This is only available with a license from DrugBank: https://go.drugbank.com/releases/latest. Each of the PDB files must also be downloaded from their respective PDB pages (as described in the script) and placed in the `./data/ad_data/pdb_files/` directory.


## Disclosures

Please note that the code found in `model/gvp_layers` is modified slightly from the code found at: https://github.com/drorlab/gvp-pytorch/blob/82af6b22eaf8311c15733117b0071408d24ed877/gvp/__init__.py


## Funding

RK was supported by the Training Program in Computational Genomics grant from the National Human Genome Research Institute to the University of Pennsylvania (T32HG000046). JDR was supported by NIH grant R00LM013646. MDR was supported by NIH grant U01AG066833. 


## Licensing

Please note that the code in this repository is under a special license. If you are interested in using the code, please make sure to read the `LICENSE` file. In brief, though this statement does not supersede any conditions in the license as it is written, this code is usable for noncommercial and academic use with attribution and requires a special license for commercial use.


## Citation

If you use the code in this repository in a research study (note the license information above as well), please cite the following preprint (we will update this upon formal publication):

```
@article {Kumar2024.11.25.625281,
	author = {Kumar, Rachit and Romano, Joseph D. and Ritchie, Marylyn D.},
	title = {CASTER-DTA: Equivariant Graph Neural Networks for Predicting Drug-Target Affinity},
	elocation-id = {2024.11.25.625281},
	year = {2024},
	doi = {10.1101/2024.11.25.625281},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2024/11/28/2024.11.25.625281},
	eprint = {https://www.biorxiv.org/content/early/2024/11/28/2024.11.25.625281.full.pdf},
	journal = {bioRxiv}
}
```