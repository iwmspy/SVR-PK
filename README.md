# Implementation of SVR-PK and virtual screening protocol

Support vector regression with a prodcut kernel (SVR-PK) can efficiently evaluate a molecule consisting of multiple reactant components. Each component has a kernel function (default: Tanimoto kernel) and the product of the kernel functions form the kernel function in SVR. 
For more details, please refere to the [paper](URL_to_the_paper)

## Environment setting
[Conda](https://www.anaconda.com/docs/getting-started/miniconda/main) is recommended to handle packages in a virtual environment. Required packages are listed in `_env/requirements.yml`.

```bash
git clone https://github.com/iwmspy/SVR-PK.git
cd SVR-PK
conda create svr-pk -f _env/requirements.yml
conda activate svr-pk
```

To fully reproduce the procedure as shown in the [paper](URL_to_the_paper), two external repositories are necessary.
1.  [retrosim](https://github.com/connorcoley/retrosim)  (found in `SVR-PK/retrosynthesis`)
2.  [TS](https://github.com/PatWalters/TS) (found in `SVR-PK/_benchmarking/Thompson`). 

These two libraries have been modified from the original ones for meeting our purposes. The terms of licenses are provided in each folder with modification points specified and the modified files are specified by the `_iwmspy` suffix.

Datasets to reproduce the results in the paper can be downloaded from the ZENODO repository [ZENODO](https://doi.org/10.5281/zenodo.15522920). The `outputs.zip` file contains the necessary data.

Simply download the zip file and unzip the file and put the folder in the `SVR-PK` folder.
The structure of the folders should be this.
```
SVR-PK
|- outputs
    |- preprocessed : Retrosynthesized, preprocessed, split(train, test, val[only for MolCLR])
    |- emolecules : For screening
```

Using a shell, you can use the following commands under the `SVR-PK` folder.

```bash 
wget https://zenodo.org/records/15522920/files/outputs.zip
unzip outputs.zip
```

## 0. Configuration
You can use your datasets by specifying the information in a config file and run `rm_main.py` (retrosynthesis, preprocess and modeling) and `reactant_screening.py` (screening reactants with using SVR-PK models built by the previous process) with`-c [your config file].json` option. Sample json files are found in the `config` directory.

|Parameter|Description|
|----|----|
|modeling params| === |
|`files`|List of file names containing compounds and target objectives|
|`index_col`|Index column name of `files`|
|`model_smiles_col`|SMILES column name of `files`|
|`objective_col`|Column name containing objective values of `files`|
|`split_level`|Level of train-test splitting. See paragraph|
|`augmentation`|Whether data augmentation of training dataset is implemented or not|
|screening params| === |
|`cand_path`|File name of reactant candidates (If you just want to reprecate our results, you don't have to set this parameter)|
|`cand_index_col`|Index column name of `cand_path`|
|`cand_smiles_col`|SMILES column name of `cand_path`|
|`reactions`|List of reaction names for generating compounds from reactants|
|`n_samples`|Number of reactant pairs to be extracted|
|`downsize_sc`|Number of reactants to be sampled in our method|
|`downsize_ts`|Number of reactants to be sampled in Thompson sampling|
|`ext_ratio`|Ratio of number of pairs to be extracted in first screening. The smaller number of pairs will be extracted by the smaller ratio|
|`precalc`|Whether reactant smiles are already extracted and mapped or not (If you just want to reprecate our results, you have to set this parameter `1`)|
|`postprocess`|Whether postprocess (synthesizability check etc.) is inplemented or not in screening (Only for Thompson sampling)|

For modeling, a dataset of molecules and properties/activities has the following format.
```
obj_idx   obj_sml  objective
0   c1ccccc1    1.3
...
```
A corresponding configuration file is like this.
```
{
    "files": ["path/to/your/objective/dataset"],    # Must be a list
    "index_col": "obj_idx",
    "model_smiles_col": "obj_sml",
    "objective_col": "objective",
    "split_level": 1,                               # Either 1 or 2
    "augmentation": 1,                              # 1: True, 0: False
}
```

For screening reactants using the SVR-PK model built by previous process, you need to set the reactant candidate dataset with `heavy_atom_count`, which can easily be calculated with RDKit's `mol.GetNumHeavyAtoms()` module. A reactant dataset has the following format.

```
cnd_idx   cnd_sml   heavy_atom_count
0   c1ccccc1    6
...
```
A corresponding configuration file is like this.
```
{
    "files": ["path/to/your/objective/dataset"],    # Must be list, same as the modeling
    "index_col": "obj_idx",                         # Same as the modeling
    "model_smiles_col": "obj_sml",                  # Same as the modeling
    "objective_col": "objective",                   # Same as the modeling
    "split_level": 1,                               # Either 1 or 2
    "augmentation": 1,                              # 1: True, 0: False
    "cand_path": "path/to/your/candidate/dataset",
    "cand_index_col": "cnd_idx",
    "cand_smiles_col": "cnd_sml",
    "reactions": ["name_of_reaction"],              # Please refer the results of modeling stored in 
                                                    # 'SVR-PK/outputs/prediction_level{split_level}[_augmented]' 
                                                    # and deside which reaction you want to use
    "n_samples": 10000,
    "downsize_sc": 100000,
    "ext_ratio": 1e-5,
    "precalc": 1,                                   # 1: True, if you already have the results of substructure matching
}
```

## 1. Create QSAR models for screening reactants

Run the command to make QSAR models. 

```
python build_model.py -c [your_config_file].json
```

Our results in the paper can be reproduced using the json file below.

* `chembl_config_lv1.json`: Product-based splitting
* `chembl_config_lv1_augment.json`: Product-based splitting with data augmentation
* `chembl_config_lv2.json`: Reactant-based splitting
* `chembl_config_lv2_augment.json`: Reactant-based splitting with data augmentation

To make models with SVR-PK with data augmentation for the rectant-based splitting datasets, run the command below.

```bash
python build_model.py -c config/chembl_config_lv2_augment.json
```

Then, models with SVR-PK for each reaction dataset will be created. 
The results will be saved `SVR-PK/outputs/prediction_level2_augmented` folder with target IDs as subfolders.
In each subfolder, you can find these files: (Fix this:bug: :exclamation:)
- mod.pickle
- prediction_results_prd_test(train).tsv
- prediction_results_rct..

:exclamation:[MolCLR](https://github.com/yuyangw/MolCLR) is used for comparison of accuracies. Note that you need to create new environment named `molclr`. For details, please read `SVR-PK/models/MolCLR/README.md`.


## 2. Screen reactants and combine by generated models

Reactant screening using built SVR-PK (and SVR-baseline) models. Before screening, you should decide a virtual reaction and write it on your configuration file (see 0. Configuration).

```
python reactant_screening.py -c [your_config_file].json
```

Our results in the paper can be reproduced by setting the following json file.
* `chembl_config_for_screening_1k.json`: Reactant pair screening (sample 1k reactants for each, just for rate measurement)
* `chembl_config_for_screening_10k.json`: Reactant pair screening (sample 10k reactants for each, just for rate measurement)
* `chembl_config_for_screening_100k.json`: Reactant pair screening (sample 100k reactant for each)

To screen virtual reactants using sampled 1k reactants, use the following command:

```bash
python reactant_screening.py -c config/chembl_config_for_screening_1k.json
```
The screened results are stored in XXX :bug: :exclamation:

## 3. Methods for comparison
In the paper, a Thompson sampling-based screening method was used as a comparison method (https://pubs.acs.org/doi/10.1021/acs.jcim.3c01790).

To screen reactants using the sampling, run the following commnad.
```
python reactant_screening_TS.py -c [your_config_file].json
```
You can use the same `json` file used in our method for configuration.
For example, :bug: :exclamation:
```bash
python reactant_screening_TS.py -c XXX.json 
```
The screened results are stored in XXX :bug: :exclamation:


## Analyze results
All the results will be stored in the `outputs` directory, and the spread-sheets and figures used in the paper can be reproduced by following the `analysis.ipynb` notebook.
We assume the following folder structure to reproduce the results.

```
SVR-PK
|- outputs
    |- ...
    |- prediction_level{split_level}[_augmented] : results of model construction
    |- reactant_combination_level{split_level}[\_augmented]\_{n_samples}[\_rc{downsize_sc}] : proposed reactant pairs
    |- ...
```

## An example notebook
The above analysis is summarized in the `reproduce_paper_analysis.ipynb` notebook for easy reference. :bug: :exclamation: