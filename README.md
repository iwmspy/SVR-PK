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

Dataset obtained from ChEMBL31, retrosynthesized by `generate_reactant_pairs.py` and preprocessed by `preprocess.py` can be obtained from this [ZENODO](https://doi.org/10.5281/zenodo.14729011) repository. Unzip the downloaded `datasets.zip`, rename the directory to `outputs`, and place the unzipped and renamed directory so that it looks as follows.
```
SVR-PK
|- outputs
    |- preprocessed : Retrosynthesized, preprocessed, split(train, test, val[only for MolCLR])
    |- emolecules : For screening
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

Ex.) For modeling, if you have a data set of target physical properties shown in below,
```
obj_idx   obj_sml  objective
0   c1ccccc1    1.3
...
```
Then your configuration file will looks following.
```
{
    "files": ["path/to/your/objective/dataset"],    # Must be list
    "index_col": "obj_idx",
    "model_smiles_col": "obj_sml",
    "objective_col": "objective",
    "split_level": 1,                               # Either 1 or 2
    "augmentation": 1,                              # 1: True, 0: False
}
```

For screening using the SVR-PK model built by previous process, if you have a data set of reactant candidates (`heavy_atom_count`, easily calculated by RDKit's `mol.GetNumHeavyAtoms()`, must be included in the dataset).
```
cnd_idx   cnd_sml   heavy_atom_count
0   c1ccccc1    6
...
```
Then your configuration file will looks following.
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

## 1. Generate models for screening reactants

Run the command.

```
python build_model.py -c [your_config_file].json
```

Our results in the paper can be reproduced using the json file below.

* `chembl_config_lv1.json`: Product-based splitting
* `chembl_config_lv1_augment.json`: Product-based splitting with data augmentation
* `chembl_config_lv2.json`: Reactant-based splitting
* `chembl_config_lv2_augment.json`: Reactant-based splitting with data augmentation

[MolCLR](https://github.com/yuyangw/MolCLR) is used for comparison of accuracies. Note that you need to create new environment named `molclr`. For details, please refer `SVR-PK/models/MolCLR/README.md`.

```
python build_model.py -c [your_config_file].json
```

If you have your own dataset, you can run as follows.

```
python rm_main.py -c [your_config_file].json
```

## 2. Screen reactants and combine by generated models

Reactant screening using built SVR-PK (and SVR-baseline) models. Before screening, you should decide the reaction to use and write it on your configuration file (see 0. Configuration).

```
python reactant_screening.py -c [your_config_file].json
```

Our results in the paper can be reproduced by setting the following json file.
* `chembl_config_for_screening_1k.json`: Reactant pair screening (sample 1k reactants for each, just for rate measurement)
* `chembl_config_for_screening_10k.json`: Reactant pair screening (sample 10k reactants for each, just for rate measurement)
* `chembl_config_for_screening_100k.json`: Reactant pair screening (sample 100k reactant for each)


## 3. Methods for comparison
Thompson sampling was used as the comparison method (https://pubs.acs.org/doi/10.1021/acs.jcim.3c01790).

The comparison method can be run by following command. 
```
python reactant_screening_TS.py -c [your_config_file].json
```

You can use the same `json` file used in our method for configuration.

## Analyze results
Results will be stored `outputs` directory, and the spread sheets and figures can be produced following the procedure in the `analysis.ipynb` notebook.
```
SVR-PK
|- outputs
    |- ...
    |- prediction_level{split_level}[_augmented] : results of model construction
    |- reactant_combination_level{split_level}[\_augmented]\_{n_samples}[\_rc{downsize_sc}] : proposed reactant pairs
    |- ...
```
