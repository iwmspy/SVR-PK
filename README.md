# Implementation of SVR-PK and virtual screening protocol

Support vector regression with a prodcut kernel (SVR-PK) can efficiently evaluate a molecule consisting of multiple reactant components. Each component has a kernel function (default: Tanimoto kernel) and the product of the kernel functions form the kernel function in SVR. 
For more details, please refere to the [paper](URL_to_the_paper)

## Environment setting
[Conda](URL_to_conda) is recommended to handle packages in a virtual environment. Required packages are listed in `_env/requirements.yml`.

```bash
git clone https://github.com/iwmspy/SVR-PK.git
cd SVR-PK
conda create svr-pk -f _env/requirements.yml
conda activate svr-pk
```

To fully reproduce the procedure as shown in the [paper](URL_to_the_paper), two external repositories are necessary.
1.  [retrosim](https://github.com/connorcoley/retrosim)  (found in `SVR-PK/retrosynthesis`)
2.  [TS](https://github.com/PatWalters/TS) (found in `SVR-PK/_benchmarking/Thompson`). 

These two libraries have been modified from the original ones for meeting our purposes. The terms of licenses are provided in each folder with modification points specified.and the modified files are specified by the `_iwmspy` suffix.

Preprocessed datasets can be obtained from this [ZENODO](https://doi.org/10.5281/zenodo.14729011) repository. Unzip the downloaded file and place it under the `outputs` folderhas the following status.
```
SVR-PK
|- outputs
    |- preprocessed : Retrosynthesized, preprocessed, split(train, test, val[only for MolCLR])
    |- emolecules : For screening
```

## 0. Configuration
You can use your datasets by specifying the information in a config file and run XXX (what!!) with`-c [your config file].json` option. Sample json files are found in the `config` directory.

|Parameter|Description|
|----|----|
|modeling params| === |
|<code>files</code>|List of file names containing compounds and target objectives|
|<code>index_col</code>|Index column name of <code>files</code>|
|<code>model_smiles_col</code>|SMILES column name of <code>files</code>|
|<code>objective_col</code>|Column name containing objective values of <code>files</code>|
|<code>split_level</code>|Level of train-test splitting. See paragraph|
|<code>augmentation</code>|Whether data augmentation of training dataset is implemented or not|
|screening params| === |
|<code>cand_path</code>|File name of reactant candidates (If you just want to reprecate our results, you don't have to set this parameter)|
|<code>cand_index_col</code>|Index column name of <code>cand_path</code>|
|<code>cand_smiles_col</code>|SMILES column name of <code>cand_path</code>|
|<code>reactions</code>|List of reaction names for generating compounds from reactants|
|<code>n_samples</code>|Number of reactant pairs to be extracted|
|<code>downsize_sc</code>|Number of reactants to be sampled in our method|
|<code>downsize_ts</code>|Number of reactants to be sampled in Thompson sampling|
|<code>ext_ratio</code>|Ratio of number of pairs to be extracted in first screening. The smaller number of pairs will be extracted by the smaller ratio|
|<code>precalc</code>|Whether reactant smiles are already extracted and mapped or not (If you just want to reprecate our results, you have to set this parameter <code>1</code>)|
|<code>postprocess</code>|Whether postprocess (synthesizability check etc.) is inplemented or not in screening|

## 1. Generate models for screening reactants

Run the command.

```
python build_model.py -c [your_config_file].json
```

Our results in the paper can be reproduced using the json file below.

* <code>chembl_config_lv1.json</code>: Product-based splitting
* <code>chembl_config_lv1_augment.json</code>: Product-based splitting with data augmentation
* <code>chembl_config_lv2.json</code>: Reactant-based splitting
* <code>chembl_config_lv2_augment.json</code>: Reactant-based splitting with data augmentation

[MolCLR](https://github.com/yuyangw/MolCLR) is used for comparison of accuracies. Note that you need to create new environment named <code>molclr</code>. For details, please refer <code>SVR-PK/models/MolCLR/README.md</code>.

```
python build_model.py -c [your_config_file].json
```

If you have your own dataset, you can run as follows.

```
python rm_main.py -c [your_config_file].json
```

## 2. Screen reactants and combine by generated models

Reactant screening.

```
python reactant_screening.py -c [your_config_file].json
```

Our results in the paper can be reproduced by setting the following json file.

* <code>chembl_config_for_screening_1k.json</code>: Reactant pair screening (sample 1k reactants for each, just for rate measurement)
* <code>chembl_config_for_screening_10k.json</code>: Reactant pair screening (sample 10k reactants for each, just for rate measurement)
* <code>chembl_config_for_screening_100k.json</code>: Reactant pair screening (sample 100k reactant for each)

If you want to obtain reactant pairs from your own reactant file, the <code>heavy_atom_count</code> (number of heavy atoms of compound, easily calculated by RDKit) column should be contained in your own reactant file.


## 3. Methods for comparison
Thompson sampling was used as the comparison method (https://pubs.acs.org/doi/10.1021/acs.jcim.3c01790).

The comparison method can be run by following command. 
```
python reactant_screening_TS.py -c [your_config_file].json
```

You can use the same <code>json</code> file used in our method for configuration.

## Analyze results
Results will be stored <code>outputs</code> directory, and the figures can be produced following the procedure in the `analysis.ipynb` notebook.

```
SVR-PK
|- outputs
    |- ...
    |- prediction_level{n}[_augmented] : results of model construction
    |- reactant_combination_level{n}[\_augmented]\_{m}[\_rc{l}] : proposed reactant pairs
    |- ...
```