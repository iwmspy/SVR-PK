# Screening by SVR-PK

This codeset works following below procedure.

## Prepare enviromment and datasets

Required packages are writtten on <code>_env/requirements.yml</code>.

Please create and activate conda environment by following command.

```
conda env create -f _env/requirements.yml
conda activate svr-pk
```

Datasets we used can be obtained from this [ZENODO](https://doi.org/10.5281/zenodo.14729011) repository.

## 0. Configuration
You can use your original datasets by creating config file and run below code with <code>-c [your config file].json</code> option. Please refer json files stored in <code>config</code> to create your own config file. 

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

Run below code.

```
python rm_main.py -c [your_config_file].json
```

This procedure contains below modules.

* <code>generate_retrosynthesis.py</code>: For generating reactants from actual molecules
* <code>preprocess_for_retrosynthesis.py</code>: For preprocessing generated datasets
* <code>modeling_retrosynthesis_ecfp_split.py</code>: For modeling from reactants

Same parameter (-c) can be used in all of process script. If you already have retrosynthesized and preprocessed files, you can skip <code>generate_retrosynthesis.py</code> and <code>preprocess_for_retrosynthesis.py</code>. i.e. You can run <code>modeling_retrosynthesis_ecfp_split.py -c [your_config_file].json</code> by itself (stand-alone).

Our results can be replicated by this json file.

* <code>chembl_config_lv1.json</code>: Product-based splitting
* <code>chembl_config_lv1_augment.json</code>: Product-based splitting with data augmentation
* <code>chembl_config_lv2.json</code>: Reactant-based splitting
* <code>chembl_config_lv2_augment.json</code>: Reactant-based splitting with data augmentation

## 2. Screen reactants and combine by generated models

Reactant screening procedure. 

```
python reactant_screening.py -c [your_config_file].json
```

This script will run Thompson sampling used as a compared method. 

```
python reactant_screening_TS.py -c [your_config_file].json
```

Our results can be replicated by this json file.

* <code>chembl_config_for_screening_1k.json</code>: Reactant pair screening (sample 1k reactants for each, just for rate measurement)
* <code>chembl_config_for_screening_10k.json</code>: Reactant pair screening (sample 10k reactants for each, just for rate measurement)
* <code>chembl_config_for_screening_100k.json</code>: Reactant pair screening (sample 100k reactant for each)

If you want to obtain reactant pairs from your own reactant file, <code>heavy_atom_count</code> (number of heavy atoms of compound, easily calculated by RDKit) should be contained in your own reactant file.

## Analyze results
Results will be stored <code>outputs</code> directory.

```
outputs
|- datasets : retrosynthesized datasets
|- preprocessed : preprocessed datasets
|- prediction_level{n}[_augmented] : results of model construction
|- reactant_combination_level{n}[\_augmented]\_{m}[\_rc{l}] : proposed reactant pairs
```

Results you obtained can be analyzed by using <code>analysis.ipynb</code>.
