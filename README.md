# Designing_Functional_Molecules

This codeset works following below procedure.

## Prepare enviromment and datasets

Required packages are writtten on <code>_env/environments.yml</code>.

Please create conda environment by following command.

<p>
<code>conda env create -f _env/environments.yml</code><br>
</p>

Datasets we used can be obtained from [ChEMBL](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_31/) website. And commercially available reactants were obtained from [eMolecules](https://www.emolecules.com/) website.

## 0. Configuration
You can use your original datasets by creating config file and run below code with <code>-c [your config file].json</code> option. Please refer <code>config/chembl_config.json</code> to create your own config file. For testing code, we provide <code>config/example_config.json</code>. By this configuration, <code>example/tid-51_example.tsv</code> (for retrosynthesizing and modeling) and <code>config/emolecule_compounds_curated_sample_450k.tsv</code> (for obtaining reactants) will be used. Parameters can be set in json files are shown in below table.

|Parameter|Description|
|----|----|
|<code>files</code>|List of file names containing compounds and target objectives|
|<code>index_col</code>|Index column name of <code>files</code>|
|<code>model_smiles_col</code>|SMILES column name of <code>files</code>|
|<code>cand_path</code>|File name of reactant candidates|
|<code>cand_index_col</code>|Index column name of <code>cand_path</code>|
|<code>cand_smiles_col</code>|SMILES column name of <code>cand_path</code>|
|<code>split_level</code>|Level of train-test splitting. See paragraph|
|<code>reactions</code>|List of reaction names for generating compounds from reactants|
|<code>precalc</code>|Whether reactants are saved in specific files|

## 1. Generate models for screening reactants

Run below code.
<p><code>python rm_main.py -c [your_config_file].json</code></p>
This procedure contains below modules.

* <code>generate_retrosynthesis.py</code>: For generating reactants from actual molecules.
* <code>preprocess_for_retrosynthesis.py</code>: For preprocessing generated datasets.
* <code>modeling_retrosynthesis_ecfp_split.py</code>: For modeling from reactants.

## 2. Screen reactants and combine by generated models

Reactant screening procedure. 

<p><code>python reactant_screening.py -c [your_config_file].json</code></p>

## Analyze results

Results you obtained can be analyzed by using <code>analysis.ipynb</code>.