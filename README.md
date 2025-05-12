# Screening by SVR-PK

This codeset works following below procedure.

## Prepare enviromment and datasets

Required packages are writtten on <code>_env/requirements.yml</code>.

Please create and activate conda environment by following command.

```
conda env create -f _env/requirements.yml
conda activate svr-pk
```

Datasets we used can be obtained from this [ZENODO](https://doi.org/10.5281/zenodo.14729011) repository. Please unzip after downloading and place the contents directory so that it has the following status.
```
SVR-PK
|- chembl31 : For model construction
|- emolecules : For screening
```


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

Our results can be replicated by this json file.

* <code>chembl_config_for_screening_1k.json</code>: Reactant pair screening (sample 1k reactants for each, just for rate measurement)
* <code>chembl_config_for_screening_10k.json</code>: Reactant pair screening (sample 10k reactants for each, just for rate measurement)
* <code>chembl_config_for_screening_100k.json</code>: Reactant pair screening (sample 100k reactant for each)

If you want to obtain reactant pairs from your own reactant file, <code>heavy_atom_count</code> (number of heavy atoms of compound, easily calculated by RDKit) should be contained in your own reactant file.

## 3. Method for comparison

Thompson sampling was used as the comparison method (https://pubs.acs.org/doi/10.1021/acs.jcim.3c01790).

The script was downloaded from [GitHub](https://github.com/PatWalters/TS) on 2024/6/7. Please download via GitHub, rename dirname <code>TS-main</code> to <code>TS_main_20240607</code> and save in <code>_benchmarking</code> directory.

The following change must be applied to reproduce our results.

Addition of the evaluation function (<code>ObjectiveEvaluatorByTanimotoKernel</code> in <code>TS_main_20240607/evaluators.py</code>) was implemented.
```
class ObjectiveEvaluatorByTanimotoKernel(Evaluator):
    """Added by iwmspy (09/06/2024)
        A evaluation class that calculates objective values (Such as inhibitation constants)
    """

    def __init__(self, input_dict):
        mod_path = input_dict['mod_path']
        reaction = input_dict['reaction_metaname']
        self.mod = pickle.load(open(mod_path,'rb'))
        self.vg  = lambda x: self.mod._var_gen(x,'smiles',False)
        self.svr = self.mod.ml_prd_[reaction].cv_models_['svr_tanimoto'].best_estimator_
        self.num_evaluations = 0

    @property
    def counter(self):
        return self.num_evaluations

    def evaluate(self, mol):
        self.num_evaluations += 1
        mdict = {'smiles':[Chem.MolToSmiles(mol)]}
        var   = self.vg(mdict)
        return self.svr.predict(var)[0]
```

Also, random seed was set to ensure reproducibility.

In <code>TS_main_20240607/disallow_tracker.py</code>,
```
rng_tr = np.random.default_rng(seed=0)  ## before class 'DisallowTracker'
...
selection_order = rng_tr.permutation(selection_order).tolist()  ## Inplace 'random.shuffle' in 'DisallowTracker.sample'
...
selection_candidate_scores = rng_tr.uniform(size=self._initial_reagent_counts[cycle_id])    ## Inplace 'selection_candidate_scores = np.random.uniform(size=self._initial_reagent_counts[cycle_id])' in 'DisallowTracker.sample'
```

In <code>TS_main_20240607/reagent.py</code>,
```
rng_rg = np.random.default_rng(seed=0)  ## before class Reagent
...
return rng_rg.normal(loc=self.current_mean, scale=self.current_std)  ## Inplace 'return np.random.normal(loc=self.current_mean, scale=self.current_std)' in 'Reagent.sample'
```

In <code>TS_main_20240607/thompson_sampling.py</code>,
```
rng_ts = np.random.default_rng(seed=0)  ## before class 'ThompsonSampler'
...
return rng_ts.choice(probs.shape[0], p=probs)  ## Inplace 'return np.random.choice(probs.shape[0], p=probs)' in 'ThompsonSampler._boltzmann_reweighted_pick'
...
selection_scores = rng_ts.uniform(size=reagent_count_list[p])   ## Inplace 'selection_scores = np.random.uniform(size=reagent_count_list[p])' in 'ThompsonSampler.warm_up'
```

The comparison method can be run by following command. 
```
python reactant_screening_TS.py -c [your_config_file].json
```

You can use the same <code>json</code> file used in our method for configuration.

## Analyze results
Results will be stored <code>outputs</code> directory.

```
SVR-PK
|- outputs
    |- datasets : retrosynthesized datasets
    |- preprocessed : preprocessed datasets
    |- prediction_level{n}[_augmented] : results of model construction
    |- reactant_combination_level{n}[\_augmented]\_{m}[\_rc{l}] : proposed reactant pairs
```

Results you obtained can be analyzed by using <code>analysis.ipynb</code>.
