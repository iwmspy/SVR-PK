'''Code for Retrosynthesis
    Do retrosynthesis to products using RDKit
    This code has been developped by reconstructing the following notebook:
    https://github.com/connorcoley/retrosim/blob/0a272f0b5de833c448f41491e81e4dc00b4d85b0/retrosim/scripts/usable_model.ipynb
    Original paper:
    https://pubs.acs.org/doi/10.1021/acscentsci.7b00355

    <Method>
    1. Calculate similarity between the input product and 
       products from USPTO-50k (and sort in order to it)
    2. Extract templates from USPTO-50k
       (Only upper 100 similarities)
    3. Iterate below method to 100 templates
        a. Fragment using extracted templates
        b. Calculate similarity between the generated 
           fragment and reactants from USPTO-50k
        c. Calculate overall similarity (1.*b.)
        d. If an item having same product and reactant(s)
           are already existing, compare overall similarities
        e. If not existing or overall similarity is higher,
           add this to the consequence list
'''

from __future__ import print_function
from rdkit.Chem.Draw import MolToImage
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
from rdkit import DataStructs
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import os
from IPython.display import display

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'retrosim'))

from generate_retro_templates_iwmspy import process_an_example
from main_iwmspy import rdchiralRun, rdchiralReaction, rdchiralReactants

def get_data_df(fpath='data_processed.csv'):
    return pd.read_csv(fpath)

data_ = get_data_df(os.path.join(os.path.dirname(__file__),'retrosim','retrosim','data','data_processed.csv'))

# curate template
for idx in data_.index:
    rxn = data_.loc[idx,'rxn_smiles'].split('>>')
    if len(rxn[1].split('.'))!=1 or len(rxn[0].split('.'))!=2:
        data_.at[idx,'keep']=False

data = data_[data_['keep']].copy()

similarity_metric = DataStructs.BulkTanimotoSimilarity # BulkDiceSimilarity or BulkTanimotoSimilarity
getfp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 2, useFeatures=True)

all_fps = []
for smi in tqdm(data['prod_smiles']):
    all_fps.append(getfp(smi))
data['prod_fp'] = all_fps

v = False
draw = False
debug = False
jx_cache = {}
   
def do_one(prod_smiles, draw=draw, debug=debug, v=v, rc_cumurate=True,
           ringinfo=True, num_products=1, num_reactants=2, strict_template=True):
    global jx_cache
    
    ex = Chem.MolFromSmiles(prod_smiles)
    rct = rdchiralReactants(prod_smiles)
    fp = getfp(prod_smiles)
    
    start_time = time.time()
    sims = similarity_metric(fp, [fp_ for fp_ in data['prod_fp']]) # calculate similarities between products want to divide and products from templates.
    print('took {:.3f} seconds to get similarity'.format(time.time() - start_time))
    js = np.argsort(sims,kind='mergesort')[::-1] # sort in order to similarities.

    if draw: 
        display(MolToImage(ex))
    
    # Get probability of precursors
    probs = {}
    outcomes_w_temp = {}
    
    start_time = time.time()
    
    # If similarities of compounds around 100th are equivalent, 
    # then these compounds are incorporated to consideration. 
    counter = 100
    while True:
        if sims[js[counter]]==sims[js[counter+1]]:
            counter += 1  
        else:
            break

    templates = []
    for j in js[:counter]: # use templates which have top 100 similarities to candidate products

        jx = data.index[j]

        if jx in jx_cache:
            (template, rcts_ref_fp) = jx_cache[jx]
        else:
            try:
                template = '(' + process_an_example(
                    data['rxn_smiles'][jx], super_general=True, ringinfo=ringinfo).replace('>>', ')>>')
            except Exception as e:
                if v: print(e)
                outcomes = []
                continue
            rcts_ref_fp = getfp(data['rxn_smiles'][jx].split('>')[0])
            jx_cache[jx] = (template, rcts_ref_fp)
            templates.append([template,data['rxn_smiles'][jx]])
            
        rxn = rdchiralReaction(template)

        # outcomes = rdchiralRun(rxn, rct, combine_enantiomers=False, rc_cumurate=rc_cumurate)
        try:
            outcomes = rdchiralRun(rxn, rct, rc_cumurate=rc_cumurate, 
                                   num_products=num_products, num_reactants=num_reactants,
                                   strict_template=strict_template)
        except Exception as e:
            if v: print(e)
            outcomes = []
            continue
            
        for prod,precursors_unmapped,precursors_mapped in outcomes:
            precursors_fp = getfp(precursors_unmapped)
            precursors_sim = similarity_metric(precursors_fp, [rcts_ref_fp])[0]
            overall_sim = precursors_sim * sims[j]
            outcomes_w_temp_tmp = [prod, precursors_mapped, template, data.loc[jx,'id'], 
                                   data.loc[jx,'class'], data.loc[jx,'prod_smiles'], 
                                   sims[j], data.loc[jx,'rxn_smiles'], 
                                   precursors_sim, overall_sim]
            if precursors_mapped in probs:
                if overall_sim > probs[precursors_mapped]:
                    probs[precursors_mapped] = overall_sim
                    outcomes_w_temp[precursors_mapped] = outcomes_w_temp_tmp
            else:
                probs[precursors_mapped] = overall_sim
                outcomes_w_temp[precursors_mapped] = outcomes_w_temp_tmp
            oc_sort = sorted(outcomes_w_temp.items())
            outcomes_w_temp_sort = dict((x,y) for x,y in oc_sort)
            outcome_list = [outcomes_w_temp_sort[key] for key in outcomes_w_temp_sort.keys()]

    print('took {:.3f} seconds to apply <= {} templates'.format(time.time() - start_time, counter))

    return outcome_list

def analysis_of_retrosynthesis(df:pd.DataFrame,name='retrosynthesis_summary.txt'):
    f = open(name,'w')
    f.write(f'==={name.split(".")[0]}=== \n \n')

    f.write('<Number of class appearances>\n')
    f.write(df['class'].value_counts().to_string())

    f.write('\n \n')
    
    f.write('<Number of template appearances> \n')
    f.write(df['template'].value_counts().to_string())

    f.close()

if __name__=='__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

 