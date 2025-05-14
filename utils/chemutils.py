import numpy as np
import pandas as pd
import re
import sys
import os
import copy
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, FilterCatalog
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmilesFromSmiles
from joblib import Parallel, delayed

if __name__=='__main__':
    pwd = os.path.dirname(os.path.abspath(__file__))
    workspace = os.path.abspath(os.path.join(pwd, os.pardir))
    sys.path.append(workspace)

from utils.SA_Score import sascorer
from retrosynthesis.retrosep import rdchiralReactants, rdchiralReaction, rdchiralRun

param = FilterCatalog.FilterCatalogParams()
param.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
filter = FilterCatalog.FilterCatalog(param)

cpus = os.cpu_count()

rng = np.random.default_rng(seed = 0)

class ReactionCenter:
    def __init__(self,sma_query,cumurate=True):
        self.frs = [AllChem.ReactionFromSmarts(f'{sma}>>{sma}') for sma in sma_query.split('.')]
        cent = 1000
        unmap = 900

        for fr_ in self.frs:
            for prd in fr_.GetProducts():
                for atom in prd.GetAtoms():
                    if atom.HasProp('molAtomMapNumber'):
                        atom.SetIntProp('Reaction_center',cent)
                        if cumurate:
                            cent += 1
                    else:
                        atom.SetIsotope(unmap)
                        if cumurate:
                            unmap += 1
                    atom.UpdatePropertyCache()
        return

    def SetReactionCenter(self, smi, fr=1, extend=False):
        mol = Chem.MolFromSmiles(smi)
        rxn = self.frs[fr-1]
        outcomes = rxn.RunReactants((mol,),)
        if len(outcomes)==0:
            return None
        outcomes_inrow = []
        for outcome in outcomes:
            for o in outcome:
                flag = True
                for atom in o.GetAtoms():
                    if atom.HasProp('Reaction_center'):
                        if atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
                            flag=False
                            break
                        atom.SetIsotope(int(atom.GetProp('Reaction_center')))
                if flag:outcomes_inrow.append(o)
                else:return None
        if outcomes_inrow:
            if not extend:
                return [Chem.MolToSmiles(outcome) for outcome in outcomes_inrow]
            if len(outcomes_inrow)!=1:
                return None
            return Chem.MolToSmiles(outcomes_inrow[0])

class reactor:
    def __init__(self,template:str,cumurate=True):
        temp_p, temp_r_ = template.split('>>')
        temp_r_mols = [Chem.MolFromSmarts(sma) for sma in  temp_r_.split('.')]
        cent = 1000
        oth = 900
        for mol in temp_r_mols:
            for atom in mol.GetAtoms():
                if atom.HasProp('molAtomMapNumber'):
                    atom.SetIsotope(cent)
                    if cumurate:
                        cent += 1
                else:
                    atom.SetIsotope(oth)
                    if cumurate:
                        oth += 1
        temp_r_smas = [Chem.MolToSmarts(sma) for sma in temp_r_mols]
        temp_r = '.'.join(temp_r_smas)
        self.template = '>>'.join([temp_r,temp_p])
        self.reverser = AllChem.ReactionFromSmarts(self.template)

    def reactor(self,mol_1:Chem.rdchem.Mol, mol_2:Chem.rdchem.Mol, return_mol=False):
        out = self.reverser.RunReactants((mol_1,mol_2),)
        assert len(out)==1
        out = out[0]
        self.prd_iso = '.'.join([Chem.MolToSmiles(mol) for mol in out])
        for o in out:
            for atom in o.GetAtoms():
                if int(atom.GetIsotope()) >= 1000:
                    atom.SetIsotope(0)
        if return_mol:return out
        prd = '.'.join([Chem.MolToSmiles(mol) for mol in out])
        return prd

def reset_isotopes(smiles):
    umap_mol = Chem.MolFromSmiles(smiles)
    for atom in umap_mol.GetAtoms():
        atom.SetIsotope(0)
    return Chem.MolToSmiles(umap_mol)

def contains_only_specific_elements(molecule):
    if molecule is not None:
        elements_to_keep = set(["C", "N", "O", "S", "P", "F", "Cl", "Br", "I"])
        for atom in molecule.GetAtoms():
            if atom.GetSymbol() not in elements_to_keep:
                return False
        return True
    return False
    
def ring_size_checker(molecule):
    if molecule is not None:
        rings = molecule.GetRingInfo()
        for ring in rings.AtomRings():
            if len(ring) >= 8:
                return False
        return True
    return False

def molwt_checker(molecule,thres=800):
    if molecule is not None:
        if rdMolDescriptors._CalcMolWt(molecule)<=thres:return True
    return False

def contains_only_one_frags(molecule):
    if len(Chem.GetMolFrags(molecule))==1:
        return True
    return False

def parser(p,par):
    if par:return p
    return False

def is_valid_molecule(smiles,method='all',parse=False):
    """method : choose 'specific', 'one', 'mwt', or 'ring'."""
    assert (isinstance(method, str) or isinstance(method, list))
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None : return parser('invalid')
    met = {'specific' : contains_only_specific_elements, 
           'one' : contains_only_one_frags,
           'ring' : ring_size_checker,
           'mwt' : molwt_checker}
    if method=='all':
        for m in met.values():
            if not m(molecule):return parser('condition not fulfilled',parse)
    elif isinstance(method, str):
        if not met[method](molecule):return parser('condition not fulfilled',parse)
    else:
        for key in method:
            if not met[key](molecule):return parser('condition not fulfilled',parse)
    return True

def MorganbitCalcAsVector(mol,rad=2,bits=8192,useChirality=True,split_components=False):
    if split_components:
        b_comps = list()
        for m in mol.GetMolFrags():
            b_comps.extend(MorganbitCalcAsVector(m,rad,bits,useChirality,split_components=False))
        return b_comps
    return list(AllChem.GetMorganFingerprintAsBitVect(mol,rad,bits,useChirality=useChirality))

def MorganbitCalcAsVectorFromSmiles(sma,rad=2,bits=8192,useChirality=True,split_components=False):
    if split_components:
        b_comps = list()
        for s in sma.split('.'):
            b_comps.extend(MorganbitCalcAsVectorFromSmiles(s,rad,bits,useChirality,split_components=False))
        return b_comps
    return list(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(sma),rad,bits,useChirality=useChirality))

def _MorganbitCalcMod(mol_list,rad,bits,useChirality,split_components):  
    return list(MorganbitCalcAsVector(mol,rad,bits,useChirality=useChirality,split_components=split_components) for mol in mol_list)

def _MorganbitCalcModFromSmiles(sma_list,rad,bits,useChirality,split_components):
    return list(MorganbitCalcAsVectorFromSmiles(sma,rad,bits,useChirality=useChirality,split_components=split_components) for sma in sma_list)

def MorganbitCalcAsVectors(l:list,rad=2,bits=8192,useRawSmiles=True,useChirality=True,n_jobs=1,split_components=False):
    ar = np.array(l)
    ar_split = np.array_split(ar,min(cpus,ar.shape[0]))
    if useRawSmiles:
        b = Parallel(n_jobs=n_jobs,backend="threading")(
            delayed(_MorganbitCalcModFromSmiles)(sma_list,rad,bits,useChirality=useChirality,split_components=split_components) 
            for sma_list in ar_split
            )
    else:
        b = Parallel(n_jobs=n_jobs,backend="threading")(
            delayed(_MorganbitCalcMod)(mol_list,rad,bits,useChirality=useChirality,split_components=split_components) 
            for mol_list in ar_split
            )
    return [x for row in b for x in row]

def TransReactantByTemplate(df: pd.DataFrame, index_col, product_col, reactant_col, template_col, reaction_col, objective_col, center_isotope=1000, leaving_isotope=900, **args):
    is_first = True
    for template, temp_df_ in df.groupby(template_col):
        temp_df = temp_df_.copy()
        rxn_template = rdchiralReaction(template)
        for idx, row in temp_df.iterrows():
            index   = row[index_col]
            prd     = row[product_col]
            prd_raw = row['Product_raw']
            rxn     = row[reaction_col]
            obj     = row[objective_col]
            for r in row.index:
                row[r] = None
            new_smis = rdchiralRun(rxn_template,rdchiralReactants(prd_raw),rc_cumurate=False,num_products=1,num_reactants=2)
            for new_smi_ in new_smis:
                new_smi = new_smi_[-1]
                if new_smi in temp_df_[reactant_col]: continue
                aug_row  = row.copy()
                aug_row[index_col] = index
                aug_row[product_col] = prd
                aug_row[reactant_col] = new_smi
                aug_row[template_col] = template
                aug_row[reaction_col] = rxn
                aug_row[objective_col] = obj
                aug_row['Product_raw'] = prd_raw
                aug_row['augmented']  = True
                temp_df = pd.concat([temp_df, pd.DataFrame(aug_row).T])
        temp_df.drop_duplicates(reactant_col,inplace=True)
        df_augmented = temp_df.copy() if is_first else pd.concat([df_augmented,temp_df])
        is_first = False
    return df_augmented

def MurckoScaffoldSmilesListFromSmilesList(smiles_list: list, n_jobs: int=-1, split_components: bool=False):
    l_split = np.array_split(np.array(smiles_list), min(cpus,len(smiles_list)))
    _mod_Murcko = lambda l: np.array([[MurckoScaffoldSmilesFromSmiles(smi)] for smi in l]) \
        if not split_components else np.array([[MurckoScaffoldSmilesFromSmiles(s) for s in smi.split('.')] for smi in l])
    ret = Parallel(n_jobs=n_jobs,backend='threading')(delayed(_mod_Murcko)(arr) for arr in l_split)
    return np.concatenate([x for x in ret])

def SmilesExtractor(tpath,smiles_col,idx_col,opath):
    df = pd.read_table(tpath,header=0,index_col=0,chunksize=100000)
    with open(opath,'w') as of:
        df_chunked = next(df)
        idx_list = df_chunked[idx_col].to_list()
        smi_list = df_chunked[smiles_col].to_list()
        joined   = [f'{smi}\t{idx}' for idx, smi in zip(idx_list,smi_list)]
        joined.append('')
        smi_str  = '\n'.join(joined)
        of.write(smi_str)
    with open(opath,'a') as of:
        for df_chunked in df:
            idx_list = df_chunked[idx_col].to_list()
            smi_list = df_chunked[smiles_col].to_list()
            joined   = [f'{smi}\t{idx}' for idx, smi in zip(idx_list,smi_list)]
            joined.append('')
            smi_str  = '\n'.join(joined)
            of.write(smi_str)

if __name__=='__main__':
    print(1)
