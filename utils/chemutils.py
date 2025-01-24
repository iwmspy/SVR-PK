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

def _replace(match):
    string, num = match.groups()
    return f"[{1000+int(num)-1}{string.upper()}:{num}]"

def colextend(l):
    if len(l)!=1:
        return False
    return l[0]

class ReactionCenter:
    def __init__(self,sma_query,cumurate=True):
        # pat = r'\[(.*?)\:(\d+)\]'
        # sma_q_map = re.sub(pat,_replace,sma_query)
        # fr1, fr2 = Chem.MolFromSmarts(sma_q_map.split('.')[0]),Chem.MolFromSmarts(sma_q_map.split('.')[1])
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
    
    def IterSetReactionCenter(self, smi_list, fr=1):
        outcomes_list = []
        for smi in smi_list:
            pre = self.SetReactionCenter(smi, fr=fr)
            outcomes_list.append(pre)
        return outcomes_list

def SetReactionCenter(sma_query):
    # pat = r'\[(.*?)\:(\d+)\]'
    # sma_q_map = re.sub(pat,_replace,sma_query)
    # fr1, fr2 = Chem.MolFromSmarts(sma_q_map.split('.')[0]),Chem.MolFromSmarts(sma_q_map.split('.')[1])
    fr1, fr2 = Chem.MolFromSmarts(sma_query.split('.')[0]),Chem.MolFromSmarts(sma_query.split('.')[1])

    unmap = 900
    for atom in fr1.GetAtoms():
        # print(f'{atom.GetSymbol()}:{atom.GetIsotope()}, {atom.GetPropsAsDict()}')
        if atom.HasProp('molAtomMapNumber'):
            atom.SetIsotope(1000+int(atom.GetProp('molAtomMapNumber'))-1)
        else:
            atom.SetIsotope(unmap)
            unmap += 1
        atom.UpdatePropertyCache()
    for atom in fr2.GetAtoms():
        # print(f'{atom.GetSymbol()}:{atom.GetIsotope()}, {atom.GetPropsAsDict()}')
        if atom.HasProp('molAtomMapNumber'):
            atom.SetIsotope(1000+int(atom.GetProp('molAtomMapNumber'))-1)
        else:
            atom.SetIsotope(unmap)
            unmap += 1
        atom.UpdatePropertyCache()

    print(f'{Chem.MolToSmarts(fr1)}.{Chem.MolToSmarts(fr2)}')

def removestereo(smiles):
    print(f'Converting {smiles}')
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        if atom.GetChiralTag() in [Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW]:
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    smiles_nostereo = Chem.MolToSmiles(mol)
    print(f"Converted to {smiles_nostereo}")
    return smiles_nostereo

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
    # if smiles is not None:
        # molecule = Chem.MolFromSmiles(smiles)
    if molecule is not None:
        elements_to_keep = set(["C", "N", "O", "S", "P", "F", "Cl", "Br", "I"])
        for atom in molecule.GetAtoms():
            if atom.GetSymbol() not in elements_to_keep:
                return False
        return True
    return False
    
def ring_size_checker(molecule):
    # if smiles is not None:
        # molecule = Chem.MolFromSmiles(smiles)
    if molecule is not None:
        rings = molecule.GetRingInfo()
        for ring in rings.AtomRings():
            if len(ring) >= 8:
                return False
        return True
    return False

def molwt_checker(molecule,thres=800):
    # if smiles is not None:
        # molecule = Chem.MolFromSmiles(smiles)
    if molecule is not None:
        if rdMolDescriptors._CalcMolWt(molecule)<=thres:return True
    return False

def contains_only_one_frags(molecule):
    # if smiles is not None:
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

def ChemAnalysis(mol):
    score_dict = {
        'sa_score' : sascorer.calculateScore(mol),
        'is_pains' : filter.HasMatch(mol),
    }
    return score_dict

def ChemAnalysisFromSmiles(smiles, flatten_list=False):
    mol = Chem.MolFromSmiles(smiles)
    analyzed = ChemAnalysis(mol)
    if flatten_list: return list(analyzed.values())
    return analyzed

def ChemAnalysisFromSmilesList(smiles_list, return_list=False):
    for i, smiles in enumerate(smiles_list):
        _analyzed = ChemAnalysisFromSmiles(smiles)
        if i==0:
            analyzed = {k : [] for k in _analyzed.keys()}
        for key, item in _analyzed.items():
            analyzed[key] = analyzed[key] + [item]
    if return_list: return list(analyzed.values())
    return analyzed

def ChemAnalysisFromMolList(mol_list, return_list=False):
    for i, mol in enumerate(mol_list):
        _analyzed = ChemAnalysis(mol)
        if i==0:
            analyzed = {k : [] for k in _analyzed.keys()}
        for key, item in _analyzed.items():
            analyzed[key] = analyzed[key] + [item]
    if return_list: return list(analyzed.values())
    return analyzed

def FlipFlopReactants(reactants_smiles: str, rsmarts: str):
    reactants 	= Chem.MolFromSmiles(reactants_smiles)

    # reactants of the smarts
    rcts = rsmarts.split('>>')[1].split('.')
    print('Extracted reactants:', rcts)

    # count number of atom mappings (must be 1)
    mapdetector = re.compile('\:[0-9]+\]')
    first_midx  = mapdetector.findall(rcts[0])
    second_midx = mapdetector.findall(rcts[1]) 
    assert len(first_midx) == len(second_midx) == 1

    # replace 2nd (1st) mapindx with the 1st (2nd)
    new_second  = re.sub(second_midx[0], first_midx[0], rcts[1])
    new_first   = re.sub(first_midx[0], second_midx[0], rcts[0])

    # if you want to apply separate molecules (not a single mol object), remove (): parenthesis.
    newquery = f'({rcts[0]}.{rcts[1]})>>{new_first}.{new_second}'
    rxn = AllChem.ReactionFromSmarts(newquery)

    flipfrags 	= rxn.RunReactants((reactants,))# do not why 8 patterns appear
    unique_flip_smi = np.unique(['.'.join([Chem.MolToSmiles(fr) for fr in frag]) for frag in flipfrags])
    assert len(unique_flip_smi) == 1, 'Only single replacement is allowed.'

    new_reactants = [Chem.MolFromSmiles(smi) for smi in unique_flip_smi[0].split('.')]
    # assign isotope label
    for i,new_reactant in enumerate(new_reactants):
        for rct in rcts:
            query = Chem.MolFromSmarts(rct)
            for qidx, aidx in enumerate(new_reactant.GetSubstructMatch(query)):
                atom  = new_reactant.GetAtomWithIdx(aidx)
                qatom = query.GetAtomWithIdx(qidx) 
                if qatom.GetAtomMapNum():
                    atom.SetIsotope(1000)
                else:
                    atom.SetIsotope(900)

    smi_flipfrags = '.'.join([Chem.MolToSmiles(new_reactant) for new_reactant in new_reactants])

    return smi_flipfrags

def direct_swapping_frags(rsmiles, rsmarts):
    icenter, ilg = 1000, 900
    reacts = Chem.MolFromSmiles(rsmiles)
    qreacts = rsmarts.split('>>')[1].split('.')

    # get reaction centers  (playing with index is better for RDKit because duplicates are created for the same atom object)
    rcenters=[atom for atom in reacts.GetAtoms() if atom.GetIsotope() == icenter]
    assert len(rcenters) == 2, 'Only two reaction centers are allowed'

    # get substructure per rcenter
    bonds = []
    for rcenter in rcenters:
        connect_atom_lg = [nei.GetIdx() for nei in rcenter.GetNeighbors() if nei.GetIsotope() == ilg]
        assert len(connect_atom_lg) == 1, 'attachment point between LG and rcenter must be specificed.'
        bonds.append((rcenter.GetIdx(), connect_atom_lg[0]))
    
    # flip the substructure
    emol    = Chem.EditableMol(reacts)
    emol.RemoveBond(bonds[0][0], bonds[0][1])
    emol.RemoveBond(bonds[1][0], bonds[1][1])
    emol.AddBond(bonds[0][0], bonds[1][1], order=Chem.rdchem.BondType.SINGLE)
    emol.AddBond(bonds[1][0], bonds[0][1], order=Chem.rdchem.BondType.SINGLE)
    mol = emol.GetMol()
    fmols = list(Chem.GetMolFrags(mol, asMols=True))

    match_atom_first  = fmols[0].GetSubstructMatch(Chem.MolFromSmarts(qreacts[0]))
    match_atom_second = fmols[0].GetSubstructMatch(Chem.MolFromSmarts(qreacts[1]))
    match_first_query, match_second_query = len(match_atom_first) != 0, len(match_atom_second) != 0
    
    if match_first_query and match_second_query: # need to identify which is the true fragment corresponding to the template
        for aidx in match_atom_first:
            if fmols[0].GetAtomWithIdx(aidx).GetIsotope()==0:
                match_first_query = False
        if match_first_query: # found in the first
            match_second_query = False
    
    if match_second_query:
        fmols[0], fmols[1] = fmols[1], fmols[0]
    
    return '.'.join([Chem.MolToSmiles(mol) for mol in fmols])

def ReactantAugmentationByTemplate(df: pd.DataFrame, index_col, product_col, reactant_col, template_col, reaction_col, objective_col, center_isotope=1000, leaving_isotope=900, **args):
    is_first = True
    for template, temp_df_ in df.groupby(template_col):
        temp_df = temp_df_.copy()
        rxncen   = set()
        tmp_obj_for_checking_unique  = AllChem.ReactionFromSmarts(template)
        tmp_obj_for_checking_unique.Initialize()
        rcts_obj_for_checking_unique = tmp_obj_for_checking_unique.GetProducts()
        for i, rct in enumerate(rcts_obj_for_checking_unique):
            for atom in rct.GetAtoms():
                if atom.HasProp('molAtomMapNumber'):
                    atom.SetIntProp('ReactionCenter', int(atom.GetProp('molAtomMapNumber')))
                    atom.ClearProp('molAtomMapNumber')
                    atom.SetIsotope(center_isotope)
                    rxncen.add(atom.GetSmarts())
                else:
                    atom.SetIsotope(leaving_isotope)
        subst_1, subst_2 = rcts_obj_for_checking_unique
        mol_1_to_2 = [copy.deepcopy(subst_1),copy.deepcopy(subst_2)]
        mol_2_to_1 = [copy.deepcopy(subst_2),copy.deepcopy(subst_1)]
        for mol in mol_1_to_2:
            for atom in mol.GetAtoms():
                if atom.HasProp('ReactionCenter'):
                    atom.SetIntProp('molAtomMapNumber',1)
        for mol in mol_2_to_1:
            for atom in mol.GetAtoms():
                if atom.HasProp('ReactionCenter'):
                    atom.SetIntProp('molAtomMapNumber',1)

        temp_df['augmented'] = False
        if len(rcts_obj_for_checking_unique)!=2 or len(rxncen)!=1:
            df_augmented = temp_df.copy() if is_first else pd.concat([df_augmented,temp_df])
            is_first = False
            continue

        for idx, row in temp_df.copy().iterrows():
            index = row[index_col]
            prd   = row[product_col]
            rxn   = row[reaction_col]
            rct   = row[reactant_col]
            obj   = row[objective_col]
            for r in row.index:
                row[r] = None
            # new_smi = FlipFlopReactants(rct, template)
            new_smi = direct_swapping_frags(rct,template)

            aug_row  = row.copy()
            aug_row[index_col] = index
            aug_row[product_col] = prd
            aug_row[reactant_col] = new_smi
            aug_row[template_col] = template
            aug_row[reaction_col] = rxn
            aug_row[objective_col] = obj
            if 'product_ECFP_col' in args: aug_row[args['product_ECFP_col']] = reset_isotopes(prd)
            aug_row['augmented']  = True
            temp_df = pd.concat([temp_df, pd.DataFrame(aug_row).T])
        temp_df.drop_duplicates(reactant_col,inplace=True)
        df_augmented = temp_df.copy() if is_first else pd.concat([df_augmented,temp_df])
        is_first = False
    return df_augmented

def TransReactantByTemplate(df: pd.DataFrame, index_col, product_col, reactant_col, template_col, reaction_col, objective_col, center_isotope=1000, leaving_isotope=900, **args):
    is_first = True
    for template, temp_df_ in df.groupby(template_col):
        temp_df = temp_df_.copy()
        # rxncen   = set()
        rxn_template = rdchiralReaction(template)
        # tmp_obj_for_checking_unique  = AllChem.ReactionFromSmarts(template)
        # tmp_obj_for_checking_unique.Initialize()
        # rcts_obj_for_checking_unique = tmp_obj_for_checking_unique.GetProducts()
        # for i, rct in enumerate(rcts_obj_for_checking_unique):
        #     for atom in rct.GetAtoms():
        #         if atom.HasProp('molAtomMapNumber'):
        #             atom.SetIntProp('ReactionCenter', int(atom.GetProp('molAtomMapNumber')))
        #             atom.ClearProp('molAtomMapNumber')
        #             atom.SetIsotope(center_isotope)
        #             rxncen.add(atom.GetSmarts())
        #         else:
        #             atom.SetIsotope(leaving_isotope)
        # subst_1, subst_2 = rcts_obj_for_checking_unique
        # mol_1_to_2 = [copy.deepcopy(subst_1),copy.deepcopy(subst_2)]
        # mol_2_to_1 = [copy.deepcopy(subst_2),copy.deepcopy(subst_1)]
        # for mol in mol_1_to_2:
        #     for atom in mol.GetAtoms():
        #         if atom.HasProp('ReactionCenter'):
        #             atom.SetIntProp('molAtomMapNumber',1)
        # for mol in mol_2_to_1:
        #     for atom in mol.GetAtoms():
        #         if atom.HasProp('ReactionCenter'):
        #             atom.SetIntProp('molAtomMapNumber',1)

        # temp_df['augmented'] = False
        # if len(rcts_obj_for_checking_unique)!=2 or len(rxncen)!=1:
        #     df_augmented = temp_df.copy() if is_first else pd.concat([df_augmented,temp_df])
        #     is_first = False
        #     continue

        for idx, row in temp_df.copy().iterrows():
            index   = row[index_col]
            prd     = row[product_col]
            prd_raw = row['Product_raw']
            rxn     = row[reaction_col]
            obj     = row[objective_col]
            for r in row.index:
                row[r] = None
            # new_smi = FlipFlopReactants(rct, template)
            # new_smi = direct_swapping_frags(rct,template)
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

def SmilesExtractor(tpath,smiles_col,idx_col,opath,downsize):
    if downsize is not None:
        size = -1
        with open(tpath,"r") as f:
            for _ in f.readlines():
                size += 1
        if size > downsize:
            random_dice = np.array([1] * downsize + [0] * (size - downsize))
        else:
            random_dice = np.array([1] * size)
        random_dice = random_dice.astype(bool)
        rng.shuffle(random_dice)
        endpoint = 0
    df = pd.read_table(tpath,header=0,index_col=0,chunksize=CHUNK)
    with open(opath,'w') as of:
        df_chunked = next(df)
        if downsize is not None:
            chsize = df_chunked.shape[0]
            df_chunked = df_chunked[random_dice[endpoint:endpoint+chsize]]
            endpoint += chsize
        if not df_chunked.empty:
            idx_list = df_chunked[idx_col].to_list()
            smi_list = df_chunked[smiles_col].to_list()
            joined   = [f'{smi}\t{idx}' for idx, smi in zip(idx_list,smi_list)]
            joined.append('')
            smi_str  = '\n'.join(joined)
            of.write(smi_str)
    with open(opath,'a') as of:
        for df_chunked in df:
            if downsize is not None:
                chsize = df_chunked.shape[0]
                df_chunked = df_chunked[random_dice[endpoint:endpoint+chsize]]
                endpoint += chsize
            if not df_chunked.empty:
                idx_list = df_chunked[idx_col].to_list()
                smi_list = df_chunked[smiles_col].to_list()
                joined   = [f'{smi}\t{idx}' for idx, smi in zip(idx_list,smi_list)]
                joined.append('')
                smi_str  = '\n'.join(joined)
                of.write(smi_str)



if __name__=='__main__':
    # SetReactionCenter("O-[C;H0;+0:1].[NH2;+0:2]")
    template = '[C;H0;+0:1]-[NH;+0:2]>>O-[C;H0;+0:1].[NH2;+0:2]'
    rx = ReactionCenter(template.split('>>')[1])
    smi_1 = 'CC(C)(C)O'
    smi_2 = 'CCCCN'
    ret_1 = rx.SetReactionCenter(smi_1,fr=1)
    ret_2 = rx.SetReactionCenter(smi_2,fr=2)
    # print(ret)
    rc = reactor(template)
    rc.reactor(Chem.MolFromSmiles(ret_1[0]),Chem.MolFromSmiles(ret_2[0]))