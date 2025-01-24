'''
This script is a part of the retrosynthesis tool for designing functional molecules. 
It utilizes RDKit for chemical informatics and cheminformatics operations. 
The script defines several functions to handle chemical reactions, particularly focusing on chirality and isotopic mapping.

Key Functions:
- `is_stereo`: Checks if an atom or bond has stereochemistry.
- `is_cpds_exactly_number`: Checks if the number of compounds matches the specified number.
- `rdchiralRunText`: Runs a reaction from SMARTS and SMILES strings.
- `_MolMapping`: Handles isotopic mapping for molecules.
- `reactantsMapping`: Maps isotopes for reactants.
- `productMapping`: Maps isotopes for products.
- `rdchiralRun`: Main function to run the retrosynthesis reaction, handling chirality and isotopic mapping.

The script also includes a main block to demonstrate the usage of `rdchiralRunText` function.

This script is based on the original script from the retrosim repository by Connor Coley, which can be found at:
https://github.com/connorcoley/retrosim/blob/0a272f0b5de833c448f41491e81e4dc00b4d85b0/rdchiral/main.py

Contributions from the original script:
- The structure and logic of the functions `is_stereo`, `is_cpds_exactly_number`, `rdchiralRunText`, `_MolMapping`, `reactantsMapping`, `productMapping`, and `rdchiralRun` are derived from the original script.
- The handling of chirality and isotopic mapping is adapted from the original script.

Differences from the original script:
- The original script does not include the check for chiral centers in the reaction center.
- The original script does not skip subsequent processes if a chiral center is found in the reaction center.
- The handling of the 'ReactionCenter' property and the 'fr' property for atoms is an addition in this script.
- The script has been adapted to work within the retrosynthesis tool framework.
        
'''

from __future__ import print_function
import sys 
import os

import copy
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
from rdkit.Chem.rdchem import ChiralType,BondType, BondDir, BondStereo
from itertools import combinations

from retrosynthesis.retrosyn.rdchiral.utils import vprint, iterable
from retrosynthesis.retrosyn.rdchiral.initialization import rdchiralReaction, rdchiralReactants
from retrosynthesis.retrosyn.rdchiral.chiral import template_atom_could_have_been_tetra, copy_chirality, atom_chirality_matches
from retrosynthesis.retrosyn.rdchiral.clean import canonicalize_outcome_smiles, combine_enantiomers_into_racemic

def is_stereo(atom, bond=False):
    if atom.GetChiralTag()!=ChiralType.CHI_UNSPECIFIED:
        return True
    if bond:
        for b in atom.GetBonds():
            if b.GetStereo()!=BondStereo.STEREONONE:
                return True
    return False

def is_cpds_exactly_number(compounds,num=None):
    compounds_list = compounds.split('.') if isinstance(compounds, str) else compounds
    is_satisfy_num = len(compounds_list) == num
    is_satisfy_con = is_satisfy_num if num is not None else True
    return is_satisfy_con

def rdchiralRunText(reaction_smarts, reactant_smiles, **kwargs):
    '''Run from SMARTS string and SMILES string. This is NOT recommended
    for library application, since initialization is pretty slow. You should
    separately initialize the template and molecules and call run()'''
    rxn = rdchiralReaction(reaction_smarts)
    reactants = rdchiralReactants(reactant_smiles)
    return rdchiralRun(rxn, reactants, **kwargs)

def _MolMapping(tmp, mol, iso_num_mapped, iso_num_unmapped, invalid_chiral, demapping, rc_cumurate):
    '''Map isotopes for molecules
        <input>
        tmp: template molecule
        mol: target molecule
        iso_num_mapped: isotope number for mapped atoms
        iso_num_unmapped: isotope number for unmapped atoms
        invalid_chiral: flag to check for invalid chiral centers
        demapping: flag to demap isotopes
        rc_cumurate: flag to cumurate isotope numbers for reaction center atoms
        
        return: flag, mapped molecule
        '''
    mc = copy.deepcopy(mol)
    flag = False
    iso_mapped = iso_num_mapped
    iso_unmapped = iso_num_unmapped
    try:
        if iso_unmapped is not None:
            for atom_mol in mc.GetAtoms():
                if atom_mol.HasProp('ReactionCenter'):
                    if invalid_chiral:
                        assert not(is_stereo(atom_mol,bond=True)), 'Chiral atom was detected in reaction center !'
                    atom_mol.SetIsotope(iso_mapped)
                    if rc_cumurate: iso_mapped += 1
                elif atom_mol.HasProp('unmapped'):
                    atom_mol.SetIsotope(iso_unmapped)
                    if rc_cumurate: iso_unmapped += 1
                else:
                    if demapping:
                        atom_mol.SetIsotope(0)
        else:
            for atom_tmp in tmp.GetAtoms():
                for atom_mol in mc.GetAtoms():
                    if atom_mol.GetIsotope()==atom_tmp.GetIsotope():
                        if invalid_chiral:
                            assert not(is_stereo(atom_mol,bond=True)), 'Chiral atom was detected in reaction center !'
                        atom_mol.SetIntProp('ReactionCenter',atom_mol.GetIsotope())
                        atom_mol.SetIsotope(iso_mapped)
                        if rc_cumurate: iso_mapped += 1
            if demapping:
                for atom_mol in mc.GetAtoms():
                    if not atom_mol.HasProp('ReactionCenter'):
                        atom_mol.SetIsotope(0)
    except Exception as e:
        vprint(2, e)
        flag = True
        mc = None
        return flag, mc
    return flag, mc

def reactantsMapping(tmp,mol:Chem.rdchem.Mol,iso_num_mapped=1000,iso_num_unmapped=900,rc_cumurate=True):
    return _MolMapping(tmp,mol,iso_num_mapped,iso_num_unmapped,True,True,rc_cumurate)

def productMapping(tmp,mol:Chem.rdchem.Mol,iso_num=1000,rc_cumurate=True):
    return _MolMapping(tmp,mol,iso_num,None,True,True,rc_cumurate)

def rdchiralRun(rxn, 
                reactants, 
                rc_cumurate=True,
                num_products=None,
                num_reactants=None,
                strict_template=True):
    '''
    rxn = rdchiralReaction (rdkit reaction + auxilliary information)
    reactants = rdchiralReactants (rdkit mol + auxilliary information)

    note: there is a fair amount of initialization (assigning stereochem), most
    importantly assigning isotope numbers to the reactant atoms. It is 
    HIGHLY recommended to use the custom classes for initialization.
    '''
    if strict_template:
        prd_smarts, rct_smarts = rxn.reaction_smarts.split('>>')
        if not(is_cpds_exactly_number(prd_smarts,num_products) 
               and is_cpds_exactly_number(rct_smarts,num_reactants)):
            return list()
    
    if not is_cpds_exactly_number(reactants.reactant_smiles, num_products):
        return list()
    
    p_iso = 1000
    r_iso = 900

    final_outcomes = list()

    # We need to keep track of what map numbers 
    # (i.e., isotopes) correspond to which atoms
    # note: all reactant atoms must be mapped, so this is safe
    atoms_r = reactants.atoms_r

    # Copy reaction template so we can play around with isotopes
    template_r, template_p = rxn.template_r, rxn.template_p

    # Get molAtomMapNum->atom dictionary for tempalte reactants and products
    atoms_rt_map = rxn.atoms_rt_map
    atoms_pt_map = rxn.atoms_pt_map

    # Set property 'ReactionCenter' for subsequent analysis
    for product in rxn.rxn.GetProducts():
        for atom in product.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                atom.SetIntProp('ReactionCenter',atom.GetIntProp('molAtomMapNumber'))
    
    # chiral_centers = [atom.GetIdx() for atom in reactants.reactants.GetAtoms() if atom.GetChiralTag() != ChiralType.CHI_UNSPECIFIED]

    # if len(chiral_centers) > 0:
    #     pass
    ###############################################################################
    # Run naive RDKit on ACHIRAL version of molecules

    outcomes = rxn.rxn.RunReactants((reactants.reactants_achiral,))
    vprint(2, 'Using naive RunReactants, {} outcomes', len(outcomes))
    if not outcomes:
        return []
    
    ###############################################################################

    for outcome in outcomes:
        chicenter = False
        for i, cpd in enumerate(outcome):
            for atom in cpd.GetAtoms():
                if atom.HasProp('ReactionCenter'):
                    if is_stereo(reactants.reactants.GetAtomWithIdx(int(atom.GetIntProp('old_mapno')))):
                        chicenter = True
                atom.SetProp('fr',str(i))
        if chicenter:
            vprint(2,"Reactant '{}' has a chiral center in reaction center. Skip consequent processes.",
                   Chem.MolToSmiles(reactants.reactants))
            continue        
            
        ###############################################################################
        # Look for new atoms in products that were not in 
        # reactants (e.g., LGs for a retro reaction)
        vprint(2, 'Processing {}', str([Chem.MolToSmiles(x, True) for x in outcome]))
        unmapped = r_iso
        umd_counter = 0
        for m in outcome:
            for a in m.GetAtoms():
                # Assign "map" number via isotope
                if not a.GetIsotope():
                    a.SetIsotope(unmapped)
                    a.SetProp('unmapped',str(unmapped))
                    umd_counter += 1
                    unmapped    += 1
        vprint(2, 'Added {} map numbers to product', umd_counter)
        ###############################################################################


        ###############################################################################
        # Check to see if reactants should not have been matched (based on chirality)

        # Define isotope -> reactant template atom map
        atoms_rt =  {a.GetIsotope(): atoms_rt_map[a.GetIntProp('old_mapno')] \
            for m in outcome for a in m.GetAtoms() if a.HasProp('old_mapno')}

        # Set isotopes of reactant template
        # note: this is okay to do within the loop, because ALL atoms must be matched
        # in the templates, so the isotopes will get overwritten every time
        [a.SetIsotope(i) for (i, a) in atoms_rt.items()]

        # Make sure each atom matches
        if not all(atom_chirality_matches(atoms_rt[i], atoms_r[i]) for i in atoms_rt):
            vprint(2, 'Chirality violated! Should not have gotten this match')
            continue
        vprint(2, 'Chirality matches! Just checked with atom_chirality_matches')

        # Check bond chirality
        #TODO: add bond chirality considerations to exclude improper matches

        ###############################################################################



        ###############################################################################
        # Convert product(s) to single product so that all 
        # reactions can be treated as pseudo-intramolecular
        # But! check for ring openings mistakenly split into multiple
        # This can be diagnosed by duplicate map numbers (i.e., SMILES)

        isotopes = [a.GetIsotope() for m in outcome for a in m.GetAtoms() if a.GetIsotope()]
        if len(isotopes) != len(set(isotopes)): # duplicate?
            vprint(1, 'Found duplicate isotopes in product - need to stitch')
            # need to do a fancy merge
            merged_mol = Chem.RWMol(outcome[0])
            merged_iso_to_id = {a.GetIsotope(): a.GetIdx() for a in outcome[0].GetAtoms() if a.GetIsotope()}
            for j in range(1, len(outcome)):
                new_mol = outcome[j]
                for a in new_mol.GetAtoms():
                    if a.GetIsotope() not in merged_iso_to_id:
                        merged_iso_to_id[a.GetIsotope()] = merged_mol.AddAtom(a)
                for b in new_mol.GetBonds():
                    bi = b.GetBeginAtom().GetIsotope()
                    bj = b.GetEndAtom().GetIsotope()
                    vprint(10, 'stitching bond between {} and {} in stich has chirality {}, {}'.format(
                        bi, bj, b.GetStereo(), b.GetBondDir()
                    ))
                    if not merged_mol.GetBondBetweenAtoms(
                            merged_iso_to_id[bi], merged_iso_to_id[bj]):
                        merged_mol.AddBond(merged_iso_to_id[bi],
                            merged_iso_to_id[bj], b.GetBondType())
                        merged_mol.GetBondBetweenAtoms(
                            merged_iso_to_id[bi], merged_iso_to_id[bj]
                        ).SetStereo(b.GetStereo())
                        merged_mol.GetBondBetweenAtoms(
                            merged_iso_to_id[bi], merged_iso_to_id[bj]
                        ).SetBondDir(b.GetBondDir())
            outcome = merged_mol.GetMol()
            vprint(1, 'Merged editable mol, converted back to real mol, {}', Chem.MolToSmiles(outcome, True))
        else:
            for j in range(len(outcome)):
                for atom in outcome[j].GetAtoms():
                    atom.SetProp('fr',str(j))
            new_outcome = outcome[0]
            if not is_cpds_exactly_number(outcome,num_reactants):
                vprint(2, "Compound does not satisfy 'num_reactants'. skipped'", None)
                continue
            for j in range(1, len(outcome)):
                new_outcome = AllChem.CombineMols(new_outcome, outcome[j])
            outcome = new_outcome
        vprint(2, 'Converted all outcomes to single molecules')
        ###############################################################################

        # continue

        ###############################################################################
        # Figure out which atoms were matched in the templates
        # atoms_rt and atoms_p will be outcome-specific.
        atoms_pt = {a.GetIsotope(): atoms_pt_map[a.GetIntProp('old_mapno')] \
            for a in outcome.GetAtoms() if a.HasProp('old_mapno')}
        atoms_p = {a.GetIsotope(): a for a in outcome.GetAtoms() if a.GetIsotope()}

        # Set isotopes of product template
        # note: this is okay to do within the loop, because ALL atoms must be matched
        # in the templates, so the isotopes will get overwritten every time
        # This makes it easier to check parity changes
        [a.SetIsotope(i) for (i, a) in atoms_pt.items()]
        ###############################################################################



        ###############################################################################
        # Check for missing bonds. These are bonds that are present in the reactants,
        # not specified in the reactant template, and not in the product. Accidental
        # fragmentation can occur for intramolecular ring openings
        missing_bonds = []
        for (i, j, b) in reactants.bonds_by_isotope:
            if i in atoms_p and j in atoms_p:
                # atoms from reactant bond show up in product
                if not outcome.GetBondBetweenAtoms(atoms_p[i].GetIdx(), atoms_p[j].GetIdx()):
                    #...but there is not a bond in the product between those atoms
                    if (i not in atoms_rt or 
                        j not in atoms_rt or 
                        not template_r.GetBondBetweenAtoms(atoms_rt[i].GetIdx(), atoms_rt[j].GetIdx())):
                        # the reactant template did not specify a bond between those atoms (e.g., intentionally destroy)
                        missing_bonds.append((i, j, b))
        if missing_bonds:
            vprint(1, 'Product is missing non-reacted bonds that were present in reactants!')
            outcome_unmapped = Chem.MolToSmiles(outcome).split('.')
            outcome = Chem.RWMol(outcome)
            rwmol_iso_to_id = {a.GetIsotope(): a.GetIdx() for a in outcome.GetAtoms() if a.GetIsotope()}
            for (i, j, b) in missing_bonds:
                outcome.AddBond(rwmol_iso_to_id[i], rwmol_iso_to_id[j])
                new_b = outcome.GetBondBetweenAtoms(rwmol_iso_to_id[i], rwmol_iso_to_id[j])
                new_b.SetBondType(b.GetBondType())
                new_b.SetBondDir(b.GetBondDir())
                new_b.SetIsAromatic(b.GetIsAromatic())
            outcome = outcome.GetMol()
            if len(Chem.MolToSmiles(outcome).split('.'))!=outcome_unmapped:
                for out in Chem.GetMolFrags(outcome,asMols=True):
                    fr = set()
                    for atom in out.GetAtoms():
                        try:
                            fr.add(int(atom.GetProp('fr')))
                        except:
                            continue
                    fr = list(fr)
                    for atom_ in out.GetAtoms():
                        atom_.SetProp('fr',str(fr[0]))

        else:
            vprint(3, 'No missing bonds')
        ###############################################################################


        # Now that we've fixed any bonds, connectivity is set. This is a good time
        # to udpate the property cache, since all that is left is fixing atom/bond
        # stereochemistry.
        try:
            outcome.UpdatePropertyCache()
        except ValueError as e: 
            vprint(1, '{}, {}'.format(Chem.MolToSmiles(outcome, True), e))
            continue


        ###############################################################################
        # Correct tetra chirality in the outcome

        for a in outcome.GetAtoms():
            # Participants in reaction core (from reactants) will have old_mapno
            # Spectators present in reactants will have react_atom_idx
            # ...so new atoms will have neither!
            if not a.HasProp('old_mapno'):
                # Not part of the reactants template
                
                if not a.HasProp('react_atom_idx'):
                    # Atoms only appear in product template - their chirality
                    # should be properly instantiated by RDKit...hopefully...
                    vprint(4, 'Atom {} created by product template, should have right chirality', a.GetIsotope())
                
                else:
                    vprint(4, 'Atom {} outside of template, copy chirality from reactants', a.GetIsotope())
                    copy_chirality(atoms_r[a.GetIsotope()], a)
            else:
                # Part of reactants and reaction core
                
                if template_atom_could_have_been_tetra(atoms_rt[a.GetIsotope()]):
                    vprint(3, 'Atom {} was in rct template (could have been tetra)', a.GetIsotope())
                    
                    if template_atom_could_have_been_tetra(atoms_pt[a.GetIsotope()]):
                        vprint(3, 'Atom {} in product template could have been tetra, too', a.GetIsotope())
                        
                        # Was the product template specified?
                        
                        if is_stereo(atoms_pt[a.GetIsotope()]):
                            # No, leave unspecified in product
                            vprint(3, '...but it is not specified in product, so destroy chirality')
                            a.SetChiralTag(ChiralType.CHI_UNSPECIFIED)
                        
                        else:
                            # Yes
                            vprint(3, '...and product is specified')
                            
                            # Was the reactant template specified?
                            
                            if is_stereo(atoms_rt[a.GetIsotope()]):
                                # No, so the reaction introduced chirality
                                vprint(3, '...but reactant template was not, so copy from product template')
                                copy_chirality(atoms_pt[a.GetIsotope()], a)
                            
                            else:
                                # Yes, so we need to check if chirality should be preserved or inverted
                                vprint(3, '...and reactant template was, too! copy from reactants')
                                copy_chirality(atoms_r[a.GetIsotope()], a)
                                if not atom_chirality_matches(atoms_pt[a.GetIsotope()], atoms_rt[a.GetIsotope()]):
                                    vprint(3, 'but! reactant template and product template have opposite stereochem, so invert')
                                    a.InvertChirality()
                    
                    else:
                        # Reactant template chiral, product template not - the
                        # reaction is supposed to destroy chirality, so leave
                        # unspecified
                        vprint(3, 'If reactant template could have been ' +
                            'chiral, but the product template could not, then we dont need ' +
                            'to worry about specifying product atom chirality')

                else:
                    vprint(3, 'Atom {} could not have been chiral in reactant template', a.GetIsotope())
                    
                    if not template_atom_could_have_been_tetra(atoms_pt[a.GetIsotope()]):
                        vprint(3, 'Atom {} also could not have been chiral in product template', a.GetIsotope())
                        vprint(3, '...so, copy chirality from reactant instead')
                        copy_chirality(atoms_r[a.GetIsotope()], a)
                    
                    else:
                        vprint(3, 'Atom could/does have product template chirality!', a.GetIsotope())
                        vprint(3, '...so, copy chirality from product template')
                        copy_chirality(atoms_pt[a.GetIsotope()], a)
                    
            vprint(3, 'New chiral tag {}', a.GetChiralTag())
        vprint(2, 'After attempting to re-introduce chirality, outcome = {}',
            Chem.MolToSmiles(outcome, True))
        ###############################################################################


        ###############################################################################
        # Correct bond directionality in the outcome
        fr = set()
        frags_dict = {}
        skip=False

        flag,outcome = reactantsMapping(template_p,outcome,rc_cumurate=rc_cumurate)

        if flag:
            vprint(2, "Reactant '{}' has a chiral center in reaction center. "
                   "Skip consequent processes.", Chem.MolToSmiles(reactants.reactants))
            continue

        for fragment in Chem.GetMolFrags(outcome,asMols=True):
            a = set()
            for atom in fragment.GetAtoms():
                a.add(int(atom.GetProp('fr')))
            a = list(a)
            if a[0] in frags_dict:
                skip = True
            frags_dict[a[0]]=fragment
        if skip:
            vprint(2, 'Compound {} has confusable fragments. Skip', 
                   ".".join([Chem.MolToSmiles(m) for m in list(frags_dict.values())]))
            continue

        if not is_cpds_exactly_number(frags_dict, num_reactants):
            vprint(2, "Fragment(s) '{}' does not satisfy 'num_reactants'. Skip.",
                   ".".join([Chem.MolToSmiles(m) for m in list(frags_dict.values())]))
            continue
        
        frags_dict_unmapped = copy.deepcopy(frags_dict)
        
        for mol in frags_dict_unmapped.values():
            for atom in mol.GetAtoms():
                atom.SetIsotope(0)

        mols_smi = [Chem.MolToSmiles(mol) for mol in frags_dict.values()]
        mols_smi_unmapped = [Chem.MolToSmiles(mol) for mol in frags_dict_unmapped.values()]
        outcome = '.'.join(mols_smi)
        outcome_unmapped = '.'.join(mols_smi_unmapped)
        flag,re_mapped = productMapping(template_r,copy.deepcopy(reactants.reactants),rc_cumurate=rc_cumurate)
        if flag:
            vprint(2, "Reactant '{}' has a chiral center in reaction center. "
                   "Skip consequent processes.", Chem.MolToSmiles(reactants.reactants))
            continue
        final_outcomes.append([Chem.MolToSmiles(re_mapped),outcome_unmapped,outcome])
        ######################################################

    final_outcomes = [x for i, x in enumerate(final_outcomes) 
                      if x not in final_outcomes[:i]]
    return final_outcomes


if __name__ == '__main__':
    reaction_smarts = '[C:1][OH:2]>>[C:1][O:2][C]'
    reactant_smiles = 'CC(=O)OCCCO'
    outcomes = rdchiralRunText(reaction_smarts, reactant_smiles)
    print(outcomes)