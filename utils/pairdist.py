'''
'''

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

def pairdist(smi1,smi2=None,radius=2,bitlen=8192):
    if smi2==None:
        smi1,smi2 = smi1.split('.')
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    vec1 = AllChem.GetMorganFingerprintAsBitVect(mol1,radius=radius,nBits=bitlen)
    vec2 = AllChem.GetMorganFingerprintAsBitVect(mol2,radius=radius,nBits=bitlen)
    return 1 - DataStructs.TanimotoSimilarity(vec1,vec2)

if __name__=='__main__':
    print(pairdist('c1ccccc1','c1cnccc1'))