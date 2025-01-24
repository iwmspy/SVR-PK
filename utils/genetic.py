"""GA"""

import random
import pandas as pd
import numpy as np
from tqdm import tqdm

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, DataStructs, MACCSkeys
from sklearn.model_selection import train_test_split,GridSearchCV,KFold
from sklearn.ensemble import RandomForestRegressor


class genetic:
    def __init__(self,df:pd.DataFrame,method='sim_ecfp',
                 obj_col=None,size=5,ngen=50,pop=50,cxpb=0.5,mutpb=0.3,seed=0):
        self.df = df
        self.size = size
        self.ngen = ngen
        self.pop = pop
        self.cxpb = cxpb
        self.mutpb = mutpb
        if obj_col != None:
            if isinstance(obj_col,str):
                self.obj_col = df[obj_col].to_list()
            else:
                self.obj_col = obj_col
        self.method = method
        self.num_of_compounds = df.shape[0]
        if self.num_of_compounds%2==1:
            raise Exception
        self.num_of_selecta = int(self.num_of_compounds/2)
        random.seed(seed)
        self.seed = seed

        # if isinstance(sel,bool):
        #     if sel:
        #         rx_fr1 = '([900*,901*]-[c;H0;+0:1])>>Br-[c;H0;+0:1]'
        #         rx_fr2 = '([900*,901*]-[c;H0;+0:1])>>O-B(-O)-[c;H0;+0:1]'
        #     else:
        #         rx_fr1 = '([900*,901*]-[c;H0;+0:1])>>F-C(-F)(-F)-S(=O)(=O)-O-[c;H0;+0:1]'
        #         rx_fr2 = '([900*,901*]-[c;H0;+0:1])>>O-B(-O)-[c;H0;+0:1]'
        # else:
        #     rx_fr1 = sel[0]
        #     rx_fr2 = sel[1]

        # self.rxn_fr1 = AllChem.ReactionFromSmarts(rx_fr1)
        # self.rxn_fr2 = AllChem.ReactionFromSmarts(rx_fr2)
        # self.rxn_list = [self.rxn_fr1,self.rxn_fr2]

    def select_fragments(self,bits_list:list,col_list=[0,1]):
        lista = []
        self.selecta = []
        for s in range(self.num_of_selecta):
            self.selecta.append(s*2+bits_list[s])
        # selecta = selecta + bits_list
        for idx in self.selecta:
            lista.append([self.df.iloc[idx,col_list[0]],self.df.iloc[idx,col_list[1]]])
        return lista

    def obj(self,ind,col_list=[0,1]):
        fr_list = self.select_fragments(ind,col_list=col_list)
        # for i, bit in enumerate(ind):
        #     if bit:
        #         fr_list.append([self.df.iloc[i,col_list[1]],self.df.iloc[i,col_list[0]]])
        #     else:
        #         fr_list.append([self.df.iloc[i,col_list[0]],self.df.iloc[i,col_list[1]]])
        self.calculate_ecfp(fr_list)
        fr_1 = np.array(self.ecfp_list_1)
        fr_2 = np.array(self.ecfp_list_2)
        fr_1_mean = np.mean(fr_1,axis=0)
        fr_2_mean = np.mean(fr_2,axis=0)
        fr_1_dist = np.array([np.sum((j-fr_1_mean)**2) for j in fr_1])
        fr_2_dist = np.array([np.sum((j-fr_2_mean)**2) for j in fr_2])
        return np.sum(fr_1_dist)+np.sum(fr_2_dist),

    def obj_(self,ind,col_list=[0,1]):
        fr_list = self.select_fragments(ind,col_list=col_list)
        # for i, bit in enumerate(ind):
        #     if bit:
        #         fr_list.append([self.df.iloc[i,col_list[1]],self.df.iloc[i,col_list[0]]])
        #     else:
        #         fr_list.append([self.df.iloc[i,col_list[0]],self.df.iloc[i,col_list[1]]])
        self.calculate_ecfp(fr_list)
        ecfp_list = []
        for h, o in enumerate(self.obj_col):
            ecfp_list_tmp = []
            for g in self.ecfp_list_1[h]:
                ecfp_list_tmp.append(g)
            for g in self.ecfp_list_2[h]:
                ecfp_list_tmp.append(g)
            ecfp_list_tmp.append(o)
            ecfp_list.append(ecfp_list_tmp)
        ecfp_list = np.array(ecfp_list)
        train,test = train_test_split(ecfp_list,test_size=0.2,random_state=self.seed)
        train_x,train_y = train[:,:-2],train[:,-1]
        test_x,test_y = test[:,:-2],test[:,-1]
        parameters = {'n_estimators' : [10, 100, 500]}
        gcv = GridSearchCV(
            estimator = RandomForestRegressor(random_state=self.seed),
            param_grid = parameters,
            scoring = "r2",
            cv=KFold(n_splits=5, shuffle=True, random_state=self.seed),
            n_jobs = -1,
            )
        gcv.fit(train_x,train_y)
        est = gcv.best_estimator_
        return est.score(test_x,test_y),

    def obj_maccs(self,ind,col_list=[0,1]):
        fr_list = self.select_fragments(ind,col_list=col_list)
        # for i, bit in enumerate(ind):
        #     if bit:
        #         fr_list.append([self.df.iloc[i,col_list[1]],self.df.iloc[i,col_list[0]]])
        #     else:
        #         fr_list.append([self.df.iloc[i,col_list[0]],self.df.iloc[i,col_list[1]]])
        self.calculate_ecfp(fr_list)
        fr_1 = np.array(self.ecfp_list_1)
        fr_2 = np.array(self.ecfp_list_2)
        fr_1_mean = np.mean(fr_1,axis=0)
        fr_2_mean = np.mean(fr_2,axis=0)
        fr_1_dist = np.array([np.sum((j-fr_1_mean)**2) for j in fr_1])
        fr_2_dist = np.array([np.sum((j-fr_2_mean)**2) for j in fr_2])
        return np.sum(fr_1_dist)+np.sum(fr_2_dist),

    def ga_flip(self,):
        if self.method=="sim_ecfp":
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMin)
        elif self.method=="pki":
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)
        elif self.method=="sim_maccs":
            creator.create("FitnessMin", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMin)
        else:
            print("warning : You chose irregal option. 'sim_ecfp' is automatically selected.")
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMin)        

        toolbox = base.Toolbox()
        toolbox.register("attribute", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, 
                         creator.Individual, toolbox.attribute, 
                         n=self.num_of_selecta)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("select", tools.selTournament, tournsize=5)
        toolbox.register("mate", tools.cxOnePoint)
        toolbox.register("mutate", tools.mutUniformInt,low=0,up=1,indpb=self.mutpb)
        
        if self.method=="sim_ecfp":
            toolbox.register("evaluate", self.obj)
        elif self.method=="pki":
            toolbox.register("evaluate", self.obj_)
        elif self.method=="sim_maccs":
            toolbox.register("evaluate", self.obj_maccs)
        else:
            toolbox.register("evaluate", self.obj)

        pop = toolbox.population(n=self.pop)
        for indv in tqdm(pop):
            indv.fitness.values = toolbox.evaluate(indv)
        hof=tools.ParetoFront()
        algorithms.eaSimple(pop,toolbox,cxpb=self.cxpb,mutpb=self.mutpb,
                            ngen=self.ngen,halloffame=hof)
        
        best_ind = tools.selBest(pop, 1)[0]

        # new_df = self.df.copy()
        # for i, bit in enumerate(best_ind):
        #     if bit:
        #         tmp = new_df.iloc[i,0]
        #         new_df.iloc[i,0] = new_df.iloc[i,1]
        #         new_df.iloc[i,1] = tmp
        # new_smiles = new_df.to_numpy().tolist()
        new_smiles = self.select_fragments(best_ind)
        self.edit_smiles(new_smiles)
        new_smiles_post = [[Chem.MolToSmiles(self.mol_list_1[h]),
                            Chem.MolToSmiles(self.mol_list_2[h])] 
                            for h in range(len(self.mol_list_1))]
        new_df_ = pd.DataFrame(new_smiles_post,columns=self.df.iloc[self.selecta,:].columns,
                               index=self.df.iloc[self.selecta,:].index)
        return new_df_


    def calculate_ecfp(self, smiles_list):
        self.ecfp_list_1 = []
        self.ecfp_list_2 = []
        self.edit_smiles(smiles_list)
        for mol in self.mol_list_1:
            ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=8192)
            self.ecfp_list_1.append([h for h in ecfp])
        for mol in self.mol_list_2:
            ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=8192)
            self.ecfp_list_2.append([h for h in ecfp])
        

    def calculate_maccs(self, smiles_list):
        self.maccs_list_1 = []
        self.maccs_list_2 = []
        self.edit_smiles(smiles_list)
        for mol in self.mol_list_1:
            maccs = MACCSkeys.GenMACCSKeys(mol)
            self.maccs_list_1.append([h for h in maccs])
        for mol in self.mol_list_2:
            maccs = MACCSkeys.GenMACCSKeys(mol)
            self.maccs_list_2.append([h for h in maccs])

    
    def edit_smiles(self,smiles_list:list):
        self.mol_list_1 = []
        self.mol_list_2 = []
        for smiles in smiles_list:
            # mol_1 = Chem.MolFromSmiles(smiles[0])
            # mol_2 = Chem.MolFromSmiles(smiles[1])
            # ret_1 = self.rxn_fr1.RunReactants((mol_1,))
            # ret_2 = self.rxn_fr2.RunReactants((mol_2,))
            # r_1 = Chem.MolToSmiles(ret_1[0][0])
            # r_2 = Chem.MolToSmiles(ret_2[0][0])
            # self.mol_list_1.append(Chem.MolFromSmiles(r_1))
            # self.mol_list_2.append(Chem.MolFromSmiles(r_2))
            self.mol_list_1.append(Chem.MolFromSmiles(smiles[0]))
            self.mol_list_2.append(Chem.MolFromSmiles(smiles[1]))
            
        

if __name__=="__main__":
    smiles_list = ['[900*]c1ccc(C(F)(F)F)cc1.[901*]c1ccc2c(-c3ccc(C4CCN(CCCCc5cn(CCCCCCN)nn5)CC4)cc3)cc(C(=O)O)cc2c1', 
                   '[900*]c1ccccc1.[901*]c1ccccc1N1CCN(CCCCCCC(=O)N2Cc3ccccc3C[C@H]2C(=O)N2CCCCC2)CC1', 
                   '[900*]c1ccccc1.[901*]c1ccccc1N1CCN(CCCCCC(=O)N2Cc3ccccc3C[C@H]2C(=O)N2CCCCC2)CC1', 
                   '[900*]c1ccccc1.[901*]c1ccccc1N1CCN(CCCCC(=O)N2Cc3ccccc3C[C@H]2C(=O)N2CCCCC2)CC1', 
                   '[900*]c1c(=O)n(CCCCN2CCCC(c3c[nH]c4ccc(OC)cc34)C2)c(=O)n2ccccc12.[901*]c1ccccc1Cl', 
                   '[900*]c1c(=O)n(CCCCN2CCCC(c3c[nH]c4ccc(OC)cc34)C2)c(=O)n2ccccc12.[901*]c1ccc(Cl)cc1', 
                   '[900*]c1c(=O)n(CCCCN2CCCC(c3c[nH]c4ccc(OC)cc34)C2)c(=O)n2ccccc12.[901*]c1ccc(OC)cc1', 
                   '[900*]c1c(=O)n(CCCCN2CCCC(c3c[nH]c4ccc(OC)cc34)C2)c(=O)n2ccccc12.[901*]c1ccccc1OC', 
                   '[900*]c1ccc(C(=O)NCCC(CN2CCN(c3cccc(Cl)c3Cl)CC2)OC(C)=O)cc1.[901*]c1ccccn1', 
                   '[900*]c1c2n(c(=O)n(CCCCN3CCCC(c4c[nH]c5ccc(OC)cc45)C3)c1=O)CCCC2.[901*]c1ccc(C)cc1', 
                   '[900*]c1c(=O)n(CCCCN2CCCC(c3c[nH]c4ccc(F)cc34)C2)c(=O)n2ccccc12.[901*]c1ccccc1OC', 
                   '[900*]c1c(=O)n(CCCCN2CCCC(c3c[nH]c4ccc(F)cc34)C2)c(=O)n2ccccc12.[901*]c1ccc(OC)cc1', 
                   '[900*]c1c(=O)n(CCCCN2CCCC(c3c[nH]c4ccc(OC)cc34)C2)c(=O)n2ccccc12.[901*]c1ccccc1F', 
                   '[900*]c1c(=O)n(CCCCN2CCCC(c3c[nH]c4ccc(OC)cc34)C2)c(=O)n2ccccc12.[901*]c1ccc(F)cc1', 
                   '[900*]c1ccc(CNCCc2ccccc2)cc1.[901*]c1ccc(CN(Cc2cccnc2)C(=O)/C=C/c2ccccc2)cc1', 
                   '[900*]c1c(=O)n(CCCCN2CCCC(c3c[nH]c4ccc(OC)cc34)C2)c(=O)n2ccccc12.[901*]c1ccc(C)cc1', 
                   '[900*]c1c(=O)n(CCCCN2CCCC(c3c[nH]c4ccc(OC)cc34)C2)c(=O)n2ccccc12.[901*]c1ccccc1C', 
                   '[900*]c1ccccc1N1CCN(CCCCCC(=O)NCc2ccc(OS(C)(=O)=O)cc2)CC1.[901*]c1ccccc1', 
                   '[900*]c1ccc(OC)cc1.[901*]c1ccccc1N1CCN(C[C@@H](O)COc2ccc(-c3cn4ccccc4n3)cc2)CC1', 
                   '[900*]c1ccc(OC)cc1.[901*]c1ccccc1C1CCN(C[C@@H](O)COc2ccc(-c3cn4ccccc4n3)cc2)CC1']
    
    smiles_list = [smi.split('.') for smi in smiles_list]
    df = pd.DataFrame(smiles_list, columns=['fr1','fr2'])
    gn = genetic(df,True,method='sim_maccs')
    new_df = gn.ga_flip()

    new_df