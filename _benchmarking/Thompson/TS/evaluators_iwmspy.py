"""
Module: evaluators_iwmspy.py

This module defines a custom evaluator class for calculating objective values, such as inhibition constants, 
using a Tanimoto kernel-based Support Vector Regressor (SVR). The evaluator is designed to work with molecular 
data and integrates with the RDKit library for handling molecular structures.

### Classes:

1. **ObjectiveEvaluatorByTanimotoKernel**:
    - A subclass of `Evaluator` that evaluates molecular objective values using a pre-trained SVR model.
    - The model is loaded from a file specified in the input dictionary.

### Class: ObjectiveEvaluatorByTanimotoKernel

#### Description:
    This class evaluates objective values (e.g., inhibition constants) for molecules using a Tanimoto kernel-based 
    SVR model. It tracks the number of evaluations performed and provides a method to predict objective values 
    for a given molecule.

#### Methods:
1. **`__init__(self, input_dict)`**:
    - Initializes the evaluator with a pre-trained model and reaction metadata.
    - **Parameters**:
        - `input_dict` (dict): A dictionary containing:
            - `'mod_path'`: Path to the pre-trained model file.
            - `'reaction_metaname'`: Name of the reaction metadata to use.
    - **Attributes**:
        - `self.mod`: The loaded model object.
        - `self.vg`: A lambda function for generating molecular descriptors.
        - `self.svr`: The Tanimoto kernel-based SVR model for predictions.
        - `self.num_evaluations`: Counter for the number of evaluations performed.

2. **`counter(self)`**:
    - A property that returns the number of evaluations performed.
    - **Returns**:
        - `int`: The number of evaluations.

3. **`evaluate(self, mol)`**:
    - Evaluates the objective value for a given molecule.
    - **Parameters**:
        - `mol` (rdkit.Chem.Mol): An RDKit molecule object to evaluate.
    - **Returns**:
        - `float`: The predicted objective value for the molecule.
    - **Details**:
        - Converts the molecule to a SMILES string.
        - Generates molecular descriptors using the pre-trained model's descriptor generator.
        - Predicts the objective value using the SVR model.
        - Increments the evaluation counter.

### Dependencies:
- `pickle`: For loading the pre-trained model.
- `rdkit.Chem`: For handling molecular structures and converting them to SMILES strings.
- `TS.evaluators.Evaluator`: The base class for the custom evaluator.

### Example Usage:
```python
from evaluators_iwmspy import ObjectiveEvaluatorByTanimotoKernel
from rdkit import Chem

# Input dictionary with model path and reaction metadata
input_dict = {
    'mod_path': 'path/to/model.pkl',
    'reaction_metaname': 'reaction_name'
}

# Initialize the evaluator
evaluator = ObjectiveEvaluatorByTanimotoKernel(input_dict)

# Create an RDKit molecule
mol = Chem.MolFromSmiles('CCO')

# Evaluate the molecule
objective_value = evaluator.evaluate(mol)
print(f"Objective Value: {objective_value}")

# Get the number of evaluations performed
print(f"Number of Evaluations: {evaluator.counter}")
"""
from abc import ABC, abstractmethod
import pickle
from rdkit import Chem

class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, mol):
        pass

    @property
    @abstractmethod
    def counter(self):
        pass

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

