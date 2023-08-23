'''
Module to manage model keys in a structured manner.

This module contains the `ModelKeyCreator` class which facilitates the 
creation of unique model keys based on provided modeling instructions. These 
keys serve as unique identifiers for models and play a crucial role in 
tracking and referencing models throughout their lifecycle.

Example:
--------
    modeling_instructions = [
        {
            "features_source" : "opera", 
            "estimators" : ["estimatorA", "estimatorB"]
            },
        {
            "features_source" : "comptox", 
            "estimators" : ["estimatorA"]
            }
    ]
    
    key_creator = ModelKeyCreator(modeling_instructions)
    mapping = key_creator.create_identifier_key_mapping()
    print(mapping)
    
    # Output might look like:
    {
        '1': ('opera', 'estimatorA'), 
        '2': ('opera', 'estimatorB'), 
        '3': ('comptox', 'estimatorA')
        }
'''

import itertools

class ModelKeyCreator:
    '''
    A class to create unique model keys based on modeling instructions.
    
    Given a set of modeling instructions, this class can generate unique model 
    keys that identify specific configurations of models. Additionally, it 
    provides functionality to map these complex keys to simpler numeric 
    identifiers for easier referencing.

    Attributes
    ----------
    instructions : list of dict
        A list containing dictionaries with modeling instructions detailing 
        how models should be constructed and evaluated.

    Methods
    -------
    create_identifier_key_mapping():
        Generate a mapping from complex model keys to simpler numeric 
        identifiers.
    create_model_key(instruction, estimator_name):
        Construct a unique model key.
    create_model_key_names():
        Get the model key names based on the structure of the modeling 
        instructions.
    '''
    def __init__(self, modeling_instructions):
        '''
        Initialize the ModelKeyCreator.

        Parameters
        ----------
        modeling_instructions : list of dict
            A list containing dictionaries with modeling instructions. 
            Each dictionary should detail how a particular model should be 
            constructed and evaluated.
        '''
        self.instructions = modeling_instructions 

    def create_identifier_key_mapping(self):
        '''
        Generate a mapping from complex model keys to simpler numeric 
        identifiers.

        Returns
        -------
        dict
            A dictionary with simple numeric identifiers as keys and complex 
            model keys as values.
        '''
        mapping = {}  # initialize
        identifier_counter = itertools.count(start=1)  # Infinite counter

        for instruction in self.instructions:
            for estimator_name in instruction['estimators']:
                model_key = self.create_model_key(instruction, estimator_name)
                identifier = str(next(identifier_counter))
                mapping[identifier] = model_key
                
        return mapping

    def create_model_key(self, instruction, estimator_name):
        '''
        Construct a unique model key.

        Parameters
        ----------
        instruction : dict
            A dictionary containing details about how a model should be 
            constructed and evaluated.
        estimator_name : str
            Name of the estimator used in modeling.

        Returns
        -------
        tuple
            A unique key representing the model defined by the instruction and
            estimator name.
        '''
        model_key = [v for k, v in instruction.items() if k != 'estimators']
        model_key.append(estimator_name)
        return tuple(model_key)
    
    def create_model_key_names(self):
        '''
        Get the model key names based on the structure of the modeling 
        instructions.

        Returns
        -------
        list of str
            A list of strings representing the names of each part of the model 
            key.
        '''
        # Use the first instruction as a template
        sample_instruction = self.instructions[0]
        model_key_names = [
            k for k in sample_instruction.keys() 
            if k != 'estimators'
            ]
        model_key_names.append('estimator')
        return model_key_names