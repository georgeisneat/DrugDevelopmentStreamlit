import pickle
import datamaker
import pandas as pd
import streamlit as st
import deepchem as dc
from rdkit import Chem
import numpy as np
import enum

class DatasetSwitcher:

   def switch(self, int):

       default = "No dataset selected!"

       return getattr(self, 'Case_' + str(int), lambda: default)()

   def Case_0(self):
       featurizer = dc.feat.ConvMolFeaturizer(per_atom_fragmentation=False)
       tasks, datasets, transformers = dc.molnet.load_bbbp(featurizer='GraphConv')
       train_dataset, valid_dataset, test_dataset = datasets

       model = dc.models.GraphConvModel(n_tasks=1, mode='classification', dropout=0.2)
       model.fit(train_dataset, nb_epoch=100)

       return_type = 'single'

       return train_dataset, valid_dataset, test_dataset, model, featurizer, return_type

   def Case_1(self):
       featurizer = dc.feat.ConvMolFeaturizer(per_atom_fragmentation=False)
       tasks, datasets, transformers = dc.molnet.load_clintox(featurizer='GraphConv')
       train_dataset, valid_dataset, test_dataset = datasets

       model = dc.models.GraphConvModel(n_tasks=2, mode='classification', dropout=0.2)
       model.fit(train_dataset, nb_epoch=100)

       return_type = 'multi'

       return train_dataset, valid_dataset, test_dataset, model, featurizer, return_type

   def Case_2(self):

       return "Hello! It's Tuesday"

   def Case_3(self):

       return "Hello! It's Wednesday"

   def Case_4(self):

       return "Hello! It's Thursday"

   def Case_5(self):

       return "Hello! It's Friday"

   def Case_6(self):

       return "Hello! It's Saturday"

   def Case_7(self):

       return "Hello! It's Sunday"

class OutputSwitcher:

   def switch(self, int, ndarray, input_array):

       default = "No dataset selected!"

       return getattr(self, 'Case_' + str(int), lambda: default)(ndarray, input_array)

   def Case_0(self, ndarray, input_array):


       input = np.array([[i] for i in input_array])


       print(ndarray.shape)
       print(input.shape)
       new_array = np.hstack((ndarray.squeeze(),input))
       pandas_dataframe = pd.DataFrame(data=new_array, columns=['No BBBP', 'BBBP', 'Molecule'])

       #pandas_dataframe.assign(Molecule=input_array)


       dataframe = st.table(pandas_dataframe)

       return dataframe

   def Case_1(self, ndarray, input_index):

       pandas_dataframe = pd.DataFrame(data=ndarray, index=['Negative', 'Positive'], columns=['FDA_APPROVED', 'CT_TOX'])
       dataframe = st.table(pandas_dataframe)

       return dataframe

   def Case_2(self):

       return "Hello! It's Tuesday"

   def Case_3(self):

       return "Hello! It's Wednesday"

   def Case_4(self):

       return "Hello! It's Thursday"

   def Case_5(self):

       return "Hello! It's Friday"

   def Case_6(self):

       return "Hello! It's Saturday"

   def Case_7(self):

       return "Hello! It's Sunday"


dataset_dictionary = {
0 : "BBBP: Binary labels of blood-brain barrier penetration(permeability).",
1 : "ClinTox: Qualitative data of drugs approved by the FDA and those that have failed clinical trials for toxicity reasons." ,
2 : "Tox21: Qualitative toxicity measurements on 12 biological targets, including nuclear receptors and stress response pathways.",
3    : "ESOL: Water solubility data(log solubility in mols per litre) for common organic small molecules."
}

def set_dataset_selection(i):
    dataset_selection = i

    datasets = DatasetSwitcher()
    train_dataset, valid_dataset, test_dataset, model, featurizer,return_type = datasets.switch(i)


    return model, featurizer, return_type


def get_key(val):
    for key in dataset_dictionary.keys():
        if dataset_dictionary.get(key) == val:
            return key

    return None

def update_settings():
    train_dataset, valid_dataset, test_dataset = DatasetSwitcher()
    model = ModelSwitcher()

    return train_dataset, valid_dataset, test_dataset, model





with st.form(key="model_form"):
    selectbox = st.selectbox("Choose Dataset:",
                            [dataset_dictionary.get(0), dataset_dictionary.get(1), dataset_dictionary.get(2),
                            dataset_dictionary.get(3)])

    user_input = st.text_input("Input SMILES molecule(s) here:")
    submitted = st.form_submit_button("Load dataset, train model, and predict.")
    if submitted:

        #if selectbox in dataset_dictionary.values():
        input_array = user_input.split()

        loading_message = st.empty()
        loading_message.text('Training model... This takes a bit.')

        model, featurizer, return_type = set_dataset_selection(get_key(selectbox))




        loading_message.empty()

        output_switcher = OutputSwitcher()
        if return_type == 'multi':
            y = np.random.randint(2, size=(2 * len(input_array), 2 * len(input_array)))
            # Have to add dummy labels to input.
            # I don't know why it has to be this way but it does.
            # The size of Y has to be divisible by the number of input molecules.

            predicted_dataset = model.predict(dc.data.NumpyDataset(featurizer.featurize(input_array), y))

            for x in range(0, len(input_array)):
                title = st.write('`' + input_array[x] + '`')
                output = output_switcher.switch(get_key(selectbox), predicted_dataset[x], x)#change to array
        else:
            predicted_dataset = model.predict(dc.data.NumpyDataset(featurizer.featurize(input_array)))
            output = output_switcher.switch(get_key(selectbox), predicted_dataset, input_array)










@st.cache(allow_output_mutation=True)


def load_data():
    #switch(st.widget)
    tasks, datasets, transformers = dc.molnet.load_bbbp(featurizer='GraphConv')
    train_dataset, valid_dataset, test_dataset = datasets

    return train_dataset, valid_dataset, test_dataset

def setup_model():
    model = dc.models.GraphConvModel(n_tasks=1, mode='classification', dropout=0.2)
    model.fit(training_data, nb_epoch=100)
    return model


#training_data, valid_datatset, test_dataset = load_data()


#model = setup_model()

##input_dataset = ['CCC', 'CCN(CC)C(=O)C1CN(C2CC3=CNC4=CC=CC(=C34)C2=C1)C', '[Cl].CC(C)NCC(O)COc1cccc2ccccc12', 'C1CCN(CC1)Cc1cc(OCCCO)ccc1', 'CC(=O)Oc1ccccc1C(O)=O']
#?, ?, 1, 1, 0
##featurizer = dc.feat.ConvMolFeaturizer(per_atom_fragmentation=False)
#predicted_dataset = model.predict(dc.data.NumpyDataset(featurizer.featurize(input_dataset)))


##print(predicted_dataset.shape)
##print(predicted_dataset)
#   st.table(test_dataset)
##st.table(training_data.to_dataframe())



