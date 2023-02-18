##########
### Imports

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdFMCS
from scipy.spatial import distance
from scipy.stats import pearsonr
from scipy.stats import fisher_exact
from scipy.stats.contingency import crosstab
from scipy.stats import hypergeom
from sklearn.manifold import TSNE
import random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pickle
import os


# from spec2vec import Spec2Vec
# from matchms import calculate_scores

# from openeye import oechem

os.chdir('../raw_data')


# write or load files

class Files:
    def __init__(self, filename):
        self.file = filename
        
    def write_to_file(self, data):
        with open(self.file, 'wb') as f:
            pickle.dump(data, f) 
        return None
    
    def load_pickle(self):
        data = pd.read_pickle(self.file)
        return data
    
    def load_csv(self, sep, usecols=None):
        data = pd.read_csv(self.file, sep=sep, usecols=usecols)
        return data
        
        

#########THE PREDICTOR HIMSELF
def predict(subject_df,query_df,dims,
            subject_embed='sdl_z2',
            metric='euc',
           query_embed='sdl_z1',
           top_k=20
           random_preds=False
           ): #both dfs should have z1 and z2 coloumns

    # initiate an empty df 
    preds = pd.DataFrame()
    preds.index=[x for x in query_df['spec_id']] # set query spec id as rownames
    preds[f'top_{top_k}_{metric}'] = None
    preds[f'true_{metric}'] = None
    preds[f'top_{top_k}_inchi14'] = None
    preds[f'top_{top_k}_tanis'] = None
    preds[f'true_tanimoto'] = None # only necessary to check that our indexing is correct; true always tanis == 1
    

    
    # compute the distances and select top k
    
    for query_index, query in enumerate(tqdm(query_df[query_embed])): #(query z1)
        
        #calculate tanimotos if it does not meet threshold;pass
        
        query_id = query_df['spec_id'].iloc[query_index]
        query_inchi = query_df['inchikey14'].iloc[query_index]
        
        similarity = []
        
        
        #print(query_inchi)
        
        for subject in subject_df[subject_embed]:#(subject z2)
        
            #subject = subject_df.loc[subject_index,'z2']
            if metric == 'corr':
                corr = pearsonr(query[:dims], subject[:dims])[0]
                similarity.append(corr)
        
            if metric== 'euc':
                euc = distance.euclidean(query[:dims], subject[:dims])
                similarity.append(euc)
        
            if metric == 'cos':
                cos = distance.cosine(query[:dims], subject[:dims])
                similarity.append(cos)
        
        if metric == 'corr': # if corr then higher the better
            top_k_corr = np.sort(similarity)[::-1][:top_k]
            top_k_ichi = []
            top_k_tanis = []
            for corr in top_k_corr:
                
                if random_preds:
                    sunject_index = random.randint(0, subject_df.shape[0])  
                    inchi = subject_df['inchikey14'].iloc[subject_index]
                    #print(inchi)
                    top_k_ichi.append(inchi)
                else:
                    subject_index = similarity.index(corr) # pick subject index
                    inchi = subject_df['inchikey14'].iloc[subject_index]
                    #print(inchi)
                    top_k_ichi.append(inchi)
                
                
                #compute tanis
                smile1 = subject_df['smiles'].iloc[subject_index] #extract the subject smile
                smile2= query_df['smiles'].iloc[query_index]#extract the query smile  
                tani = tanimoto(smile1,smile2)
                top_k_tanis.append(tani)
            
            # update the df
            preds.at[query_id, f'top_{top_k}_{metric}'] = top_k_corr # add corr
            preds.at[query_id, f'top_{top_k}_inchi14' ] = top_k_ichi # add inchi of the predicted
            preds.at[query_id, f'top_{top_k}_tanis'] = top_k_tanis
                
                
            #compute true dist between query and the true structure
            if query_inchi in list(subject_df['inchikey14']):
                
                #extract true hit embeddings to compute true cosine distance
                true_subject_embed = subject_df.loc[subject_df['inchikey14'] == query_inchi, subject_embed].iloc[0]
                
                true_corr = pearsonr(query[:dims], true_subject_embed[:dims])[0]
                preds.at[query_id,f'true_{metric}'] = true_corr
                
                # also compute true tanimoto
                smile2= query_df['smiles'].iloc[query_index]
                true_smile = subject_df.loc[subject_df['inchikey14'] == query_inchi, 'smiles'].iloc[0]
                preds.at[query_id,f'true_tanimoto'] = tanimoto(true_smile,smile2) # smile2 == query  smile
                
#             else: 
#                 pass
            
        
        else: # if cos or euc, then lower the better
            top_k_dist = np.sort(similarity)[:top_k]
            top_k_ichi = []
            top_k_tanis = []
            for dist in top_k_dist:
                
                if random_preds:
                    
                    subject_index =  random.randint(0, subject_df.shape[0]) #randomly pick a structure index
                    inchi = subject_df['inchikey14'].iloc[subject_index]
                    top_k_ichi.append(inchi)
                    
                else:
                    subject_index = similarity.index(dist)
                    #print(subject_index)
                    inchi = subject_df['inchikey14'].iloc[subject_index]
                    top_k_ichi.append(inchi)
                
                #compute tanis
                smile1 = subject_df['smiles'].iloc[subject_index] #extract the subject smile
                smile2= query_df['smiles'].iloc[query_index]#extract the query smile  
                tani = tanimoto(smile1,smile2)
                top_k_tanis.append(tani)
            
            preds.at[query_id, f'top_{top_k}_{metric}'] = top_k_dist # add corr
            preds.at[query_id, f'top_{top_k}_inchi14'] = top_k_ichi # add inchi of the predicted
            preds.at[query_id, f'top_{top_k}_tanis'] = top_k_tanis
                
            #compute true dist between query and the true structure
            if query_inchi in list(subject_df['inchikey14']):
                
                #extract true hit embedding to compute true cosine distance              
                true_subject_embed = subject_df.loc[subject_df['inchikey14'] == query_inchi, subject_embed].iloc[0]
                
                true_dist = distance.cosine(query[:dims], true_subject_embed[:dims])
                preds.at[query_id,f'true_{metric}'] = true_dist
                
                
                # also compute true tanimoto
                smile2= query_df['smiles'].iloc[query_index]
                true_smile = subject_df.loc[subject_df['inchikey14'] == query_inchi, 'smiles'].iloc[0]
                preds.at[query_id,f'true_tanimoto'] = tanimoto(true_smile,smile2) # smile2 == query  smile
                
#             else:
#                 pass
                
            
            
    
    return preds
    

#######TO COMPUTE TANIS
# function to calculate pairwise tanimoto scores
def tanimoto(smi1, smi2):
    #molecule
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    #fingerprint
    fp1 = Chem.RDKFingerprint(mol1)
    fp2 = Chem.RDKFingerprint(mol2)
    
    #similarity
    score = round(DataStructs.FingerprintSimilarity(fp1,fp2),4)
    return score


##### Compute predictions from the final model ¶

# load the data
paths = ['./sdl_logs/sdl_optimized_params/train_df_max3_sdl_and_cca_final_model_z_scores.pickle',
         './sdl_logs/sdl_optimized_params/test_df_max3_sdl_and_cca_final_model_z_scores.pickle',
         #'./sdl_logs/sdl_optimized_params/val_df_max3_sdl_and_cca_final_model_z_scores.pickle'
        ]


train_df = Files(paths[0]).load_pickle()
test_df = Files(paths[1]).load_pickle()
test_df.head(3)

# select unique inchikey14 and create a database from training set
train_df.drop_duplicates('inchikey14', inplace=True)
print(train_df.shape)



# #%%time
# metrics = ['cos', 'corr']

# models = ['sdl', 'cca']
# #size = 5
# for metric in metrics:
#     for model in models:
#         print(f'\nModel {model}')
#         # cosine distance
#         predictions_df = predict(subject_df=train_df,\
#                         query_df=train_df,dims=100,
#                         subject_embed=f'{model}_z2', # base name of z scores cols in subject df
#                        query_embed=f'{model}_z1', # base name of z scores cols in query df
#                        metric=metric,
#                                 top_k=20)
    
# #         print('\nComputing Distance is Completed successfully\n')
# #        # tanimotos and hits
# #         scores, hit = get_tanimotos(dist,subject_df=train_df,\
# #                                     query_df=test_df.head(size),\
# #                                     metric=metric)
    
#         print('\nComputing Tanimoto and hits is Completed successfully\n')
#         # write the distances to file
#         Files(f'./sdl_logs/sdl_optimized_params/train_in_train/{model}_preds/{model}_final_model_train_in_train_{metric}_predictions_df.pickle').write_to_file(predictions_df)
    
#         del dist # rescue memory :) 
    
#         Files(f'./sdl_logs/sdl_optimized_params/{model}_preds/{model}_final_model_test_{metric}_tanimoto.pickle').write_to_file(scores[0]) # scores has [tanimoto, mcs]
#         Files(f'./sdl_logs/sdl_optimized_params/{model}_preds/{model}_final_model_test_{metric}_hits.pickle').write_to_file(hit)
    
#         del scores, hit # only load when actually using them.


# for comparison, also compute random predictions.

predictions_random = predict(subject_df=train_df,\
                        query_df=test_df.head(5),dims=100,
                        subject_embed='sdl_z2', # base name of z scores cols in subject df
                       query_embed='sdl_z1', # base name of z scores cols in query df
                       metric='cos',
                      top_k=20,random_preds=True)

Files(f'./sdl_logs/sdl_optimized_params/test_in_train/sdl_preds/random_final_model_train_in_train_cos_predictions_df.pickle').write_to_file(predictions_random)
    



