

![image](https://user-images.githubusercontent.com/71344715/213717024-8e8cd437-8600-47fd-bacb-d653eebf0793.png)

This project aims at predicting chemical structure of metabolites from LC-MS/MS spectra using Deep Canonical Correlation Analysis(DeepCCA). This a deep learning extension of [CCA](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2572-9). This work is done in three phases as outlined below.







<h3> 1. Data processing and Embeddings </h3>

In this [notebook](https://github.com/LukaLmelias/DeepCCA_thesis/blob/main/data_preprocessing_and_embeddings.ipynb), we clean, intergrate and generate embeddings of structure and spectra dataset.

<h3> 2. Model development </h3>

This [notebook](https://github.com/LukaLmelias/DeepCCA_thesis/blob/main/DeepCCA_models_training.ipynb) contains DeepCCA optimization codes.

<h3> 3. Prediction and Evaluation </h3>

[Here](https://github.com/LukaLmelias/DeepCCA_thesis/blob/main/spectra_structure_prediction.ipynb) we perfom a cross modal retrieval. It takes in the spectra embeddings and outputs the most likely structure of that spectra. Next we evaluate using Tanimoto scores whether the predicted structure is similar to the true structure.





 
