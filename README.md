# 706 project
# Code for course Deep Learning for Sequence Analysis 
### Daniela Stern- Gabsi 

### github- dgabsi/SLNI_Project706
(updates were made from danielaneuralx which is my working github but its all mine)

This project is based on the SLNI dataset(The Stanford Natural Language Inference (SNLI) Corpus) and my the project aim is to classify corrrectly
the inference between tow sentences which can be: Contradiction, Entailment or Neutral.
I will implement 4 different models which are based on deep neural network with sequence models.
The first two models are based on LSTM's, and the third is based on the Transformer model.
In the fourth model I will use the Bert pre-trained transformer.
While with the 3 first models I was able to achieve a maximum of 73% accuracy, by using the Bert pre-trained model I achieved 
90% accuracy rate.


In addition since the dataset is very large approx 500,000 pairs , I have also saved pickle files of the vocubulary that the model
build during its construction.
I have also used TensorBoard for following the training.


Main notebook:
**SLNI_706project.ipynb** 

#Please make sure you have the following structure 
Project structure:
- data (Directory for data)
  - slni_1.0 (The dataset json files)
- experiments (Directory for Tensorboard logs)
- saved_models (Directory for saved models, pickles and visualization results charts- PLEASE do not delete)
  -best_models
-  slni 
    - __init__.py 
    - basic_rnn.py 
    - bert_model.py
    - rnn_combined_model.py
    - transformer_model.py
    - slni_dataset.py  (Dataset)
    - utils.py 
    - main_slni.py 
    - training.py (training function. Serves all networks)
    - evaluation.py 
    - params.py
    - SLNI_706project.ipynb (**this is the main notebook that should be used**)

packages needed :
- torch 1.8.0 (I used with cu111)
- datetime
- time
- transformers 4.4.1
- matplotlib 3.3.4
- numpy 1.20.1
- pandas 1.2.3
- scikit-learn 0.24.1
- tensorboard 2.4.1
- torchtext 0.9.0
- spacy 3.0.5
- pickle
- bertviz 1.0.0
