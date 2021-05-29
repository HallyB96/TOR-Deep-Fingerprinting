# TOR-Deep-Fingerprinting
Master-thesis project looking into the possibilites of attacking the anonymity of TOR with deeplearning fingerprinting techniques.

In the "Jupyter Notebook files" folder, all the implementations made through Jupyter notebook is located.
In the "Python files" folder, all implementations made through "normal" Python files are located.

## IMPORTANT!
The code was implemented through the use of Jupyter Notebook. The Python files provided are downloaded of the Jupyter notebook, and can because of this be messy. In order to properly replicate the setup, use the .ipynb files together with the NoDefModel.py, WTFPADModel.py, WalkieTalkieModel.py, Datareader.py and converter.py

## Required Packages
The required packages are all imported at the top of each file. In order to download the used packages, pip-install "name" can be used. To properly recreate the setting used in this project, download anaconda3, download the latest version of cuda, cudNN. Additionally, download TensorFlow and keras through pip3 -install with anaconda.

## Files
Converter.py converts Python 2 pickles to Python 3.
Datareader.py contains methods to return semi-prepared data to be used for training and classification. Downloading and re-naming a dataset to the names specified is required.
NoDefModel.py, WTFPADModel.py, WalkieTalkieModel.py returns the different models used. 

Closed-World-No_Def.py/ipynb                 Trains the closed-world undefended model and evaluates its performance.

Closed-World-wtf-pad.py/ipynb                Trains the closed-world WTF-PAD model and evaluates its performance.

Closed-World-WalkieTalkie.py/ipynb           Trains the closed-world Walkie-Talkie model and evaluates its performance.

Openworld_NoDef_Evaluation.py/ipynb          Evaluates the Open-World undefended model by loading a previously trained model.

Openworld_Wtf-Pad_Evaluation.py/ipynb        Evaluates the Open-World WTF-PAD model by loading a previously trained model.

Openworld_WalkieTalkie_Evaluation.py/ipynb   Evaluates the Open-World Walkie-Talkie model by loading a previously trained model.


