# piCRISPR

piCRISPR is a Python-based prediction tool for CRISPR off-target effects using physically informed features. It is developed by [Florian Störtz](https://users.ox.ac.uk/~lina3430/) at the group of Dr. Minary, University of Oxford.

# Usage
piCRISPR can be run as
```
python picrispr.py test_input.csv
```
Positional arguments are:
- input file: path to input csv file
- model number: integer between ```0``` and ```7``` to choose which model is used, order corresponds to the one within the publication
- path to models: path to folder where models are saved, default is ```models```
- regression: ```True``` or ```False```

Example input file (comma-separated):
```
target_sequence,grna_target_sequence
AAGGCGCATAAAGATGAGGCGCTGG,GACGCATAAAGATGAGACGCTGG
AAACGCATAAAGATGAGACGCTGGG,GACGCATAAAGATGAGACGCTGG
GGTGATAAGTGGAATGACCATGTGG,GTGATAAGTGGAATGCCATGTGG
GTGATAAGCTGGAATGCCATTGTGG,GTGATAAGTGGAATGCCATGTGG
```

Further features can be supplied when named in the top line:
```
experiment_id,target_sequence,grna_target_sequence,epigen_ctcf,epigen_dnase,epigen_rrbs,epigen_h3k4me3,epigen_drip,energy_1,energy_2,energy_3,energy_4,energy_5
0,AAGGCGCATAAAGATGAGGCGCTGG,GACGCATAAAGATGAGACGCTGG,0.0,0.0,0.0,0.0,0.0,20.12,0.9369158778408675,0.9369158778408675,8.767159085783554,20.12
0,AAACGCATAAAGATGAGACGCTGGG,GACGCATAAAGATGAGACGCTGG,0.0,0.0,0.0,0.0,0.0,0.75,-32.84039169058475,-32.84039169058475,-25.639584889882755,0.75
1,GGTGATAAGTGGAATGACCATGTGG,GTGATAAGTGGAATGCCATGTGG,0.0,0.0,0.0,0.0,0.0,-2.83,-27.117216094014633,-27.117216094014633,-21.500510272206174,-0.33000000000000007
1,GTGATAAGCTGGAATGCCATTGTGG,GTGATAAGTGGAATGCCATGTGG,0.0,0.0,0.0,0.0,0.0,-13.614999999999998,-49.92028390034596,-49.92028390034596,-44.50227987453346,-11.114999999999998
```
The maximum set of features is given in ```test_input.csv```.

Running piCRISPR like this results in an output file ```output.csv``` with columns
- piCRISPR prediction: binary label or score between 0 and 1, depending on whether regression mode was chosen
- ground truth: true label, if given in the input file as column ```cleavage_freq```
- ground truth_transformed: true label transformed to lie between 0 and 1, if a true label has been given in the input file as column ```cleavage_freq```


## Requirements
```python==3.8.3, jupyter-notebook==6.0.3, torch==1.7.0, tensorflow==2.3.1, sklearn==0.23.1, scipy==1.5.0, numpy==1.18.5, pandas==1.0.5, xgboost==1.4.2, matplotlib==3.2.2, pickle==4.0```

## Installation
Unzip ```models/models_torch.zip```.

## Data preparation
In order to predict on custom off-target cleavage data, it must be annotated with epigenetic markers and physically informed features as detailed in the [publication](https://www.biorxiv.org/content/10.1101/2021.11.16.468799v3). We provide a readily annotated dataset in the file [offtarget_260520_nuc.csv.zip](https://github.com/florianst/picrispr/blob/main/offtarget_260520_nuc.csv.zip) which contains a zip-compressed pandas dataframe and can be loaded using ```pd.read_csv('offtarget_260520_nuc.csv.zip')```.

