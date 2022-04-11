# piCRISPR

piCRISPR is a Python-based prediction tool for CRISPR off-target effects using physically informed features. It is developed by [Florian Störtz](https://users.ox.ac.uk/~lina3430/) at the group of Dr. Minary, University of Oxford.

## Requirements
```python==3.8.3, jupyter-notebook==6.0.3, torch==1.7.0, tensorflow==2.3.1, sklearn==0.23.1, scipy==1.5.0, numpy==1.18.5, pandas==1.0.5, xgboost==1.4.2, matplotlib==3.2.2, pickle==4.0```

## Data preparation
In order to predict on custom off-target cleavage data, it must be annotated with epigenetic markers and physically informed features as detailed in the [publication](https://www.biorxiv.org/content/10.1101/2021.11.16.468799v1).

## How to run
Unzip ```models/models_torch.zip```. piCRISPR can then be run using ```python picrispr.py test_input.csv```, or using the jupyter notebook ```picrispr.ipynb```.

The jupyter notebook ```picrispr.ipynb``` contains testing as well, which can be done according to the two different scenarios described in the publication. This is configurable via the boolean variable ```compare_deepcrispr```.