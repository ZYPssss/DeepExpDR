# DRP-MTGSA
DeepExpDR: rug Response Prediction through Molecular Topological Grouping and Substructure-aware Expert


## Data
'data/Binary_class' : All the data needed for the classification experiment, including warm setting and cold setting


'data /bulk': All the data needed for the regression experiment, including warm setting and cold setting


'data /CDR_Matrix': Drug response information matrix of cancer dataset


'data/substructure_data': Drug substructures store information


'data/drug_smiles.csv': Drug Smiles sequence information


'data/unique_cells.csv': Cancer cell line name information in the database


'data/unique_drugs.csv': Drug name information in the database


## Environment
`You can create a conda environment for DRP-MTGSA  by ‘conda env create -f environment.yml‘.`


## Train and test
- ### regression experiment
  - #### warm setting
        `python main_test.py`
  - #### cold setting for cell
        `python main_cell_leave.py`
  - #### cold setting for drug
        `python main_drug_leave.py`
- ### classification experiment
  - #### warm setting
        `python main_test_classify.py`
  - #### cold setting for cell
        `python main_cell_leave_classify.py`
  - #### cold setting for drug
        `python main_drug_leave_classify.py`
