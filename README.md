# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data.

The following techniques have been used:

 - Linear regression
 - Decision Tree
 - Random Forest

 ## Clone the git repository in home directory
```
cd ~
git clone git@github.com:aman-sharma-ta/TAMLEP.git
```

## Folder structure

mle-training<br />
|<br />
|---artifacts/<br />
|---data/<br />
&nbsp;&nbsp;&nbsp;&nbsp;|<br />
&nbsp;&nbsp;&nbsp;&nbsp;|---raw/<br />
&nbsp;&nbsp;&nbsp;&nbsp;|---processed/<br />
|---deploy/<br />
|---dist/<br />
|---docs/<br />
&nbsp;&nbsp;&nbsp;&nbsp;|<br />
&nbsp;&nbsp;&nbsp;&nbsp;|---html/<br />
|---logs/<br />
|---notebooks/<br />
|---scripts/<br />
|---src/<br />
&nbsp;&nbsp;&nbsp;&nbsp;|---housing_price_model/<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---ingest_data.py<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---train.py<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---score.py<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---logging_util.py<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---config.py<br />
|---tests/<br />
.README.md<br />
.gitignore<br />
setup.py<br />

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error and mean_abs_error.

## Installation steps
### To setup environment
- create a new conda enviroment using env.yml
```
cd tamlep
conda env create -f env.yml
```

### Activate the environment created
```
conda activate mle-dev-tamlep
```

## Steps to execute the scripts

Change the current directory to housing_price_model folder
```
cd src/housing_price_model/
```

### Download the data and split it to train and test sets
**ingest_data.py** - This script downloads and creates training and validation datasets. The script also accept the output folder/file path as an user argument.
```
python ingest_data.py
```
### Process the data and train the model
**train.py** - script to train the model(s). This script accepts arguments for input (dataset) and output folders (model pickles).
```
python train.py
```

### Score the model on test data
**score.py** - script to score the model(s). The script accepts arguments for model folder, dataset folder and any outputs.
```
python score.py
```

### Other scripts

**config.py** - This script consists of all the project default configurations
**logging_util.py** - Script to configure the logger function
**code_refactor.sh** - use to reformat any py file in isort,black and flake8 format

## Logging
All the log files are saved to logs/ folder


## Installation as package
run these commands
```
cd tamlep
python setup.py sdist
```
Copy the path of .gz file from dist/ dir
```
pip install .gz_file_path
```
## Testing
### To run all the tests
```
py.test tests/
```

### To run specific test
```
pytest tests/{test_folder_path}/{specific_test_name}.py
```

### Unit testing
Below test can be run to perform unit testing
```
pytest tests/unit_tests/unit_test.py
```

## Sphinx Documentation
Run below commands to start the sphinx documentation in local
```
cd docs/build/html/
python -m http.server 8001
```