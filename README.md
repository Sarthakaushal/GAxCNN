# GAxCNN
The demonstrates how a CNN inspired selection mechanism can be used in Genetic Algorithms.

The module contains three files:
- genetic_algo.__init__.py - main file containing the pooling selection GA
- genetic_algo.vanilla_ga.py - file having standard GA implementation with roulette wheel & tournament selection
- genetic_algo/utils.py - file having utility functions

Data I/O:
- data/ - place knapsack initialization config files, program saves the populaiton here
- output/ - each run creates a new folder here along within which csv files for each GA combination is saved, function reads the output folder location from the config parameter 'output_dir'

Configurations :
- all configs set in config.GAxCNN.py file, prevail unless overridden while execution

# Dependency Resolution 
```console
foo@bar:~$ cd GAxCNN
foo@bar:GAxCNN$ python Sample_Execution_driver.py
``` 

# Sample implementaion
```console
foo@bar:GAxCNN$ pip install -r requirements.txt
``` 

# Research Notebooks

Code commented so that the file runs quicly, therwise can take more than half a day to run both notebooks
- Variation over different mutation params.ipynb : contains code to run different pooling mechanims for different mutataion parmaeters. 
- Algo Consistancy test.ipynb : Runs selection algoritms multiple times to check consitancy and provides stats on variation and mean fitness of fittest individual along with execution time

# Dataset used
All the data required for this program was artificially generated, the creator is our teaching assistant for Fundamentals of AI Teching assistant for Fall'23 at Rochester Institute of Technology
