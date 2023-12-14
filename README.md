# GAxCNN
The demonstrates how a CNN inspired selection mechanism can be used in Genetic Algorithms.

The module contains three files:
- genetic_algo.__init__.py - the main file containing the pooling selection GA
- genetic_algo.vanilla_ga.py - file having standard GA implementation with roulette wheel & tournament selection
- genetic_algo/utils.py - file having utility functions

### Data I/O:
- data/ - place knapsack initialization config files, the program saves the population here
- output/ - each run creates a new folder here within which CSV files for each GA combination are saved, function reads the output folder location from the config parameter 'output_dir'

### Configurations :
- all configs set in config.GAxCNN.py file, prevail unless overridden while execution

### Dependency Resolution 
```console
foo@bar:GAxCNN$ pip install -r requirements.txt
```

### Sample implementation
```console
foo@bar:~$ cd GAxCNN
foo@bar:GAxCNN$ python Sample_Execution_driver.py
``` 


### Research Notebooks

Code commented so that the file runs quickly, it can take more than half a day to run both notebooks
- Variation over different mutation parameters.ipynb: contains code to run different pooling mechanisms for different mutation parameters. 
- Algo Consistency test.ipynb: Runs selection algorithms multiple times to check consistency and provides stats on variation and mean fitness of fittest individual along with execution time

### Dataset used
All the data required for this program was artificially generated, the creator is our teaching assistant for Fundamentals of AI Teaching assistant for Fall'23 at Rochester Institute of Technology. There are config files added to the **data** folder
