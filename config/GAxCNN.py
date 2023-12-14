##### Parameters for GAxCNN initialization #####
################################################

# Selection mechanism to be used
selection = "roulette_selection"

# pool selection Parameters
################################################
# pooling type  
pooling_type = 'max'

# kernel size
kernel_size = 3

# Maximun number of population geneartions
max_gen = 1500

# Polulation size at Generation 0
pop_size = 150

# Chromosome Index for GA Crossover 
crossover_index = 25

# Mutaiton rate
mutation_param = 0.1

# Boost week chromosomes
boost_weak_chromosomes = True

# Data config file to be used
generation_data ='data/config_1.txt'

# Data output dir
output_dir = "output/"