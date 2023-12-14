import math
import pandas as pd
import numpy as np
from .vanilla_ga import GA
from config import GAxCNN as params
from numpy.linalg import inv

class MetaData():
    def __init__(self, result,did_mutate=False,
                chromosome_mutated=-1, alee_index=-1) -> None:
        self.result = result
        self.did_mutate = did_mutate
        self.chromosome_mutated = chromosome_mutated
        self.alee_index = alee_index

class GAxCNN(GA):
    def __init__(self,
                 pop_size= params.pop_size, 
                 selection = params.selection,
                 max_gen = params.max_gen,
                 crossover_index = params.crossover_index,
                 mutation_param = params.mutation_param,
                 boost_weak_chromosomes = params.boost_weak_chromosomes,
                 generation_data =params.generation_data,
                 pooling_type = params.pooling_type,
                 kernel_size = params.kernel_size):
        """
        constructor for parent class
        """
        self.pop_size = pop_size
        self.padding = 0
        self.pooling_type = pooling_type
        self.kernel_size = kernel_size
        super().__init__(
            selection = selection,
            max_gen = max_gen,
            crossover_index = crossover_index,
            mutation_param = mutation_param,
            boost_weak_chromosomes = boost_weak_chromosomes,
            generation_data =generation_data,
            pop_size=pop_size
            )
        self.orig_population = self.population[:]
        self.generation_snaps = {}
        
    def update_selection(self, new_selection):
        """
        Updates the selection algorithg for a  class
        """
        self.selection = new_selection
        return None
    
    def max_possible_square(self):
        """
        returns max possible square value
        """
        return math.floor(self.pop_size**(1/2))
    
    def init_knapsack_population(self, config):
        """
        Populates variables from config and initialize P at gen 0.
        Parameters :
        config ( str): path to config file
        Returns :
        g (int): current generation
        P ( matrix or two D array ): population of individuals
        W (int): Knapsack capacity
        S ( list of tuples ): Each tuple is an item (w_i , v_i)
        stop ( int) : final generation ( stop condition )
        """
        print("I am the changed Init function")
        # Populate the problem variables
        if self.verbose:
            print(config)
        with open(config, 'r') as file :
            lines = file.readlines ()
        pop_size , n, stop , W = map(int , [lines[i].strip() for i in range (4) ])
        if not self.pop_size:
            self.pop_size = pop_size 
        S = [ tuple(map(int , line.strip().split())) for line in lines[4:]]
        # Initialize population at generation 0
        g = 0
        P = np. random . randint (2, size = ( self.max_possible_square()**2 , n))
        return P, W, S, g, stop
    
    def mutate(self, whole_pop=False):
        """
        Based on the mutation rate N number of chromosomes are mutated in a 
        population, number of genes that are mutated is also calculated based on
        mutation_prob
        """
        mutation_index = []
        gene_index = []
        if whole_pop:
            perc_population_rate = 1
        else:
            perc_population_rate = self.alpha
        for _ in range(math.ceil(perc_population_rate*self.population.shape[0])):
            while True:
                index= np.random.choice(range(self.population.shape[0]))
                if index not in mutation_index:
                    mutation_index.append(index)
                    break
            new_gene = np.random.choice(range(self.population.shape[1]))
            gene_index.append(new_gene)
            if self.verbose:
                print(f"Mutation in Chromosome {index} & gene no. {new_gene}")
            self.population[index][new_gene] = int(not self.population[index][new_gene])
        return mutation_index, gene_index
    
    def to_mutate(self)->MetaData:
        """
        check_if mutation for this set is on and muatate if condition satifies
        """
        if np.random.choice(range(100), 1)[0]/100 < self.alpha:
            if self.verbose:
                print("Lets Mutate")
            mutation_index, gene_index = self.mutate()
            return MetaData(
                    result= self.evaluate_generation(),
                    did_mutate= True,
                    chromosome_mutated=mutation_index,
                    alee_index= gene_index
                )
        elif (np.unique(self.fitness()).shape[0] ==1) and (
                    self.selection == "pool_selection"):
            if self.verbose:
                print("Lets Mutate, need a little shuffling")
            mutation_index, gene_index = self.mutate(whole_pop=True)
            return MetaData(
                    result= self.evaluate_generation(),
                    did_mutate= True,
                    chromosome_mutated=mutation_index,
                    alee_index= gene_index
                )
        else:
            if self.verbose:
                print(";( No Mutation")
            return MetaData(
                    result= self.evaluate_generation(),
                    did_mutate= False
                )
    
    def pop_rearrangement(self):
        """
        Create a new 2D matrix from the poplulation contaning population
        fitness & adding the chromosome behind the fitness score to create a
        ( N x N x chromosome_length )
        """
        reshaped_pop = []
        fit_vector = self.fitness()
        for row in range(0,self.population.shape[0]):
            # Access every element and change it into a n x n xlen
            ele = self.population[row]
            len_ele = ele.shape[0]
            fit_ele = np.array([fit_vector[row]])
            ele_w_fitness = np.concatenate((fit_ele, ele))
            ele_w_fitness =ele_w_fitness.reshape((1,1,len_ele+1))
            reshaped_pop.append(ele_w_fitness)
        reshaped_pop = np.array(reshaped_pop)
        reshaped_pop = reshaped_pop.reshape(
            (self.max_possible_square(),
                    self.max_possible_square(),
                    len_ele+1))
        return reshaped_pop
    
    def pool_selection(self, padded:bool, kernel_size:int, type:str):
        """
        imputs : padded, kernel_size, type
            padded : adds padding to the tensor so that the size of the tensor remains same
            kernel_size : integer 3,5,7
            type : 'avg', 'min','min', 'max'
        return tensor
        
        """
        rearranged_pop = self.pop_rearrangement()
        self.padding = 0
        max_dims = self.max_possible_square()
        orignal_shape = rearranged_pop.shape
        if padded:
            self.padding = kernel_size-2
            pop_mat = np.zeros((
                max_dims+(kernel_size-2)*2, 
                max_dims+(kernel_size-2)*2, 
                rearranged_pop.shape[2]))
            pop_mat[
                kernel_size-2:kernel_size+max_dims-2,
                kernel_size-2:kernel_size+max_dims-2, 
                :] = rearranged_pop
            rearranged_pop = pop_mat[:,:,:]
        # probabalistic rearrangement & then crossover
        tmp_Rearrangement = np.zeros(orignal_shape)
        kernel_indexs = list(range(0,kernel_size**2))
        for cols in range(0, max_dims):
            for row in range(0, max_dims):
                kernel = rearranged_pop[
                    row:row+kernel_size,cols:cols+kernel_size]
                values = kernel[:,:,0].ravel()
                if type == 'max':
                    selection = values.argmax()
                elif type == 'min':
                    values = np.reciprocal(values)
                    values[np.isinf(values)] = 0
                    selection = values.argmax()
                elif type == 'avg':
                    values = np.abs(values - np.average(values))
                    selection = values.argmin()
                else:
                    if np.average(values) ==0:
                        values = np.ones(shape=values.shape)
                    selection = np.random.choice(kernel_indexs,p=values/np.sum(values))
                tmp_Rearrangement[row,cols,:] =  kernel[selection//kernel_size, selection%kernel_size,:]
        return tmp_Rearrangement
    
    def pool_sel_generator(self, rearranged_mat):
        max_dims = self.max_possible_square()
        i = 0
        if self.current_gen % 10 == 0:
            self.generation_snaps[self.current_gen] = rearranged_mat[:,:,0]
        while i<((max_dims**2)//2) +1:
            if i< ((max_dims**2)//2):
                yield(
                    rearranged_mat[(i)//max_dims, (i)%max_dims, :].ravel(),
                    rearranged_mat[(max_dims -i)//max_dims, (max_dims-i)%max_dims, :].ravel(),
                    True)
            elif max_dims%2 == 1:
                tmp = self.population[i]
                tmp[self.crossover_index:] = (rearranged_mat[(max_dims -i)//max_dims, (max_dims-i)%max_dims, :].ravel())[self.crossover_index+1:]
                yield(tmp,tmp, False)
            i+=1
            
    def new_gen(self):
        """
        One pass of the GA algorithm
        """
        # print(self.population.shape)
        if self.verbose:
            print(f"{'='*20} GENERATION : {self.current_gen} {'='*20}")
        p_new = []
        self.augment_init_gen()
        meta_data = self.to_mutate()
        if self.selection == "roulette_selection":
            selection = self.roulette_selection
        elif self.selection == "pool_selection":
            selection =  self.pool_sel_generator
        else:
            selection = self.tournament_selection
        # creating new generation
        if self.selection == "pool_selection":
            for (g_ith_1, g_ith_2, flag) in selection(
                self.pool_selection(
                    padded=True,
                    kernel_size= self.kernel_size,
                    type= self.pooling_type)):
                if flag:
                    p_new.append(g_ith_1[1:])
                    p_new.append(g_ith_2[1:])
                else:
                    p_new.append(g_ith_1)
        else:
            while len(p_new)<self.population.shape[0]:
                (g_ith_1, g_ith_2) = selection()
                p_new.append(g_ith_1)
                p_new.append(g_ith_2)
        self.current_gen+=1
        
        self.sol_metadata[self.current_gen] = meta_data
        self.population = np.array(p_new)
        return np.asarray(p_new)
