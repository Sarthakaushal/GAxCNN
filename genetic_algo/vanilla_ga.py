import numpy as np
import pandas as pd
import math

class MetaData():
    def __init__(self, result,did_mutate=False,
                chromosome_mutated=-1, alee_index=-1) -> None:
        self.result = result
        self.did_mutate = did_mutate
        self.chromosome_mutated = chromosome_mutated
        self.alee_index = alee_index

class GA():
    def __init__(
        self,
        selection = None,
        max_gen = 0,
        crossover_index = 0,
        mutation_param = 0.1,
        boost_weak_chromosomes = False,
        generation_data ='data/config_1.txt'
                 ) -> None:
        self.selection = selection
        self.max_gen = max_gen
        self.crossover_index = crossover_index
        self.alpha = mutation_param
        self.boost_weak_chromosomes = boost_weak_chromosomes
        self.population = []
        self.current_gen = 0
        self.constraints = {}
        self.verbose = 1
        P, W, S, g, stop = self.init_knapsack_population(
                generation_data)
        self.population = P
        self.constraints['W'] = W
        self.constraints['S'] = S
        self.current_gen = g
        self.stop = stop
        self.sol_metadata = {}
    
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
        # np.random.seed(1470)
        # Populate the problem variables
        if self.verbose:
            print(config)
        with open(config, 'r') as file :
            lines = file.readlines ()
        pop_size , n, stop , W = map(int , [lines[i].strip() for i in range (4) ])
        # if self.pop_size:
        #     pop_size = self.pop_size
        S = [ tuple(map(int , line.strip().split())) for line in lines[4:]]
        # Initialize population at generation 0
        g = 0
        P = np. random . randint (2, size = ( pop_size , n))
        return P, W, S, g, stop
    
    def dot(self, chromosome, spl=False):
        """
        Computes the dot product between a chromosome and set of weights 
        return  0 if dot product is greater than total s=knapsack weight W
        else the dot product
        """
        # print(f"Sum ciWi = {np.dot(chromosome, np.array(self.S)[:,0]).sum()}")
        if np.dot(chromosome, np.array(self.constraints['S'])[:,0]).sum()>self.constraints['W']:
            return 0
        else:
            if not spl:
                return np.dot(chromosome, np.array(self.constraints['S'])[:,1])
            else:
                return np.dot(chromosome, np.array(self.constraints['S'])[:,0])
            
    def fitness(self, chr=None)->np.ndarray:
        #As described in the paper
        if type(chr)==np.ndarray:
            return self.dot(chr)
        fit_vector = np.array(list(map(self.dot, self.population)))
        for i in range(fit_vector.shape[0]):
            if fit_vector[i] == 0:
                fit_vector[i] = 1
        return fit_vector
    
    def crossover(self,parent1,parent2):
        """
        Post selection gets the parents and depending on the crossover point
        declared when initializing the population does the crossover resulting
        in two children.
        If crossover point is 0 then the parents are only returned
        """
        if self.crossover_index>0:
            p1_fh = parent1[:self.crossover_index]
            p1_sh = parent1[self.crossover_index:]
            p2_fh = parent2[:self.crossover_index]
            p2_sh = parent2[self.crossover_index:]
            # print(p1_fh,p2_sh)
            return (
                np.concatenate((p1_fh,p2_sh)),
                np.concatenate((p2_fh,p1_sh))
                )
        else:
            return(parent1, parent2)
    
    def roulette_selection(self):
        """
        depending upon the fitness function's output the new parents are 
        selected parents selected randomly from a probability distribution
        described as by fitness func values
        """
        fit_vec = self.fitness()
        if self.boost_weak_chromosomes:
            for index in range(fit_vec.shape[0]):
                if fit_vec[index]==0:
                    fit_vec[index] == 10
        prob_fit_vec = fit_vec/np.sum(fit_vec)
        #Will not fit values not get selected at all? for now have excluded them
        p1 = np.random.choice(self.population.shape[0], 1, p=prob_fit_vec)
        p2 = np.random.choice(self.population.shape[0], 1, p=prob_fit_vec)
        c1, c2 = self.crossover(self.population[p1[0]][:],self.population[p2[0]][:])
        return (c1, c2)
    
    def tournament_selection(self):
        """
        Select two parents randomly
        """
        fit_vec = self.fitness()
        selected_parents = []
        while len(selected_parents)<2:
            potential_p1 = np.random.choice(self.population.shape[0], 1)
            potential_p2 = np.random.choice(self.population.shape[0], 1)
            if fit_vec[potential_p1[0]]>fit_vec[potential_p1[0]]:
                selected_parents.append(self.population[potential_p1[0]])
            else:
                selected_parents.append(self.population[potential_p2[0]])
        c1 , c2 = self.crossover(selected_parents[0], selected_parents[1])
        return (c1,c2)
    
    def mutate(self):
        """
        Based on the mutation rate N number of chromosomes are mutated in a 
        population, number of genes that are mutated is also calculated based on
        mutation_prob
        """
        mutation_index = []
        gene_index = []
        for _ in range(math.ceil(self.alpha*self.population.shape[0])):
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
    
    def evaluate_generation(self):
        """
        Calc evaluation metrics for a generation 
        """
        fit_vec = self.fitness()
        avg_fitness = fit_vec.mean()
        fittest_individual = self.population[np.argmax(fit_vec)]
        fittest_individual_num_genes = self.population[np.argmax(fit_vec)].sum()
        std_fitness = fit_vec.std()
        
        return {
            "avg_fitness" : avg_fitness,
            "fittest_individual": fittest_individual,
            "fitness_of_FI":fit_vec[np.argmax(fit_vec)],
            "fittest_individual_num_genes":fittest_individual_num_genes,
            "std_fitness" : std_fitness
        }
    
    def to_mutate(self)->MetaData:
        """
        check_if mutation for this set is on?
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
        else:
            if self.verbose:
                print(";( No Mutation")
            return MetaData(
                    result= self.evaluate_generation(),
                    did_mutate= False
                )
    
    def new_gen(self):
        """
        One pass of the GA algorithm
        """
        if self.verbose:
            print(f"{'='*20} GENERATION : {self.current_gen} {'='*20}")
        p_new = []
        if self.selection == "roulette_selection":
            selection = self.roulette_selection
        else:
            selection = self.tournament_selection
        # creating new generation
        while len(p_new)<self.population.shape[0]:
            (g_ith_1, g_ith_2) = selection()
            p_new.append(g_ith_1)
            p_new.append(g_ith_2)
        self.current_gen+=1
        meta_data = self.to_mutate()
        self.sol_metadata[self.current_gen] = meta_data
        self.population = np.array(p_new)
        return np.asarray(p_new)
    
    def result_to_df(self):
        """
        Generic function to create a DF from the Meta data of the func:solve
        """
        indexes = []
        values = []
        for gen_index in self.sol_metadata.keys():
            indexes.append(gen_index)
            values.append(self.sol_metadata[gen_index].__dict__['result'])
        return pd.DataFrame(data=values, index=indexes)
    
    def solve(self):
        """
        driver function 
        """
        for _ in range(self.stop):
            # generate new generation
            self.new_gen()