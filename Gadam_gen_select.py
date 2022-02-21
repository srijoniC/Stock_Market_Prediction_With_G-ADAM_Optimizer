from gadam import GAdam
from core.models import inverse_model, forward_model
from torch import nn, optia
class GAdam_gen_select(Optimizer):
    def __init__(self, params):
        import numpy
        import ga
        equation_inputs =self.params

        # Number of the weights we are looking to optimize.
        num_weights = len(equation_inputs)

        sol_per_pop = 2
        num_parents_mating = 2

        pop_size = (sol_per_pop,num_weights) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
        #Creating the initial population.
        new_population = optia.Adam.step()
                
        best_outputs = []
        num_generations = 1000
        for generation in range(num_generations):
            # Measuring the fitness of each chromosome in the population.
            prev_population=new_population
            fitnessA = optia.Adam.loss(list(Optimizer.params.inversenet())+list(Optimizer.params.forwardnet()), amsgrad=True,lr=Optimizer.params.lr).step()
            
            # Selecting the best parents in the population for mating.
            parents = ga.select_mating_pool(new_population, fitness,
                                            num_parents_mating)
                                            
            # Generating next generation using crossover.
            offspring_crossover = ga.crossover(parents,
                                            offspring_size=(pop_size[0]-parents.shape[0], num_weights))

            # Adding some variations to the offspring using mutation.
            offspring_mutation = ga.mutation(offspring_crossover, num_mutations=2)
            

            # Creating the new population based on the parents and offspring.
            new_population[0:parents.shape[0], :] = parents
            new_population[parents.shape[0]:, :] = offspring_mutation
            fitnessG=optia.Adam.loss(list(new_population, amsgrad=True,lr=Optimizer.params.lr)).step()
            fitness=min([fitnessA,fitnessG])
            if (fitness=<fitnessA):
                new_population=prev_population
            # The best result in the current iteration.
            best_outputs.append(new_population)
            fitnessArr.append(fitness)
            # Getting the best solution after iterating finishing all generations.

        #At first, the fitness is calculated for each solution in the final generation.
        fitness = optia.Adam.loss(list(new_population, amsgrad=True,lr=Optimizer.params.lr)).step()
        # Then return the index of that solution corresponding to the best fitness.
        
        return new_population