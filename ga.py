import numpy as np
from numpy.random import randint
from random import random as rnd
from random import gauss, randrange
from data_process import preprocess

df = preprocess()

def individual(number_of_genes, upper_limit, lower_limit):
    individual = [round(rnd()*(upper_limit - lower_limit)+lower_limit, 1) for x in range(number_of_genes)]
    return individual 

def population(number_of_individuals, number_of_genes, upper_limit, lower_limit):
    return [individual(number_of_genes, upper_limit, lower_limit) 
        for x in range(number_of_individuals)]

def fitness_calculation(ind_weights):
    fitness_value = []
    for i in range(0,len(df)):
        fitness_value.append(df.values[i][4]*(df.values[i][1]*ind_weights[1] + df.values[i][2]*ind_weights[2] - df.values[i][3]*ind_weights[3] - df.values[i][0]*ind_weights[0]))
    return sum(fitness_value)

def roulette(cum_sum, chance):
    veriable = list(cum_sum.copy())
    veriable.append(chance)
    veriable = sorted(veriable)
    return veriable.index(chance)

def selection(generation, method='Fittest Half'):
    generation['Normalized Fitness'] =         sorted([generation['Fitness'][x]/sum(generation['Fitness']) 
        for x in range(len(generation['Fitness']))], reverse = True)
    generation['cumulative_sum'] = np.array(
        generation['Normalized Fitness']).cumsum()
        
    if method == 'Roulette Wheel':
        selected = []
        for x in range(len(generation['Individuals'])//2):
            selected.append(roulette(generation
                ['cumulative_sum'], rnd()))
            while len(set(selected)) != len(selected):
                selected[x] = (roulette(generation['cumulative_sum'], rnd()))

        selected = {'Individuals': 
            [generation['Individuals'][int(selected[x])]
                for x in range(len(generation['Individuals'])//2)]
                ,'Fitness': [generation['Fitness'][int(selected[x])]
                for x in range(
                    len(generation['Individuals'])//2)]}

    elif method == 'Fittest Half':
        selected_individuals = [generation['Individuals'][-x-1]
            for x in range(int(len(generation['Individuals'])//2))]
        selected_fitnesses = [generation['Fitness'][-x-1]
            for x in range(int(len(generation['Individuals'])//2))]
        selected = {'Individuals': selected_individuals,
                    'Fitness': selected_fitnesses}

    elif method == 'Random':
        selected_individuals =             [generation['Individuals']
                [randint(1,len(generation['Fitness']))]
            for x in range(int(len(generation['Individuals'])//2))]
        selected_fitnesses = [generation['Fitness'][-x-1]
            for x in range(int(len(generation['Individuals'])//2))]
        selected = {'Individuals': selected_individuals,
                    'Fitness': selected_fitnesses}

    return selected


def pairing(elit, selected, method = 'Fittest'):
    individuals = [elit['Individuals']]+selected['Individuals']
    fitness = [elit['Fitness']]+selected['Fitness']

    if method == 'Fittest':
        parents = [[individuals[x],individuals[x+1]] 
                   for x in range(len(individuals)//2)]

    if method == 'Random':
        parents = []
        for x in range(len(individuals)//2):
            parents.append(
                [individuals[randint(0,(len(individuals)-1))],
                 individuals[randint(0,(len(individuals)-1))]])
            while parents[x][0] == parents[x][1]:
                parents[x][1] = individuals[
                    randint(0,(len(individuals)-1))]

    if method == 'Weighted Random':
        normalized_fitness = sorted(
            [fitness[x] /sum(fitness) 
             for x in range(len(individuals)//2)], reverse = True)
        cummulitive_sum = np.array(normalized_fitness).cumsum()
        parents = []
        for x in range(len(individuals)//2):
            parents.append(
                [individuals[roulette(cummulitive_sum,rnd())],
                 individuals[roulette(cummulitive_sum,rnd())]])
            while parents[x][0] == parents[x][1]:
                parents[x][1] = individuals[
                    roulette(cummulitive_sum,rnd())]
                    
    return parents

def mating(parents, method='Single Point'):

    if method == 'Single Point':
        pivot_point = randint(1, len(parents[0]))
        offsprings = [parents[0]             [0:pivot_point]+parents[1][pivot_point:]]
        offsprings.append(parents[1]
            [0:pivot_point]+parents[0][pivot_point:])

    if method == 'Two Pionts':
        pivot_point_1 = randint(1, len(parents[0]-1))
        pivot_point_2 = randint(1, len(parents[0]))
        while pivot_point_2<pivot_point_1:
            pivot_point_2 = randint(1, len(parents[0]))
        offsprings = [parents[0][0:pivot_point_1]+
            parents[1][pivot_point_1:pivot_point_2]+
            [parents[0][pivot_point_2:]]]
        offsprings.append([parents[1][0:pivot_point_1]+
            parents[0][pivot_point_1:pivot_point_2]+
            [parents[1][pivot_point_2:]]])
            
    return offsprings

def mutation(individual, upper_limit, lower_limit, muatation_rate=2, method='Reset', standard_deviation = 0.001):
    gene = [randint(0, 7)]

    for x in range(muatation_rate-1):
        gene.append(randint(0, 7))
        while len(set(gene)) < len(gene):
            gene[x] = randint(0, 7)
    mutated_individual = individual.copy()

    if method == 'Gauss':
        for x in range(muatation_rate):
            mutated_individual[x] =             round(individual[x]+gauss(0, standard_deviation), 1)

    if method == 'Reset':
        for x in range(muatation_rate):
            mutated_individual[x] = round(rnd()*                 (upper_limit-lower_limit)+lower_limit,1)
    return mutated_individual

def next_generation(gen, upper_limit, lower_limit):
    elit = {}
    next_gen = {}

    elit['Individuals'] = gen['Individuals'].pop(-1)
    elit['Fitness'] = gen['Fitness'].pop(-1)
    selected = selection(gen)
    parents = pairing(elit, selected)
    
    offsprings = [[[mating(parents[x])
                    for x in range(len(parents))]
                    [y][z] for z in range(2)] 
                    for y in range(len(parents))]
    offsprings1 = [offsprings[x][0]
                   for x in range(len(parents))]
    offsprings2 = [offsprings[x][1]
                   for x in range(len(parents))]
    
    unmutated = selected['Individuals']+offsprings1+offsprings2
    
    mutated = [mutation(unmutated[x], upper_limit, lower_limit) 
        for x in range(len(gen['Individuals']))]
    
    unsorted_individuals = mutated + [elit['Individuals']]
    
    unsorted_next_gen =         [fitness_calculation(mutated[x]) 
         for x in range(len(mutated))]
    
    unsorted_fitness = [unsorted_next_gen[x]
        for x in range(len(gen['Fitness']))] + [elit['Fitness']]
    
    sorted_next_gen =         sorted([[unsorted_individuals[x], unsorted_fitness[x]]
            for x in range(len(unsorted_individuals))], 
                key=lambda x: x[1])
    
    next_gen['Individuals'] = [sorted_next_gen[x][0]
        for x in range(len(sorted_next_gen))]
    
    next_gen['Fitness'] = [sorted_next_gen[x][1]
        for x in range(len(sorted_next_gen))]
    
    gen['Individuals'].append(elit['Individuals'])
    gen['Fitness'].append(elit['Fitness'])
    
    return next_gen

def fitness_similarity_check(max_fitness, number_of_similarity):
    result = False
    similarity = 0
    for n in range(len(max_fitness)-1):
        if max_fitness[n] == max_fitness[n+1]:
            similarity += 1
        else:
            similarity = 0
    if similarity == number_of_similarity-1:
        result = True
    return result


def first_generation(pop):
    
    fitness = [fitness_calculation(pop[x]) 
        for x in range(len(pop))]
    sorted_fitness = sorted([[pop[x], fitness[x]]
        for x in range(len(pop))], key=lambda x: x[1])
    population = [sorted_fitness[x][0] 
        for x in range(len(sorted_fitness))]
    fitness = [sorted_fitness[x][1] 
        for x in range(len(sorted_fitness))]

    return {'Individuals': population, 'Fitness': sorted(fitness)}