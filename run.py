#thanks to https://towardsdatascience.com/continuous-genetic-algorithm-from-scratch-with-python-ff29deedd099 for implementaion of ga algorithm

from ga import population, first_generation, fitness_similarity_check, next_generation
import numpy as np

Result_file = 'tower_weights.txt'
max_weight = 10
min_weight = 0
number_of_genes = 4
number_of_individuals = 5


weights = population(number_of_individuals, number_of_genes, max_weight,min_weight)
gen = []
gen.append(first_generation(weights))

fitness_avg = np.array([sum(gen[0]['Fitness'])/len(gen[0]['Fitness'])])
fitness_max = np.array([max(gen[0]['Fitness'])])

res = open(Result_file, 'a')
res.write('\n'+str(gen)+'\n')
res.close()
finish = False

while finish == False:
    # if max(fitness_max) > 6:
    #     break
    # if max(fitness_avg) > 5:
    #     break
    if fitness_similarity_check(fitness_max, 50) == True:
        break

    gen.append(next_generation(gen[-1],max_weight,min_weight))

    fitness_avg = np.append(fitness_avg, sum(gen[-1]['Fitness'])/len(gen[-1]['Fitness']))
    fitness_max = np.append(fitness_max, max(gen[-1]['Fitness']))

    res = open(Result_file, 'a')
    res.write('\n'+str(gen[-1])+'\n')
    res.close()