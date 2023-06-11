'''
    Author:             Suxen
    Date:               2023/3/13
    File decription:    Homework 1 for the evolution computation
'''

############# Initial setting ###############
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import os

# Global variable setting
pop_size = 100                  # Population size
termination = 500               # Number of generation to stop
N = 10                          # N-dimensional Schwefel function problem
K = 10                          # Each element in vector is in [-512, 511] => 10 bits
s = 3                           # Used in tournament selection of survivor selection
current_round = 1               # Used when testing pc = [0, 0.2, 0.5, 0.8, 1]
t = [0, 0.2, 0.5, 0.8, 1]       # Test possibility for pc and pm
nl = [2, 3, 4, 6, 8]            # Test number for n in tournament
take_aver = 10                  # Numbers of averages taken

# Used to calculate the average over 'take_aver' times
aver_GA1 = [0 for k in range(termination)]
aver_GA2 = [0 for k in range(termination)]
aver_GA3 = [0 for k in range(termination)]
aver_GA4 = [0 for k in range(termination)]
GA1_min = GA2_min = GA3_min = GA4_min = np.inf

############## Global function definition #############

# The Schwefel function
def schwefel(x):
    right_sum = 0
    for i in range(N):
        right_sum += x[i] * np.sin(np.sqrt(abs(x[i])))
    return (418.98291 * N - right_sum)

# The function used to plot all GA values
def plot_all(GAs, num):
    
    # if 'images' folder doesn't exist then create it
    if not(os.path.isdir('images')):
        os.mkdir('images')

    generation = []
    fitness = []
    label = []

    for i in range(len(GAs)):
        generation.append(GAs[i].avr_generation)
        fitness.append(GAs[i].avr_fitness)

    label.append('binary_GA_uniform_CO')
    label.append('binary_GA_2point_CO')
    label.append('real_GA_uniform_CO')
    label.append('real_GA_whole_arithmetic_CO')

    for i in range(len(generation)):
        plt.plot(generation[i], fitness[i], label=label[i])
        plt.legend(loc='upper right')

    plt.title('GA comparison')
    plt.xlabel('Generation')
    plt.ylabel('f(x)')
    plt.savefig(f'images\Figure_{num+1}.jpg')
    # plt.show()
    plt.clf()

def slope_calculate():

    # if 'images' folder doesn't exist then create it
    if not(os.path.isdir('images')):
        os.mkdir('images')

    slope_GA1 = [0 for k in range(termination-1)]
    slope_GA2 = [0 for k in range(termination-1)]
    slope_GA3 = [0 for k in range(termination-1)]
    slope_GA4 = [0 for k in range(termination-1)]
    generation = [k for k in range(1, termination)]
    label = ['binary_GA_uniform_CO', 'binary_GA_2point_CO', 'real_GA_uniform_CO', 'real_GA_whole_arithmetic_CO']

    for i in range(termination-1):
        slope_GA1[i] = aver_GA1[i+1] - aver_GA1[i]
        slope_GA2[i] = aver_GA2[i+1] - aver_GA2[i]
        slope_GA3[i] = aver_GA3[i+1] - aver_GA3[i]
        slope_GA4[i] = aver_GA4[i+1] - aver_GA4[i]

    plt.plot(generation, slope_GA1, label=label[0])
    plt.legend(loc='upper right')
    plt.plot(generation, slope_GA2, label=label[1])
    plt.legend(loc='upper right')
    plt.plot(generation, slope_GA3, label=label[2])
    plt.legend(loc='upper right')
    plt.plot(generation, slope_GA4, label=label[3])
    plt.legend(loc='upper right')
    plt.title('GA slope comparison')
    plt.xlabel('Generation')
    plt.ylabel('slope of f(x)')
    plt.savefig(f'images\Round_{current_round}_Slope.jpg')
    # plt.show()
    plt.clf()

def write_aver():
    temp = [aver_GA1, aver_GA2, aver_GA3, aver_GA4]
    GA_min = [GA1_min, GA2_min, GA3_min, GA4_min]
    slope_GA1 = [0 for k in range(termination)]
    slope_GA2 = [0 for k in range(termination)]
    slope_GA3 = [0 for k in range(termination)]
    slope_GA4 = [0 for k in range(termination)]

    for i in range(termination-1):
        slope_GA1[i] = aver_GA1[i+1] - aver_GA1[i]
        slope_GA2[i] = aver_GA2[i+1] - aver_GA2[i]
        slope_GA3[i] = aver_GA3[i+1] - aver_GA3[i]
        slope_GA4[i] = aver_GA4[i+1] - aver_GA4[i]
    
    slope_GA1[termination-1] = slope_GA2[termination-1] = slope_GA3[termination-1] = slope_GA4[termination-1] = 0
    slope = [slope_GA1, slope_GA2, slope_GA3, slope_GA4]

    with open('108062226_aver.csv', 'a') as f:
        f.write('\n================\n')
        f.write('Normal with aver over 30\n')
        for i in range(4):
            f.write('================\n')
            f.write(f'GA{i+1}:\n')
            f.write(f'fitness, slope\n')

            for j in range(termination):
                f.write(f'{temp[i][j]}, {slope[i][j]}\n')
            
            f.write(f"The best result of this GA is:\n")
            f.write(f"{GA_min[i]}\n")

def plot_aver():
    
    # if 'images' folder doesn't exist then create it
    if not(os.path.isdir('images')):
        os.mkdir('images')

    generation = [k for k in range(1, termination+1)]
    label = ['binary_GA_uniform_CO', 'binary_GA_2point_CO', 'real_GA_uniform_CO', 'real_GA_whole_arithmetic_CO']
    
    plt.plot(generation, aver_GA1, label=label[0])
    plt.legend(loc='upper right')
    plt.plot(generation, aver_GA2, label=label[1])
    plt.legend(loc='upper right')
    plt.plot(generation, aver_GA3, label=label[2])
    plt.legend(loc='upper right')
    plt.plot(generation, aver_GA4, label=label[3])
    plt.legend(loc='upper right')
    plt.title('GA comparison')
    plt.xlabel('Generation')
    plt.ylabel('f(x)')
    plt.savefig(f'images\Round_{current_round}_Figure_aver_over_30.jpg')
    # plt.show()
    plt.clf()

############# Class defination ##############

# Binary GA using uniform crossover and bit-flip mutation strategies.
class binary_GA_uniform_CO:
    def __init__(self, n=2, pc=0.9, pm=1/K):
        # parameters setting
        self.n = n                  # Used in tournament selection
        self.pc = pc                # Possibility of crossover
        self.pm = pm                # Possibility of mutation
        self.local_min = []         # The best result of this GA, initially null 
        self.population = []


        # Create the initial parents, which are 100 vectors
        # And calculate their fitness
        # Each vector has N elements
        # Each element has 10 bits(2^10) representing integers ranging in [-512, 511]
        for i in range(pop_size):
            vector = []
            for j in range(N):
                vector.append([random.randint(0, 1) for k in range(K)])
            self.population.append(vector)
        
        print('Successfully create binary_GA_uniform_CO.')

    def parent_selection(self):
        # Tournament selection, n = self.n
        # Return value: a list whose elements are vectors representing the parents selected.
        new_parent = []
        for i in range(pop_size):
            best_parent = []

            # Pick n population and choose the best one
            index = []
            k = list(range(0, pop_size-1))
            random.shuffle(k)
            for j in range(self.n):
                index.append(k[j])

            # deterministic
            best_index = self.the_best(self.population, index)

            best_parent = self.population[best_index]
            
            new_parent.append(best_parent)
        
        return new_parent

    def crossover(self, parent_a, parent_b):
        # Uniform crossover, crossover possibility = 0.5
        # parent_a and parent_b are two vectors with N elements, each element has 10 bit
        child_1 = []
        child_2 = []
        for i in range(N):
            test = random.random()
            element_1 = []
            element_2 = []

            # Directly copy it instead of crossover
            if(test >= self.pc):
                element_1 = parent_a[i]
                element_2 = parent_b[i]

            # Starting uniform crossover
            else:
                for j in range(K):

                    coin = random.randint(0, 1)
                    if (coin):
                        element_1.append(parent_a[i][j])
                        element_2.append(parent_b[i][j])
                    else:
                        element_2.append(parent_a[i][j])
                        element_1.append(parent_b[i][j])

            child_1.append(element_1)
            child_2.append(element_2)

        return (child_1, child_2)

    def mutation(self, x):
        # Bit-flip mutation, mutation possibility = self.pm
        # x is a vector with 10 elements, each has 10-bit
        new_x = []
        for i in range(N):
            xj = []
            for j in range(K):
                rand = random.random()
                if(rand >= self.pm):
                    xj.append(x[i][j]) 
                else:
                    xj.append(1 - x[i][j]) # Bit-flip, 1 -> 0, 0 -> 1
            new_x.append(xj)
        return new_x

    def survivor_selection(self):
        # Also use tournament selection just like parent selection
        # Use Tournament selection to choose "pop_size" survivors from self.population and self.offspring
        competitor = []
        survivor = []
        for i in range(pop_size):
            competitor.append(self.population[i])
            competitor.append(self.offspring[i])
        
        # Tournament selection
        for i in range(pop_size):
            index = []
            k = list(range(0, 2*pop_size-1))
            random.shuffle(k)
            for j in range(s):
                index.append(k[j])

            # deterministic
            best_index = self.the_best(competitor, index)

            survivor.append(competitor[best_index])
        
        return survivor

    def binary_decimal(self, x):
        # Binary to decimal
        # x is a vector with N elements, each has "K" bits 
        # Return value: a vector with N elements, each is an integer
        new_x = []
        for i in range(N):
            ans = 0
            exp = 512
            if(x[i][0] == 0): # Sign bit = 0 => positive
                for j in range(K):
                    ans += x[i][j] * exp
                    exp = exp // 2
            else: # Negative
                tmp = [0 for j in range(K)]
                for j in range(K):
                    
                    tmp = 1 - x[i][j]
                    ans += tmp * exp
                    exp = exp // 2
                ans = (ans + 1) * -1
            new_x.append(ans)
        return new_x
    
    def the_best(self, population, index):
        # population is a list compose of vectors with 10 elements
        # index is a list with the chosen indices
        min_value = np.inf
        min_index = 0
        for i in range(len(index)):
            if min_value > schwefel(self.binary_decimal(population[index[i]])):
                min_value = schwefel(self.binary_decimal(population[index[i]]))
                min_index = index[i]
            
        return min_index

    def find_best(self):
        best = np.inf
        best_vector = []
        for i, s in enumerate(self.population):
            if best > (schwefel(self.binary_decimal(s))):
                best = schwefel(self.binary_decimal(s))
                best_vector = s

        return best, best_vector

    def evolution(self, f):
        
        self.start_time = time.time()
        self.global_min = np.inf
        average = 0
        self.avr_fitness = []
        self.avr_generation = []
        print('Start the evolution of binary_GA_uniform_CO')

        for i in range(termination):
            if (i+1) % 30 == 0:
                f.write(f'{i+1}, {average/30}\n')
                self.avr_fitness.append(average/30)
                self.avr_generation.append(i+1)
                average = 0

            if (i+1) % 90 == 30: print(f'Running on generation {i+1} ﾍ( ´∀`)ﾉ............')
            elif (i+1) % 90 == 60: print(f'Running on generation {i+1} ......ﾍ( ´∀`)ﾉ......')
            elif (i+1) % 90 == 0: print(f'Running on generation {i+1} ............ﾍ( ´∀`)ﾉ')
            elif (i+1) == termination: print(f'Finish at generation {i+1}.')
                
            self.new_parents = self.parent_selection()
            self.offspring = []

            for j in range(pop_size // 2):
                child1, child2 = self.crossover(self.new_parents[j], self.new_parents[j+1])
                self.offspring.append(child1)
                self.offspring.append(child2)
            
            for j, child in enumerate(self.offspring):
                self.offspring[j] = self.mutation(child)
            self.population = self.survivor_selection()
            
            self.local_min, self.local_min_vector = self.find_best()
            average += self.local_min

            aver_GA1[i] += self.local_min

            if self.global_min > self.local_min:
                self.global_min = self.local_min
                self.global_min_vector = self.local_min_vector
        
        self.spend_time = time.time() - self.start_time

    def plot(self):
        plt.plot(self.avr_generation, self.avr_fitness)
        plt.title('Uniform-crossover Bit-flip-mutation Binary GA')
        plt.xlabel('Generation')
        plt.ylabel('f(x)')
        plt.show()
        plt.clf()

    def show_best(self, f):
        print('='*20 + '\n')
        print('binary_GA_uniform_CO:')
        print(f'The best value of the whole processing: {self.global_min}')
        print(f'The best vector of the whole processing: {self.binary_decimal(self.global_min_vector)}')
        print(f'The end value after the whole processing: {self.local_min}')
        print(f'The end vector after the whole processing: {self.binary_decimal(self.local_min_vector)}')
        print(f'Time spent: {self.spend_time}s')
        print('\n' + '='*20)
                
        f.write('binary_GA_uniform_CO:\n')
        f.write(f'{self.global_min}\n')
        for s in self.binary_decimal(self.local_min_vector):
            f.write(f'{s}, ')
        f.write(f'\ntime spent, {self.spend_time}\n')
        f.write('\n')

# Binary GA using 2-point crossover and bit-flip mutation strategies.
class binary_GA_2point_CO:
    def __init__(self, n=2, pc=0.9, pm=1/K):
        # parameters setting
        self.n = n                  # Used in tournament selection
        self.pc = pc                # Possibility of crossover
        self.pm = pm                # Possibility of mutation
        self.local_min = []         # The best result of this GA, initially null 
        self.population = []


        # Create the initial parents, which are 100 vectors
        # And calculate their fitness
        # Each vector has N elements
        # Each element has 10 bits(2^10) representing integers ranging in [-512, 511]
        for i in range(pop_size):
            vector = []
            for j in range(N):
                vector.append([random.randint(0, 1) for k in range(K)])
            self.population.append(vector)

        print('Successfully create binary_GA_2point_CO.')

    def parent_selection(self):
        # Tournament selection, n = self.n
        # Return value: a list whose elements are vectors representing the parents selected.
        new_parent = []
        for i in range(pop_size):
            best_parent = []
            # Pick n population and choose the best one
            index = []
            k = list(range(0, pop_size-1))
            random.shuffle(k)
            for j in range(self.n):
                index.append(k[j])

            # deterministic
            best_index = self.the_best(self.population, index)

            best_parent = self.population[best_index]
            
            new_parent.append(best_parent)
        
        return new_parent

    def crossover(self, parent_a, parent_b):
        # 2-point crossover, which choose two index to divide
        # parent_a and parent_b are two vectors with N elements, each element has 10 bit
        child_1 = []
        child_2 = []
        for i in range(N):
            rand = random.random()
            element_1 = []
            element_2 = []

            # Do 2-point crossover
            if(rand < self.pc):

                # Choose two indices to divide (Both are included)
                # For example, if l = 1, r = 5, two bit string = 
                # B1:       [1, |0, 0, 0, 0, 0,| 1, 1, 0, 1], B2  = [0, |1, 1, 1, 1, 1,| 0, 0, 1, 1, 1]
                # => B1':   [1, |1, 1, 1, 1, 1,| 1, 1, 0, 1], B2' = [0, |0, 0, 0, 0, 0,| 0, 0, 1, 1, 1]
                l, r = 0, 0

                # Ensure that two swap indices are not be identical
                # Range from [0, 9]
                while l == r:
                    l = random.randint(0, K-1)
                    r = random.randint(0, K-1)

                    # Make sure that l <= r at all time
                    if l > r:
                        l, r = r, l
                

                for j in range(K):

                    # Swap
                    if j >= l and j <= r:
                        element_2.append(parent_a[i][j])
                        element_1.append(parent_b[i][j])
                    # No swap
                    else:
                        element_1.append(parent_a[i][j])
                        element_2.append(parent_b[i][j])

            # Directly copy it instead of crossover
            else:
                element_1 = parent_a[i]
                element_2 = parent_b[i]

            child_1.append(element_1)
            child_2.append(element_2)

        return (child_1, child_2)

    def mutation(self, x):
        # Bit-flip mutation, mutation possibility = self.pm
        # x is a vector with 10 elements, each has 10-bit
        new_x = []
        for i in range(N):
            xj = []
            for j in range(K):
                rand = random.random()
                if(rand < self.pm):
                    xj.append(1 - x[i][j]) # Bit-flip
                else:
                    xj.append(x[i][j])
            new_x.append(xj)
        return new_x

    def survivor_selection(self):
        # Also use tournament selection just like parent selection
        # Use Tournament selection to choose "pop_size" survivors from self.population and self.offspring
        competitor = []
        survivor = []
        for i in range(pop_size):
            competitor.append(self.population[i])
            competitor.append(self.offspring[i])

        # Tournament selection
        for i in range(pop_size):
            index = []
            k = list(range(0, 2*pop_size-1))
            random.shuffle(k)
            for j in range(s):
                index.append(k[j])

            # deterministic
            best_index = self.the_best(competitor, index)

            survivor.append(competitor[best_index])
        
        return survivor

    def binary_decimal(self, x):
        # Binary to decimal
        # x is a vector with N elements, each has "K" bits 
        # Return value: a vector with N elements, each is an integer
        new_x = []
        for i in range(N):
            ans = 0
            exp = 512
            if(x[i][0] == 0): # Sign bit = 0 => positive
                for j in range(K):
                    ans += x[i][j] * exp
                    exp = exp // 2
            else: # Negative
                tmp = [0 for j in range(K)]
                for j in range(K):
                    
                    tmp = 1 - x[i][j]
                    ans += tmp * exp
                    exp = exp // 2
                ans = (ans + 1) * -1
            new_x.append(ans)
        return new_x
    
    def the_best(self, population, index):
        # population is a list compose of vectors with 10 elements
        # index is a list with the chosen indices
        min_value = np.inf
        min_index = 0
        for i in range(len(index)):
            if min_value > schwefel(self.binary_decimal(population[index[i]])):
                min_value = schwefel(self.binary_decimal(population[index[i]]))
                min_index = index[i]
            
        return min_index

    def find_best(self):
        best = np.inf
        best_vector = []
        for i, s in enumerate(self.population):
            if best > (schwefel(self.binary_decimal(s))):
                best = schwefel(self.binary_decimal(s))
                best_vector = s

        return best, best_vector

    def evolution(self, f):
        
        self.start_time = time.time()
        self.global_min = np.inf
        average = 0
        self.avr_fitness = []
        self.avr_generation = []

        print('Start the evolution of binary_GA_2point_CO')
        for i in range(termination):
            if (i+1) % 30 == 0:
                f.write(f'{i+1}, {average/30}\n')
                self.avr_fitness.append(average/30)
                self.avr_generation.append(i+1)
                average = 0

            if (i+1) % 90 == 30: print(f'Running on generation {i+1} ﾍ( ´∀`)ﾉ............')
            elif (i+1) % 90 == 60: print(f'Running on generation {i+1} ......ﾍ( ´∀`)ﾉ......')
            elif (i+1) % 90 == 0: print(f'Running on generation {i+1} ............ﾍ( ´∀`)ﾉ')
            elif (i+1) == termination: print(f'Finish at generation {i+1}.')
                
            self.new_parents = self.parent_selection()
            self.offspring = []

            for j in range(pop_size // 2):
                child1, child2 = self.crossover(self.new_parents[j], self.new_parents[j+1])
                self.offspring.append(child1)
                self.offspring.append(child2)
            
            for j, child in enumerate(self.offspring):
                self.offspring[j] = self.mutation(child)
            self.population = self.survivor_selection()
            
            self.local_min, self.local_min_vector = self.find_best()
            average += self.local_min

            aver_GA2[i] += self.local_min

            if self.global_min > self.local_min:
                self.global_min = self.local_min
                self.global_min_vector = self.local_min_vector

        self.spend_time = time.time() - self.start_time

    def plot(self):
        plt.plot(self.avr_generation, self.avr_fitness)
        plt.title('2-point-crossover Bit-flip-mutation Binary GA')
        plt.xlabel('Generation')
        plt.ylabel('f(x)')
        plt.show()
        plt.clf()

    def show_best(self, f):
        print('='*20 + '\n')
        print('binary_GA_2point_CO:')
        print(f'The best value of the whole processing: {self.global_min}')
        print(f'The best vector of the whole processing: {self.binary_decimal(self.global_min_vector)}')
        print(f'The end value after the whole processing: {self.local_min}')
        print(f'The end vector after the whole processing: {self.binary_decimal(self.local_min_vector)}')
        print(f'Time spent: {self.spend_time}s')
        print('\n' + '='*20)
        
        f.write('binary_GA_2point_CO:\n')
        f.write(f'{self.global_min}\n')
        for s in self.binary_decimal(self.local_min_vector):
            f.write(f'{s}, ')
        f.write(f'\ntime spent, {self.spend_time}\n')
        f.write('\n')

# Real-valued GA using uniform crossover and uniform mutation strategies.
class real_GA_uniform_CO:
    def __init__(self, n=2, pc=0.9, pm=1/K):
        # parameters setting
        self.n = n                  # Used in tournament selection
        self.pc = pc                # Possibility of crossover
        self.pm = pm                # Possibility of mutation
        self.local_min = []         # The best result of this GA, initially null 
        self.population = []

        # Create the initial parents, which are 100 vectors
        # Each vector has N elements
        # Each element has a floating value ranging in [-512, 511]
        for i in range(pop_size):
            minimum = (2 ** (K-1)) * -1
            maximum = 2 ** (K-1) - 1
            vector = [random.uniform(minimum, maximum) for k in range(N)]
            self.population.append(vector)
        
        print('Successfully create real_GA_uniform_CO.')

    def parent_selection(self):
        # Tournament selection, n = self.n
        # Return value: a list whose elements are vectors representing the parents selected.
        new_parent = []
        for i in range(pop_size):
            best_parent = []
            # Pick n population and choose the best one
            index = []
            k = list(range(0, pop_size-1))
            random.shuffle(k)
            for j in range(self.n):
                index.append(k[j])

            best_index = self.the_best(self.population, index)
            best_parent = self.population[best_index]
            
            new_parent.append(best_parent)
        
        return new_parent

    def crossover(self, parent_a, parent_b):
        # Uniform crossover
        # parent_a and parent_b are two vectors with N elements, each element is a value range in [-512, 511]
        child_1 = []
        child_2 = []

        rand = random.random()

        # Directly copy it instead of crossover
        if(rand >= self.pc):
            child_1 = parent_a
            child_2 = parent_b

        # Starting uniform crossover
        else:
            for i in range(N):

                coin = random.randint(0, 1)
                if (coin):
                    child_1.append(parent_a[i])
                    child_2.append(parent_b[i])
                else:
                    child_1.append(parent_a[i])
                    child_2.append(parent_b[i])

        return (child_1, child_2)

    def mutation(self, x):
        # Uniform mutation, mutation possibility = self.pm
        # x is a vector with 10 elements, each has value ranging in [-512, 511]

        new_x = []
        for i in range(N):
            rand = random.random()
            minimum = (2 ** (K-1)) * -1
            maximum = 2 ** (K-1) - 1

            # Do mutation
            if rand < self.pm: 
                new_x.append(random.uniform(minimum, maximum))

            # Do nothing, just copy it
            else:
                new_x.append(x[i])
        return new_x

    def survivor_selection(self):
        # Also use tournament selection just like parent selection
        # Use Tournament selection to choose "pop_size" survivors from self.population and self.offspring
        competitor = []
        survivor = []
        for i in range(pop_size):
            competitor.append(self.population[i])
            competitor.append(self.offspring[i])
        # Tournament selection
        for i in range(pop_size):
            index = []
            k = list(range(0, 2*pop_size-1))
            random.shuffle(k)
            for j in range(s):
                index.append(k[j])

            best_index = self.the_best(competitor, index)
            survivor.append(competitor[best_index])
        
        return survivor
    
    def the_best(self, population, index):
        # population is a list compose of vectors with 10 elements
        # index is a list with the chosen indices
        min_value = np.inf
        min_index = 0
        for i in range(len(index)):
            if min_value > schwefel(population[index[i]]):
                min_value = schwefel(population[index[i]])
                min_index = index[i]
            
        return min_index
    
    def find_best(self):
        best = np.inf
        best_vector = []
        for s in self.population:
            if best > schwefel(s):
                best = schwefel(s)
                best_vector = s

        return best, best_vector

    def evolution(self, f):
        
        self.start_time = time.time()
        self.global_min = np.inf
        average = 0
        self.avr_fitness = []
        self.avr_generation = []
        print('Start the evolution of real_GA_uniform_CO')

        for i in range(termination):
            if (i+1) % 30 == 0:
                f.write(f'{i+1}, {average/30}\n')
                self.avr_fitness.append(average/30)
                self.avr_generation.append(i+1)
                average = 0

            if (i+1) % 90 == 30: 
                print(f'Running on generation {i+1} ﾍ( ´∀`)ﾉ............')
            elif (i+1) % 90 == 60: 
                print(f'Running on generation {i+1} ......ﾍ( ´∀`)ﾉ......')
            elif (i+1) % 90 == 0: 
                print(f'Running on generation {i+1} ............ﾍ( ´∀`)ﾉ')
            elif (i+1) == termination: print(f'Finish at generation {i+1}.')
                
            self.new_parents = self.parent_selection()
            self.offspring = []

            for j in range(pop_size // 2):
                child1, child2 = self.crossover(self.new_parents[j], self.new_parents[j+1])
                self.offspring.append(child1)
                self.offspring.append(child2)
            
            for j, child in enumerate(self.offspring):
                self.offspring[j] = self.mutation(child)
            self.population = self.survivor_selection()
            
            self.local_min, self.local_min_vector = self.find_best()
            average += self.local_min

            aver_GA3[i] += self.local_min

            if self.global_min > self.local_min:
                self.global_min = self.local_min
                self.global_min_vector = self.local_min_vector

        self.spend_time = time.time() - self.start_time

    def plot(self):
        plt.plot(self.avr_generation, self.avr_fitness)
        plt.title('Uniform-crossover Uniform-mutation real-valued GA')
        plt.xlabel('Generation')
        plt.ylabel('f(x)')
        plt.show()
        plt.clf()

    def show_best(self, f):
        print('='*20 + '\n')
        print('real_GA_uniform_CO:')
        print(f'The best value of the whole processing: {self.global_min}')
        print(f'The best vector of the whole processing: {self.global_min_vector}')
        print(f'The end value after the whole processing: {self.local_min}')
        print(f'The end vector after the whole processing: {self.local_min_vector}')
        print(f'Time spent: {self.spend_time}s')
        print('\n' + '='*20)

        f.write('real_GA_uniform_CO:\n')
        f.write(f'{self.global_min}\n')
        for s in self.global_min_vector:
            f.write(f'{s}, ')
        f.write(f'\ntime spent, {self.spend_time}\n')
        f.write('\n')

# Real-valued GA using whole-arithmetic crossover and uniform mutation strategies.
class real_GA_whole_arithmetic_CO:
    def __init__(self, n=2, pc=0.9, pm=1/K):
        # parameters setting
        self.n = n                  # Used in tournament selection
        self.pc = pc                # Possibility of crossover
        self.pm = pm                # Possibility of mutation
        self.local_min = []         # The best result of this GA, initially null 
        self.population = []

        # Create the initial parents, which are 100 vectors
        # Each vector has N elements
        # Each element has a floating value ranging in [-512, 511]
        for i in range(pop_size):
            minimum = (2 ** (K-1)) * -1
            maximum = 2 ** (K-1) - 1
            vector = [random.uniform(minimum, maximum) for k in range(N)]
            self.population.append(vector)
        
        print('Successfully create real_GA_whole_arithmetic_CO.')

    def parent_selection(self):
        # Tournament selection, n = self.n
        # Return value: a list whose elements are vectors representing the parents selected.
        new_parent = []
        for i in range(pop_size):
            best_parent = []
            # Pick n population and choose the best one
            index = []
            k = list(range(0, pop_size-1))
            random.shuffle(k)
            for j in range(self.n):
                index.append(k[j])

            best_index = self.the_best(self.population, index)
            best_parent = self.population[best_index]
            
            new_parent.append(best_parent)
        
        return new_parent

    def crossover(self, parent_a, parent_b):
        # Whole-arithmetic crossover
        # parent_a and parent_b are two vectors with N elements, each element is a value range in [-512, 511]
        child_1 = []
        child_2 = []

        # Starting whole-arithmetic crossover
        for i in range(N):

            child_1.append(self.pc * parent_a[i] + (1-self.pc) * parent_b[i])
            child_2.append(self.pc * parent_b[i] + (1-self.pc) * parent_a[i])

        return (child_1, child_2)

    def mutation(self, x):
        # Uniform mutation, mutation possibility = self.pm
        # x is a vector with 10 elements, each has value ranging in [-512, 511]

        new_x = []
        for i in range(N):
            rand = random.random()
            minimum = (2 ** (K-1)) * -1
            maximum = 2 ** (K-1) - 1

            # Do mutation
            if rand < self.pm: 
                new_x.append(random.uniform(minimum, maximum))

            # Do nothing, just copy it
            else:
                new_x.append(x[i])
        return new_x

    def survivor_selection(self):
        # Also use tournament selection just like parent selection
        # Use Tournament selection to choose "pop_size" survivors from self.population and self.offspring
        competitor = []
        survivor = []
        for i in range(pop_size):
            competitor.append(self.population[i])
            competitor.append(self.offspring[i])
        # Tournament selection
        for i in range(pop_size):
            index = []
            k = list(range(0, 2*pop_size-1))
            random.shuffle(k)
            for j in range(s):
                index.append(k[j])

            best_index = self.the_best(competitor, index)
            survivor.append(competitor[best_index])
        
        return survivor
    
    def the_best(self, population, index):
        # population is a list compose of vectors with 10 elements
        # index is a list with the chosen indices
        min_value = np.inf
        min_index = 0
        for i in range(len(index)):
            if min_value > schwefel(population[index[i]]):
                min_value = schwefel(population[index[i]])
                min_index = index[i]
            
        return min_index
    
    def find_best(self):
        best = np.inf
        best_vector = []
        for s in self.population:
            if best > schwefel(s):
                best = schwefel(s)
                best_vector = s

        return best, best_vector

    def evolution(self, f):
        
        self.start_time = time.time()
        self.global_min = np.inf
        average = 0
        self.avr_fitness = []
        self.avr_generation = []
        print('Start the evolution of real_GA_whole_arithmetic_CO')

        for i in range(termination):
            if (i+1) % 30 == 0:
                f.write(f'{i+1}, {average/30}\n')
                self.avr_fitness.append(average/30)
                self.avr_generation.append(i+1)
                average = 0

            if (i+1) % 90 == 30: 
                print(f'Running on generation {i+1} ﾍ( ´∀`)ﾉ............')
            elif (i+1) % 90 == 60: 
                print(f'Running on generation {i+1} ......ﾍ( ´∀`)ﾉ......')
            elif (i+1) % 90 == 0: 
                print(f'Running on generation {i+1} ............ﾍ( ´∀`)ﾉ')
            elif (i+1) == termination: print(f'Finish at generation {i+1}.')
                
            self.new_parents = self.parent_selection()
            self.offspring = []

            for j in range(pop_size // 2):
                child1, child2 = self.crossover(self.new_parents[j], self.new_parents[j+1])
                self.offspring.append(child1)
                self.offspring.append(child2)
            
            for j, child in enumerate(self.offspring):
                self.offspring[j] = self.mutation(child)
            self.population = self.survivor_selection()
            
            self.local_min, self.local_min_vector = self.find_best()
            average += self.local_min

            aver_GA4[i] += self.local_min

            if self.global_min > self.local_min:
                self.global_min = self.local_min
                self.global_min_vector = self.local_min_vector
            
        self.spend_time = time.time() - self.start_time

    def plot(self):
        plt.plot(self.avr_generation, self.avr_fitness)
        plt.title('Whole-arithmetic-crossover Uniform-mutation real-valued GA')
        plt.xlabel('Generation')
        plt.ylabel('f(x)')
        plt.show()
        plt.clf()

    def show_best(self, f):
        print('='*20 + '\n')
        print('real_GA_whole_arithmetic_CO:')
        print(f'The best value of the whole processing: {self.global_min}')
        print(f'The best vector of the whole processing: {self.global_min_vector}')
        print(f'The end value after the whole processing: {self.local_min}')
        print(f'The end vector after the whole processing: {self.local_min_vector}')
        print(f'Time spent: {self.spend_time}s')
        print('\n' + '='*20)

        f.write('real_GA_whole_arithmetic_CO:\n')
        f.write(f'{self.global_min}\n')
        for s in self.global_min_vector:
            f.write(f'{s}, ')
        f.write(f'\ntime spent, {self.spend_time}\n')
        f.write('\n')

############### Main function ################

if __name__ == '__main__':

    
    f = open('108062226_aver.csv', 'w')
    f.close()
    with open('108062226_result.csv', 'w') as f:
        for i in range(take_aver):
            

            f.write(f'Round {i}\n')
            print(f'Round {i+1}')

            GA1 = binary_GA_uniform_CO()
            GA1.evolution(f)
            GA1.show_best(f)

            GA2 = binary_GA_2point_CO()
            GA2.evolution(f)
            GA2.show_best(f)

            GA3 = real_GA_uniform_CO()
            GA3.evolution(f)
            GA3.show_best(f)
            
            GA4 = real_GA_whole_arithmetic_CO()
            GA4.evolution(f)
            GA4.show_best(f)

            # Calculate the min over 'takeover'
            if GA1_min > GA1.global_min:
                GA1_min = GA1.global_min
            if GA2_min > GA2.global_min:
                GA2_min = GA2.global_min
            if GA3_min > GA3.global_min:
                GA3_min = GA3.global_min
            if GA4_min > GA4.global_min:
                GA4_min = GA4.global_min

        # GA1.plot()
        # GA2.plot()
        # GA3.plot()
        # GA4.plot()

    # slope_calculate()
    for i in range(termination):
        aver_GA1[i] /= take_aver
        aver_GA2[i] /= take_aver
        aver_GA3[i] /= take_aver
        aver_GA4[i] /= take_aver
    plot_aver()
    write_aver()
    current_round += 1