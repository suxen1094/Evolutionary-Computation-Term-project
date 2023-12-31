import random
import copy
import os
import matplotlib.pyplot as plt

GENERAION_NUM = 500
STATION_NUM = 10        # The number of vertices
VELOCITY = 8            # Unit: m / second
VEHICLE_NUM = 4         # Number of vehicles
GETOFF_TIME = 5         # Unit: second
PARKING_TIME = 20
BUS_INTERVAL = 300      
INF = 99999
INITIAL_TIME = 0
DEMAND = [0, 10, 23, 250, 91, 17, 2, 20, 104, 50]  # The demand of each station
GREEN_ROUTE = [1, 2, 3, 7, 6, 5, 4, 8, 9, 10]
RED_ROUTE   = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# The representation of unlimited population -> length = {STATION_NUM} * {VEHICLE_NUM}
# Each vehicle has {STATION_NUM} bits, which stands for the goods sent to each station
# For example: [6, 7, 3, 4, 5] => send 6 good to station 1, 7 goods to station 2......

# Note: Why just let the representation be a {STATION_NUM}-bit integer representation?
# Ans: We will take time into consideration, and if there's only {STATION_NUM}-bit, 
# It means there's only 1 vehicle => parking time will be very large

# Balancing the weights between "parking time" and "customer not satisfied" is important 
class unlimited_individual:
    def __init__(self):
        self.delivery = [0 for i in range(VEHICLE_NUM * STATION_NUM)]
        # Initially random when created
        self.rand_deliver_constraints()

        # Assume the interval between every bus start = {BUS_INTERVAL}
        self.time = [i * BUS_INTERVAL for i in range(VEHICLE_NUM)]

        # The total number of delevery
        self.total_delivery = self.calculate_total_delivery()
        self.calculate_time()
        self.weight = self.fitness_function()

    # Goal: to minimize the fitness function of the entire population
    # Weight of the fitness function: time: 1, customer cannot be satisfy: 100
    def fitness_function(self):
        # Define the weight of time and customer not-satisfied and customer over satisfied
        time_weight = 1
        customer_not_satisfy_weight = 2000
        customer_over_satisfy_weight = 2000

        c = 0
        
        self.total_delivery = self.calculate_total_delivery()
        self.calculate_time()

        # Calculate whether the delivery is exceed or non-satisfy 
        for i, d in enumerate(self.total_delivery):
            if d < DEMAND[i]:   # Not satisfy at station i
                num_not_satisfied = DEMAND[i] - d
                c += customer_not_satisfy_weight * num_not_satisfied
            elif d > DEMAND[i]: # Over satisfy at station i
                num_over_satisfied = d - DEMAND[i]
                c += customer_over_satisfy_weight * num_over_satisfied

        self.weight = self.total_time * time_weight + c
        return self.weight
    
    # Unload time are variables + constant
    def calculate_time(self):
        # Calculate time - if there's a station need to serve, then add the parking time
        self.time = [i * BUS_INTERVAL for i in range(VEHICLE_NUM)]
        for i in range(VEHICLE_NUM):
            current_delivery = self.delivery[i * STATION_NUM:i * STATION_NUM + STATION_NUM]
            
            for d in current_delivery:
                if d != 0:
                    self.time[i] += PARKING_TIME
                    self.time[i] += d * GETOFF_TIME
            global DRIVING_TIME
            self.time[i] += DRIVING_TIME
            
        return

    def rand_deliver_nonconstraints(self):
        # During randoming, make sure that total_delivery should not exceed the demand
        current_demand = copy.deepcopy(DEMAND)
        for i in range(VEHICLE_NUM):
            for j in range(STATION_NUM):
                self.delivery[i * STATION_NUM + j] = random.randint(0, current_demand[j])
        return

    def rand_deliver_constraints(self):
        # During randoming, make sure that total_delivery should not exceed the demand
        current_demand = copy.deepcopy(DEMAND)
        for i in range(VEHICLE_NUM):
            for j in range(STATION_NUM):
                self.delivery[i * STATION_NUM + j] = random.randint(0, current_demand[j])
                current_demand[j] -= self.delivery[i * STATION_NUM + j]
        return
    
    def calculate_total_delivery(self):
        total_del = [0 for i in range(STATION_NUM)]
        for i in range(VEHICLE_NUM):
            for j in range(STATION_NUM):
                total_del[j] += self.delivery[j + 10 * i]

        return total_del
    
    @property
    def total_time(self):
        return max(self.time)

    def print_all(self):
        print('-'*20)
        print("Printing all vehicle's status:\n")
        for i in range(VEHICLE_NUM):
            print(f'Vehicle {i+1}:')
            print(f'Time spent: {self.time[i]} s')
            print(f'Passengers go to station   1: {self.delivery[i * STATION_NUM + 0]}')
            print(f'Passengers go to station   2: {self.delivery[i * STATION_NUM + 1]}')
            print(f'Passengers go to station   3: {self.delivery[i * STATION_NUM + 2]}')
            print(f'Passengers go to station   4: {self.delivery[i * STATION_NUM + 3]}')
            print(f'Passengers go to station   5: {self.delivery[i * STATION_NUM + 4]}')
            print(f'Passengers go to station   6: {self.delivery[i * STATION_NUM + 5]}')
            print(f'Passengers go to station   7: {self.delivery[i * STATION_NUM + 6]}')
            print(f'Passengers go to station   8: {self.delivery[i * STATION_NUM + 7]}')
            print(f'Passengers go to station   9: {self.delivery[i * STATION_NUM + 8]}')
            print(f'Passengers go to station  10: {self.delivery[i * STATION_NUM + 9]}')
            print()
        print('-'*20)
    
    def print_total(self):
        print('-'*20)
        print(f'Total time spent: {self.total_time} s')
        print(f'Passengers go to station  1: {self.total_delivery[0]}')
        print(f'Passengers go to station  2: {self.total_delivery[1]}')
        print(f'Passengers go to station  3: {self.total_delivery[2]}')
        print(f'Passengers go to station  4: {self.total_delivery[3]}')
        print(f'Passengers go to station  5: {self.total_delivery[4]}')
        print(f'Passengers go to station  6: {self.total_delivery[5]}')
        print(f'Passengers go to station  7: {self.total_delivery[6]}')
        print(f'Passengers go to station  8: {self.total_delivery[7]}')
        print(f'Passengers go to station  9: {self.total_delivery[8]}')
        print(f'Passengers go to station 10: {self.total_delivery[9]}')
        print(f'Weights: {self.weight}')
        print('-'*20)

class unlimited_capacity_GA1:
    # Integer representation
    def __init__(self, pop_size = 100, cr = 0.8, mr = 0.1, n = 2, generation_num = GENERAION_NUM):
        self.n = n
        self.pop_size = pop_size
        self.cr = cr
        self.mr = mr
        self.population = []
        self.generation_num = generation_num
        for i in range(pop_size):
            c = unlimited_individual()
            self.population.append(c)

        print("Successfully create GA with unlimited capacity")

    def parent_select(self):
        new_parent = []

        # Find pop_size number of parents
        for i in range(self.pop_size):
            picked_parent = []
            k = list(range(0, self.pop_size))
            random.shuffle(k)
            for i in range(self.n):
                index = k[i]
                picked_parent.append(self.population[index])

            best_parent = self.find_best(picked_parent)
            new_parent.append(best_parent)

        return new_parent

    # # Uniform crossover: Preventing premature convergence 
    # def crossover(self, p1, p2):
    #     child1 = unlimited_individual()
    #     child2 = unlimited_individual()
        
    #     possibility = random.random()
    #     if possibility < self.cr:   # Do crossover
    #         vehicle_index = [k for k in range(VEHICLE_NUM)]
    #         random.shuffle(vehicle_index)
    #         for j in range(STATION_NUM):

    #             child1.delivery[vehicle_index[0] * STATION_NUM + j] = p2.delivery[vehicle_index[1] * STATION_NUM + j]
    #             child1.delivery[vehicle_index[1] * STATION_NUM + j] = p2.delivery[vehicle_index[0] * STATION_NUM + j]
    #             child1.delivery[vehicle_index[2] * STATION_NUM + j] = p2.delivery[vehicle_index[3] * STATION_NUM + j]
    #             child1.delivery[vehicle_index[3] * STATION_NUM + j] = p2.delivery[vehicle_index[2] * STATION_NUM + j]
                
    #             child2.delivery[vehicle_index[0] * STATION_NUM + j] = p1.delivery[vehicle_index[1] * STATION_NUM + j]
    #             child2.delivery[vehicle_index[1] * STATION_NUM + j] = p1.delivery[vehicle_index[0] * STATION_NUM + j]
    #             child2.delivery[vehicle_index[2] * STATION_NUM + j] = p1.delivery[vehicle_index[3] * STATION_NUM + j]
    #             child2.delivery[vehicle_index[3] * STATION_NUM + j] = p1.delivery[vehicle_index[2] * STATION_NUM + j]

    #     else:                       # Do nothing
    #         child1.delivery[i] = p1.delivery[i]
    #         child2.delivery[i] = p2.delivery[i]
        
    #     # Update the weight and time spent
    #     child1.weight = child1.fitness_function()
    #     child2.weight = child2.fitness_function()
    #     return (child1, child2)

    # Uniform crossover: Preventing premature convergence 
    def crossover(self, p1, p2):
        child1 = unlimited_individual()
        child2 = unlimited_individual()
        
        for i in range(len(p1.delivery)):
            possibility = random.random()
            if possibility < self.cr:   # Do crossover
                child1.delivery[i] = p2.delivery[i]
                child2.delivery[i] = p1.delivery[i]

            else:                       # Do nothing
                child1.delivery[i] = p1.delivery[i]
                child2.delivery[i] = p2.delivery[i]
        
        # Update the weight and time spent
        child1.weight = child1.fitness_function()
        child2.weight = child2.fitness_function()
        return (child1, child2)

    # Random resetting mutation
    def reset_mutation(self, p):
        possibility = random.random()
        for i in range(len(p.delivery)):
            # Do mutation
            if possibility < self.mr:
                lower_bound = 0
                upper_bound = DEMAND[i % STATION_NUM]
                
                p.delivery[i] = random.randint(lower_bound, upper_bound)
            else:   # Do nothing
                pass
        
        # Update the weight and time
        p.weight = p.fitness_function()
        return

    # Creep mutation -> Ranging +- 20 and should not exceed the maximum demand and 0
    def creep_mutation(self, p):
        possibility = random.random()
        for i in range(len(p.delivery)):
            # Do mutation
            if possibility < self.mr:
                if (p.delivery[i] - 20) < 0:
                    lower_bound = 0
                else:
                    lower_bound = p.delivery[i] - 20
                if (p.delivery[i] + 20) > DEMAND[i % STATION_NUM]:
                    upper_bound = DEMAND[i % STATION_NUM]
                else:
                    upper_bound = p.delivery[i] + 20
                
                p.delivery[i] = random.randint(lower_bound, upper_bound)
            else:   # Do nothing
                pass
        
        # Update the weight and time
        p.weight = p.fitness_function()
        return

    # Also tournament selection, and use (lambda, mu)
    def survivor_select(self):
        new_population = []

        # Find {pop_size} number of offsprings
        for i in range(self.pop_size):
            picked_population = []
            k = list(range(0, self.pop_size))
            random.shuffle(k)
            for i in range(self.n):
                index = k[i]
                picked_population.append(self.offsprings[index])

            best_offspring = self.find_best(picked_population)
            new_population.append(best_offspring)

        return new_population

    def find_best(self, target_population):
        target_population.sort(key=unlimited_individual.fitness_function)
        return target_population[0]
    
    def evolution(self):
        self.global_min = self.find_best(self.population)
        print('Start the evolution of binary_GA_uniform_CO')

        # Do {Generation_num} times
        for i in range(self.generation_num):
            

            if (i+1) % 150 == 50: 
                print(f'Running on generation {i+1} ﾍ( ´∀`)ﾉ............')
            elif (i+1) % 150 == 100: 
                print(f'Running on generation {i+1} ......ﾍ( ´∀`)ﾉ......')
            elif (i+1) % 150 == 0: 
                print(f'Running on generation {i+1} ............ﾍ( ´∀`)ﾉ')
            elif (i+1) == self.generation_num: print(f'Finish at generation {i+1}.')


            self.offsprings = []
            # Parent selection
            new_parents = self.parent_select()

            # Crossover
            for j in range(self.pop_size // 2):
                child1, child2 = self.crossover(new_parents[j], new_parents[j+1])
                self.offsprings.append(child1)
                self.offsprings.append(child2)

            # Mutation
            for j in range(len(self.offsprings)):
                self.creep_mutation(self.offsprings[j])

            # Survivor selection
            self.population = self.survivor_select()
            self.local_min = self.find_best(self.population)

            # Global min: the global minimum during the 500 generation
            # Local min: the minimum value at each generation
            # All global min: the global minimum among all 30 runs

            global aver_GA_weight, aver_GA_time
            aver_GA_weight[i] += self.local_min.weight
            aver_GA_time[i] += self.local_min.total_time

            if self.global_min.weight > self.local_min.weight:
                self.global_min = copy.deepcopy(self.local_min)

        print("Global minimum:")
        self.global_min.print_all()
        self.global_min.print_total()

        global all_global_min
        if all_global_min.weight > self.global_min.weight:
            all_global_min = copy.deepcopy(self.global_min)

class unlimited_capacity_GA2:
    # Integer representation
    def __init__(self, pop_size = 100, cr = 0.8, mr = 0.1, n = 2, generation_num = GENERAION_NUM):
        self.n = n
        self.pop_size = pop_size
        self.cr = cr
        self.mr = mr
        self.population = []
        self.generation_num = generation_num
        for i in range(pop_size):
            c = unlimited_individual()
            self.population.append(c)

        print("Successfully create GA with unlimited capacity")

    def parent_select(self):
        new_parent = []

        # Find pop_size number of parents
        for i in range(self.pop_size):
            picked_parent = []
            k = list(range(0, self.pop_size))
            random.shuffle(k)
            for i in range(self.n):
                index = k[i]
                picked_parent.append(self.population[index])

            best_parent = self.find_best(picked_parent)
            new_parent.append(best_parent)

        return new_parent

    # Uniform crossover: Preventing premature convergence 
    def crossover(self, p1, p2):
        child1 = unlimited_individual()
        child2 = unlimited_individual()
        
        possibility = random.random()
        if possibility < self.cr:   # Do crossover
            vehicle_index = [k for k in range(VEHICLE_NUM)]
            random.shuffle(vehicle_index)
            for j in range(STATION_NUM):
                child1.delivery[vehicle_index[0] * STATION_NUM + j] = p2.delivery[vehicle_index[1] * STATION_NUM + j]
                child1.delivery[vehicle_index[1] * STATION_NUM + j] = p2.delivery[vehicle_index[0] * STATION_NUM + j]
                child1.delivery[vehicle_index[2] * STATION_NUM + j] = p2.delivery[vehicle_index[3] * STATION_NUM + j]
                child1.delivery[vehicle_index[3] * STATION_NUM + j] = p2.delivery[vehicle_index[2] * STATION_NUM + j]
                
                child2.delivery[vehicle_index[0] * STATION_NUM + j] = p1.delivery[vehicle_index[1] * STATION_NUM + j]
                child2.delivery[vehicle_index[1] * STATION_NUM + j] = p1.delivery[vehicle_index[0] * STATION_NUM + j]
                child2.delivery[vehicle_index[2] * STATION_NUM + j] = p1.delivery[vehicle_index[3] * STATION_NUM + j]
                child2.delivery[vehicle_index[3] * STATION_NUM + j] = p1.delivery[vehicle_index[2] * STATION_NUM + j]

        else:                       # Do nothing
            child1.delivery[i] = p1.delivery[i]
            child2.delivery[i] = p2.delivery[i]
        
        # Update the weight and time spent
        child1.weight = child1.fitness_function()
        child2.weight = child2.fitness_function()
        return (child1, child2)

    # # Uniform crossover: Preventing premature convergence 
    # def crossover(self, p1, p2):
    #     child1 = unlimited_individual()
    #     child2 = unlimited_individual()
        
    #     for i in range(len(p1.delivery)):
    #         possibility = random.random()
    #         if possibility < self.cr:   # Do crossover
    #             child1.delivery[i] = p2.delivery[i]
    #             child2.delivery[i] = p1.delivery[i]

    #         else:                       # Do nothing
    #             child1.delivery[i] = p1.delivery[i]
    #             child2.delivery[i] = p2.delivery[i]
        
    #     # Update the weight and time spent
    #     child1.weight = child1.fitness_function()
    #     child2.weight = child2.fitness_function()
    #     return (child1, child2)

    # Random resetting mutation
    def reset_mutation(self, p):
        possibility = random.random()
        for i in range(len(p.delivery)):
            # Do mutation
            if possibility < self.mr:
                lower_bound = 0
                upper_bound = DEMAND[i % STATION_NUM]
                
                p.delivery[i] = random.randint(lower_bound, upper_bound)
            else:   # Do nothing
                pass
        
        # Update the weight and time
        p.weight = p.fitness_function()
        return

    # Creep mutation -> Ranging +- 20 and should not exceed the maximum demand and 0
    def creep_mutation(self, p):
        possibility = random.random()
        for i in range(len(p.delivery)):
            # Do mutation
            if possibility < self.mr:
                if (p.delivery[i] - 20) < 0:
                    lower_bound = 0
                else:
                    lower_bound = p.delivery[i] - 20
                if (p.delivery[i] + 20) > DEMAND[i % STATION_NUM]:
                    upper_bound = DEMAND[i % STATION_NUM]
                else:
                    upper_bound = p.delivery[i] + 20
                
                p.delivery[i] = random.randint(lower_bound, upper_bound)
            else:   # Do nothing
                pass
        
        # Update the weight and time
        p.weight = p.fitness_function()
        return

    # Also tournament selection, and use (lambda, mu)
    def survivor_select(self):
        new_population = []

        # Find {pop_size} number of offsprings
        for i in range(self.pop_size):
            picked_population = []
            k = list(range(0, self.pop_size))
            random.shuffle(k)
            for i in range(self.n):
                index = k[i]
                picked_population.append(self.offsprings[index])

            best_offspring = self.find_best(picked_population)
            new_population.append(best_offspring)

        return new_population

    def find_best(self, target_population):
        target_population.sort(key=unlimited_individual.fitness_function)
        return target_population[0]
    
    def evolution(self):
        self.global_min = self.find_best(self.population)
        print('Start the evolution of binary_GA_uniform_CO')

        # Do {Generation_num} times
        for i in range(self.generation_num):
            

            if (i+1) % 150 == 50: 
                print(f'Running on generation {i+1} ﾍ( ´∀`)ﾉ............')
            elif (i+1) % 150 == 100: 
                print(f'Running on generation {i+1} ......ﾍ( ´∀`)ﾉ......')
            elif (i+1) % 150 == 0: 
                print(f'Running on generation {i+1} ............ﾍ( ´∀`)ﾉ')
            elif (i+1) == self.generation_num: print(f'Finish at generation {i+1}.')


            self.offsprings = []
            # Parent selection
            new_parents = self.parent_select()

            # Crossover
            for j in range(self.pop_size // 2):
                child1, child2 = self.crossover(new_parents[j], new_parents[j+1])
                self.offsprings.append(child1)
                self.offsprings.append(child2)

            # Mutation
            for j in range(len(self.offsprings)):
                self.creep_mutation(self.offsprings[j])

            # Survivor selection
            self.population = self.survivor_select()
            self.local_min = self.find_best(self.population)

            # Global min: the global minimum during the 500 generation
            # Local min: the minimum value at each generation
            # All global min: the global minimum among all 30 runs

            global aver_GA_weight2, aver_GA_time2
            aver_GA_weight2[i] += self.local_min.weight
            aver_GA_time2[i] += self.local_min.total_time

            if self.global_min.weight > self.local_min.weight:
                self.global_min = copy.deepcopy(self.local_min)

        print("Global minimum:")
        self.global_min.print_all()
        self.global_min.print_total()

        global all_global_min2
        if all_global_min2.weight > self.global_min.weight:
            all_global_min2 = copy.deepcopy(self.global_min)
   

DISTANCE_GREEN =   [ 
            [0, 400, INF, INF, INF, INF, INF, INF, INF, INF],
            [INF, 0, 260, INF, INF, INF, INF, INF, INF, INF],
            [INF, INF, 0, 500, INF, INF, INF, INF, INF, INF],
            [INF, INF, INF, 0, 450, INF, INF, INF, INF, INF],
            [INF, INF, INF, INF, 0, 290, INF, INF, INF, INF],
            [INF, INF, INF, INF, INF, 0, 280, INF, INF, INF],
            [INF, INF, INF, INF, INF, INF, 0, 550, INF, INF],
            [INF, INF, INF, INF, INF, INF, INF, 0, 260, INF],
            [INF, INF, INF, INF, INF, INF, INF, INF, 0, 400],
            [INF, INF, INF, INF, INF, INF, INF, INF, INF, 0]]

DISTANCE_RED = [ 
        [0, 400, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, 0, 260, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, 0, 550, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, 0, 280, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, 0, 290, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, 0, 450, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, 0, 500, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, 0, 260, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, 0, 400],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, 0]]

# Algorithm implementation
def floyd_warshall(G):
    distance = list(map(lambda i: list(map(lambda j: j, i)), G))

    # Adding vertices individually
    for k in range(STATION_NUM):
        for i in range(STATION_NUM):
            for j in range(STATION_NUM):
                if j > i:
                    distance[i][j] = min(distance[i][j], distance[i][k] + distance[k][j])
    return distance

distance_green = floyd_warshall(DISTANCE_GREEN)
distance_red = floyd_warshall(DISTANCE_RED)
DRIVING_TIME = distance_green[0][9] / VELOCITY

aver_GA_weight = [0 for k in range(GENERAION_NUM)]
aver_GA_time = [0 for k in range(GENERAION_NUM)]
aver_GA_weight2 = [0 for k in range(GENERAION_NUM)]
aver_GA_time2 = [0 for k in range(GENERAION_NUM)]
all_global_min = unlimited_individual()
all_global_min2 = unlimited_individual()
aver = 30

def plot_aver_weight():
    
    # if 'images' folder doesn't exist then create it
    if not(os.path.isdir('images')):
        os.mkdir('images')

    generation = [k for k in range(1, GENERAION_NUM+1)]
    label = ["average of GA's fitness"]
    
    plt.plot(generation, aver_GA_weight2, label=label[0])
    plt.legend(loc='upper right')
    plt.title('Fitness trend, average over 30')
    plt.xlabel('Generation')
    plt.ylabel('f(x)')
    plt.savefig(f'images\Fitness_aver_over_30.jpg')
    # plt.show()
    plt.clf()

def plot_aver_time():
    
    # if 'images' folder doesn't exist then create it
    if not(os.path.isdir('images')):
        os.mkdir('images')

    generation = [k for k in range(1, GENERAION_NUM+1)]
    label = ["average of GA's time"]
    
    plt.plot(generation, aver_GA_time2, label=label[0])
    plt.legend(loc='upper right')
    plt.title('Time trend, average over 30')
    plt.xlabel('Generation')
    plt.ylabel('Time spent')
    plt.savefig(f'images\Time_aver_over_30.jpg')
    # plt.show()
    plt.clf()

def plot_aver_weight_diff():
    
    # if 'images' folder doesn't exist then create it
    if not(os.path.isdir('images')):
        os.mkdir('images')

    generation = [k for k in range(1, GENERAION_NUM+1)]
    label = ["average of GA's fitness using uniform crossover directly", "average of GA's fitness using modified uniform crossover"]
    
    plt.plot(generation, aver_GA_weight, label=label[0])
    plt.plot(generation, aver_GA_weight2, label=label[1])
    plt.legend(loc='upper right')
    plt.title('Fitness trend, average over 30')
    plt.xlabel('Generation')
    plt.ylabel('f(x)')
    plt.savefig(f'images\Fitness_diff_aver_over_30.jpg')
    # plt.show()
    plt.clf()

def plot_aver_time_diff():
    
    # if 'images' folder doesn't exist then create it
    if not(os.path.isdir('images')):
        os.mkdir('images')

    generation = [k for k in range(1, GENERAION_NUM+1)]
    label = ["average of GA's fitness using uniform crossover directly", "average of GA's fitness using modified uniform crossover"]
    
    plt.plot(generation, aver_GA_time, label=label[0])
    plt.plot(generation, aver_GA_time2, label=label[1])
    plt.legend(loc='upper right')
    plt.title('Time trend, average over 30')
    plt.xlabel('Generation')
    plt.ylabel('Time spent')
    plt.savefig(f'images\Time_diff_aver_over_30.jpg')
    # plt.show()
    plt.clf()

if __name__ == "__main__":
    # for i in range(aver):
    #     print(f'Current round: Round {i+1}')
    #     GA1 = unlimited_capacity_GA1()
    #     GA2 = unlimited_capacity_GA2()
    #     GA2.evolution()
    #     GA1.evolution()

    # for i in range(GENERAION_NUM):
    #     aver_GA_weight[i] /= aver
    #     aver_GA_time[i] /= aver
    #     aver_GA_weight2[i] /= aver
    #     aver_GA_time2[i] /= aver

    # plot_aver_weight_diff()
    # plot_aver_time_diff()
    # print('-' * 20)
    # print('Global minimum:')
    # all_global_min.print_all()
    # all_global_min.print_total()
    # all_global_min2.print_all()
    # all_global_min2.print_total()

    for i in range(aver):
        print(f'Current round: Round {i+1}')
        GA2 = unlimited_capacity_GA2()
        GA2.evolution()

    for i in range(GENERAION_NUM):
        aver_GA_weight2[i] /= aver
        aver_GA_time2[i] /= aver

    plot_aver_weight()
    plot_aver_time()

    print('-' * 20)
    print('Global minimum:')
    all_global_min2.print_all()
    all_global_min2.print_total()