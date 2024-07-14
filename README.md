# Artificial snake

The goal of this project is to train a neural network so that the neural network begin to show intelligent behavior on the snake game. This 
will be done by using genetic algorithm as renforcement  training is particularly intersting on such a game mainly because it is easy to select the next generation by using the 
survival time of the snake or the number of apple that it has eaten. Well it is important to note here that I won't delve too deep in solving the snake
as it take too much time. Indeed there are too many configuration of hyperparameters, input data and fitness functions to test. What I truly want to do
is to show that intelligent behavior may emerge from a simple neural network using genetic algorithm. I've put some gif, as examples of some intelligent behavior.
These results were obtained by using a fitness function that take into the consideration the apple and the survival time. As you see the snake use differents strategies as a compromise
between the two. But if you want to try to solve the snake using this code feel free to try it, hopefully this is doable. I give a description of what've done underneath.

## Snake game

The snake game, is more or less the same one that I've coded before. I've only modified the input for the
mouvement. Before it was controled by the key pressed now it is controled by the four neurons of the output layer
of a neural network. Some things have also been added, mainly to use them as data for the input layer of the neural net.

If you want to update/change the data, you must update it not only in the score function but also in the playing function of the snakeAI_visual file.

I've divided the snake game into two version one used for training with no visual output (snakeAI_training). The other with visual output, to visualise how
the neural net obtained by training behave(SnakeAI_visual). Then there is global version (SnakeAI) that train a neural nets on the training version then show
the result using the visual version.

Here is a quick explanation of the hyparemeters of the Game class of the SnakeAI,

1. **root**: tk.Tk() instance
2. **map_dim**: int dimension of the map (map_dim x map_dim)
3. **speed**: int time before the snake make it's next move in ms
4. **net**: Network instance

## Neural network

Similarly, the neural network is based on my previous work for the multilayer percpetron. The architecture is similar but since this
time it is trained using genetic algorithm. I have deleted the backpropagation part. I have also deleted the bias, these could have been learned in 
a similar way to the weight. But this was too costly for my computer.

Here is a quick explanation of the hyparemeters of the neural network.

1. **size**: list of the neurons per layer
3. **function**: list of the activation functions
4. **weight**: list of array representing the weights of a neural net

## Genetic algorithm

The idea behind genetic algorithm is simple, just make a bad copy of how evolution work. In more details what I present here involve multiple steps.

1. Generate a population of size **size_population**, in this case multiple neural nets with randomized weight
2. Evaluate the fitness score of each individual in the population, here it involve computing the score and survival time of each snake controled by a neural net by using the **score**
   funtion
4. Select the one with the best fitness score, it is important for the selection to not be too restrictive to keep enough diversity. But it must also not be too
   loose otherwise it may not converge or too slowly. Here a total of **num_ind** are selected.
6. From the selected group, select two parents. Here this is done by assigning to each individual a probability proportionates to it's fitness score. And performing
   a weighted random choice in the selected group. This make it so that individual with higher fit score are more likely to be selected while still allowing individual with a lower
   score to be selected. A good compromise between restrictive and loose.
7. The two parents then have two childs by performing simulated binary crossover.
8. Finaly the childrens weight are mutated. This mutation is performed in the following way. We begin by selecting which weight is going to be mutated, this is determined by a
   the probability of mutation **p**. Then for each weight to be mutated we add to it a number drawn from a normal distribution of parameters **$\mu$** and **$\sigma$**.
9. Repeat step 6 to 8 until there is enough children. When this is done the next generation is the set of selected individual plus the children.
10. Repeat step 1 to 9 until a set number (**epoch**)

## Hyperparameters

1. **size_population**: int representing how many individual there in the population
2. **score**: function measuring the fitness of an individual
3. **num_ind**:  int number individual that will represent the parents pool and be kept for the next generation
5. **p**: float probability of a mutation 
6. **$\mu$**: float parameters of a normal distribution used to mutate weights
7. **$\sigma$**: float parameters of a normal distribution used to mutate weights
8. **epoch**: int number of generation

For more details directly look at the code, I've commented it.

Due to the important number of hyperparameters it may be difficult to get a good learning configuration. Moreover the choice of the score function is extremely important and it is difficult to set a good 
one. For example if the score function only involve the survival time of the snake. The snake will end up making endless rotation, the simplest solution to maximising it's survival time. But simply having a score function dependant on the number of apple eaten doesn't work either has it present a big leap in behavior. The evolution must happen incrementaly (it could work in fact in one big leap but that would be a huge stroke of luck).The input data is also of extreme importance different data may lead to differents behavior. All in all setting a good training environnement is particularly difficult and take a lot of time.
<
