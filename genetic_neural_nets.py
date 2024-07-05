import numpy as np
from scipy.special import expit

class Genetic_Network(object):

    def __init__(self,size,function,population_size):
        """
        Parameters:
        size:list describing the number of neurons per layer ([3,4,1] 3 neurons layer 1, 4 neurons layer 2 and 1 neurons layer 3
        function:list descirbing the activation function to apply on the hidden layers and output layers ("reLu","tanh","sigmoid")
        population_size: int describing the number of neural network to be generated
        """
        self.number_layer=len(size)
        self.size=size
        self.function=function
        self.population_size=population_size
        self.weight=[[np.random.uniform(low=-1,high=1,size=(size[i+1],size[i])) for i in range(self.number_layer-1)] for _ in range(population_size)] 
        
    def sigmoid(self,z):
        return(1/(1+expit(-z)))
    
    def tanh(self,z):
        return(2*self.sigmoid(z)-1)
    
    def softmax(self,z):
        shift=z-np.max(z)
        exps=np.exp(shift)
        return((exps/np.sum(exps)))
    
    def activation_apply(self,z,fun):
        if(fun=="reLu"):
            return(np.maximum(0,z))
        elif(fun=="sigmoid"):
            return(self.sigmoid(z))
        elif(fun=="tanh"):
            return(self.tanh(z))
        elif(fun=="softmax"):
            return(self.softmax(z))
        else:
            print("Unknow function, choices are:  \n reLu, sigmoid, tanh")
            raise SystemExit(0)

    def feedforward(self,input,weight):
        """
        Parameters:
        input: vector of the data to be taken into the input layer, it must be of the following dimension
        (N,1) with N corresponding to the number of neurons of the input layer
        weight: matrix of the weight of a neural network
        bias: matrix of the bias of a neural network
        """
        signal_layers=[np.array(input)]
        activation_layers=[np.array(input)]
        for i in range(self.number_layer-1):
            signal_layers.append(np.dot(weight[i],activation_layers[i]))
            activation_layers.append(self.activation_apply((signal_layers[i+1]),self.function[i]))
        return(activation_layers)


    def score_total(self,score):
        """
        Parameters:
        score: function used to obtain the score of a neural network (fitness) 
               input: weight (a neural net)
               return: a signle value that value being the fitness of the neural net

        Goal: 
        Measure the fitness of each chromosome (neural nets) in the population by
        computing the fitness score
        """
        score_tot=[]
        for i in range(self.population_size):
            score_tot.append(score(self.weight[i],self))
        return(score_tot)

    def selection_best(self,score,num_ind):
        """
        Parameters:
        score: function used to obtain the score of a neural network (fitness)
        num_ind: int how many of the best individuals we want to keep

        Goal: Order the fitness score of each individual of the population
        and keep the best one
        """
        score_tot=np.array(self.score_total(score))
        sel_ind=np.argsort(score_tot)[-num_ind:]
        return(sel_ind,score_tot[sel_ind])
    
    def weighted_random_selection(self,sel_ind,fit_ind):
        """
        Parameters:
        score: function used to obtain the score of a neural network (fitness)
        num_ind: int how many of the best individuals we want to keep

        Goal: Order the fitness score of each individual of the population
        and select an individual based on a probabality based on how good the 
        fitness of the individual is. Even individual with a low fitness score may
        be chosen this is done to keep enough diversity.
        """
        sum_fit=np.sum(fit_ind)
        prob_selection=fit_ind/sum_fit
        sel_parents=np.random.choice(sel_ind,
                                 p=prob_selection,
                                 size=2)
        return(sel_parents)

    def simulated_binary_crossover(self,parent1,parent2,n_c):
        """
        Parameters:
        parent1: neural net (chromosome of the parent 1)
        parent2: neural net (chromosome of the parent 2)
        n_c: float/int a huge value make the children closer chromosome closer to the parent

        Goal: Performinh a simulated binary crossover (SBC). This link
        give a good understanding of it 
        https://engineering.purdue.edu/~sudhoff/ee630/Lecture04.pdf
        It's intented purpose is to create two children chromosome from
        two parents by using SBC.
        """
        rdm_num=np.random.random(parent1.shape)
        beta=np.empty(parent1.shape)
        beta[rdm_num<=0.5]=(2*rdm_num[rdm_num<=0.5])**(1.0/(n_c+1))
        beta[rdm_num>0.5]=(1.0/(2.0*(1.0-rdm_num[rdm_num>0.5])))**(1.0/(n_c+1))
        children1=0.5*((1+beta)*parent1+(1-beta)*parent2)
        children2=0.5*((1-beta)*parent1+(1+beta)*parent2)
        return(children1,children2)
    
    def mutation(self,chromosome,p,mu,sigma):
        """
        Parameters:
        chromosome: neural net (chromosome of a children) to be mutated
        p: probability of a gene of the chromosome to be mutated
        mu: parameter of the normal distribution
        sigma: parameter of the normal distribution

        Goal: After the crossover we mutate some part
        of the chromosome by adding the value of a 
        normal distribution of parameters mu,sigma
        
        """
        gene_chosen=np.random.random(chromosome.shape)<p
        chromosome[gene_chosen]+=np.random.normal(mu,sigma, chromosome[gene_chosen].shape)
        return(chromosome)
    
    def children(self,score,num_ind,n_c,p,mu=0,sigma=1):
        """
        Parametes:
        Similar to the one defined before
         
        Goal:
        Create the future generation by selecting parents and 
        producing children from them. 
        """
        sel_ind,fit_ind=self.selection_best(score,num_ind)
        childs=[self.weight[i] for i in sel_ind]
        for _ in range(int(self.population_size/2)):
            sel_parents=self.weighted_random_selection(sel_ind,fit_ind)
            parent1,parent2=self.weight[sel_parents[0]],self.weight[sel_parents[1]]
            child1_temp=[]
            child2_temp=[]
            for j in range(len(parent1)):
                children1,children2=self.simulated_binary_crossover(parent1[j],parent2[j],n_c)
                child1_temp.append(self.mutation(children1,p,mu,sigma))
                child2_temp.append(self.mutation(children2,p,mu,sigma))
            childs.append(child1_temp)
            childs.append(child2_temp)
        self.weight=childs  

    def train(self,epoch,score,numb_ind,n_c,p,mu=0,sigma=1):
        """
        Parameters:
        epoch: number of generation to be created
        Similar to the one defined before

        Goal:
        Training the neural nets
        """
        for _ in range(epoch):
            score_tot=self.score_total(score)
            current_best_fit=np.max(score_tot)
            print("epoch: ",_, 
                  ", mean fitness: ",np.mean(score_tot),
                  "max fitness: ", current_best_fit)
            self.children(score,numb_ind,n_c,p,mu,sigma)

    def get_best(self,score):
        """
        Parameters:
        Similar to the one defined before

        Goal: 
        Get the best neural nets out of the last generation 
        the one with the best fitness score
        """
        score_tot=self.score_total(score)
        best=np.argmax(score_tot)
        return(self.weight[best])
