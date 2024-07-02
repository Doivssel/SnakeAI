import numpy as np

class Snake(object):
    
    def __init__(self,map):
        """
        Parameters:
        map: instance of the class Map

        Goal:

        Defining the left,right direction and upp/down direction. The position
        of the head and the score of the snake. Also create the first apple
        and the first body part(the head).

        """
        self.dir=["u","d","l","r"]
        self.direction_lr,self.direction_ud=self.init_dir(map)
        self.posX,self.posY=self.init_pos(map)
        self.score=0
        self.time=0
        self.lost=False
        self.body=np.array([[self.posX,self.posY]])
        self.tail=self.body[0]
        self.border=[map.number_col-self.posX,map.number_row-self.posY]
        map.map[self.posY,self.posX]=1
        map.add_apple()

    def init_pos(self,map):
        i=np.random.randint(1,map.number_col-1)
        j=np.random.randint(1,map.number_row-1)
        return(i,j)
    
    def init_dir(self,map):
        dir=self.dir[np.random.randint(0,4)]
        if(dir=="r"):
            self.direction_lr=1
            self.direction_ud=0
        elif(dir=="l"):
            self.direction_lr=-1
            self.direction_ud=0
        elif(dir=="d"):
            self.direction_ud=-1
            self.direction_lr=0
        elif(dir=="u"):
            self.direction_ud=1
            self.direction_lr=0
        return(self.direction_lr,self.direction_ud)

    def move(self,map):

        """
        Parameters:
        map: instance of the class Map

        Goal:
        Restart game if the snake head touch the border of it's body,
        otherwise continue to move while updating the head and tail of the snake
        at each deplacement. Moreover add a body chunk if an apple is eaten.

        Every body part is kept in self.body and the snake on the map
        is kept in map.array
        """
        self.posY+=self.direction_ud
        self.posX+=self.direction_lr
        if(self.posX>=map.number_col or self.posY>=map.number_row or self.posX<0 or self.posY<0):
            self.lost=True
        else:
            map_value=map.map[self.posY,self.posX]
            if(map_value==1):
                self.lost=True
            elif(map_value==0):
                map.map[self.posY,self.posX]=1
                self.body=np.vstack((self.body,[self.posX,self.posY]))
                map.map[self.body[0,1],self.body[0,0]]=0
            elif(map_value==2):
                self.score+=1
                map.map[self.posY,self.posX]=1
                self.body=np.vstack((self.body,[self.posX,self.posY]))
        self.time+=1

class Map(object):

    def __init__(self,row,col):
        """
        Parameters:
        row: number of rows wanted for the grid
        col: number of columns wanted for the grid


        Goal:

        Creating an array for the map and a grid of coordinates that translate the 
        indices of our array into cartesion coordinate. Mainly used 
        to render the body part of the snake and the apple.

        In the array different component of the map are represented by number
        0 for nothing
        1 for body part
        2 for apple

        """

        self.number_row=row
        self.number_col=col
        self.map=np.zeros((row,col))

    def add_apple(self):
        """
        Parameters:
        None

        Goal:
        Add an apple. To do that it get all the indices of the non zero components
        of the map array. Then select one at random and put an apple here.
        """
        indice=np.transpose(np.nonzero(self.map.T==0))
        self.rdm_pos=indice[np.random.randint(0,len(indice))]
        self.map[self.rdm_pos[1],self.rdm_pos[0]]=2

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
        return(1/(1+np.exp(-z)))
    
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
            score_tot.append(score(self.weight[i]))
        return(score_tot)

    def selection_best(self,score,num_ind):
        """
        Parameters:
        score: function used to obtain the score of a neural network (fitness)
        num_ind: int how many of the best individuals we want to keep

        Goal: Order the fitness score of each individual of the population
        and keep the best one
        """
        score_tot=self.score_total(score)
        self_ind=np.argsort(score_tot)[-num_ind:]
        return(self_ind)
    
    def weighted_random_selection(self,score,num_ind):
        """
        Parameters:
        score: function used to obtain the score of a neural network (fitness)
        num_ind: int how many of the best individuals we want to keep

        Goal: Order the fitness score of each individual of the population
        and select an individual based on a probabality based on how good the 
        fitness of the individual is. Even individual with a low fitness score may
        be chosen this is done to keep enough diversity.
        """
        score_tot=self.score_total(score)
        sum_score=np.sum(score_tot)
        prob_selection=score_tot/sum_score
        sel_ind=np.random.choice([i for i in range(self.population_size)],
                                 p=prob_selection,
                                 size=num_ind)
        return(sel_ind)

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
        parametes:
        
        numb_ind: """
        childs=[]
        parents=self.selection_best(score,num_ind)
        for _ in range(int(self.population_size/2)):
            rdm_num=np.random.choice(parents,size=2,replace=False)
            parent1,parent2=self.weight[rdm_num[0]],self.weight[rdm_num[1]]
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
        for _ in range(epoch):
            score_tot=self.score_total(score)
            print(score_tot)
            print(np.mean(score_tot))
            self.children(score,numb_ind,n_c,p,mu,sigma)

    def get_best(self,score):
        best=np.argmax(self.score_total(score))
        print(self.weight[best])

def score(weight):
    map=Map(10,10)
    snake=Snake(map)
    while(snake.lost==False):
        snake.move(map)
        data=np.array([ [snake.tail[0]],
                        [snake.tail[1]],
                        #Distance to the wall in the cross direction centerd around the head
                        [snake.posX],
                        [snake.posY],
                        [snake.border[0]],
                        [snake.border[1]],
                        #Apple in the cross direction centerd around the head
                        [(map.rdm_pos[0]==snake.posX and map.rdm_pos[1]>0 )*1],
                        [(map.rdm_pos[0]==snake.posX and map.rdm_pos[1]<0 )*1],
                        [(map.rdm_pos[1]==snake.posX and map.rdm_pos[0]>0 )*1],
                        [(map.rdm_pos[1]==snake.posX and map.rdm_pos[0]<0 )*1],
                        #These are for the current direction 
                        [(snake.direction_lr<0)*1],
                        [(snake.direction_lr>0)*1],
                        [(snake.direction_ud<0)*1],
                        [(snake.direction_ud>1)*1]])
        output=net.feedforward(data,weight)[-1]
        dir=snake.dir[np.argmax(output)]
        if(dir=="r" and snake.direction_lr==0):
            snake.direction_lr=1
            snake.direction_ud=0
        elif(dir=="l" and snake.direction_lr==0):
            snake.direction_lr=-1
            snake.direction_ud=0
        elif(dir=="d" and snake.direction_ud==0):
            snake.direction_ud=-1
            snake.direction_lr=0
        elif(dir=="u" and snake.direction_ud==0):
            snake.direction_ud=1
            snake.direction_lr=0
    return(snake.score+snake.time)


net=Genetic_Network([14,16,16,4],["reLu","reLu","sigmoid"],200)

net.train(epoch=200,score=score,numb_ind=50,n_c=100,p=0.05,mu=0,sigma=1)

net.get_best(score)
