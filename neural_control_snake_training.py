import numpy as np

class Network(object):

    def __init__(self,size,function):
        self.number_layer=len(size)
        self.size=size
        self.function=function
        self.bias=[np.random.uniform(low=-1,high=1,size=(num_neurons,1)) for num_neurons in size[1:]] 
        self.weight=[np.random.uniform(low=-1,high=1,size=(size[i+1],size[i])) for i in range(self.number_layer-1)] 
        
    def sigmoid(self,z):
        return(1/(1+np.exp(-z)))
    
    def tanh(self,z):
        return(2*self.sigmoid(z)-1)
    
    def activation_apply(self,z,fun):
        if(fun=="reLu"):
            return(np.maximum(0,z))
        if(fun=="sigmoid"):
            return(self.sigmoid(z))
        if(fun=="tanh"):
            return(self.tanh(z))
        else:
            print("Unknow function, choices are:  \n reLu, sigmoid, tanh")
            raise SystemExit(0)

    def der_apply(self,z,fun):
        if(fun=="reLu"):
            return(np.where(z>0,1,0))
        if(fun=="sigmoid"):
            return(self.sigmoid(z)*(1-self.sigmoid(z)))
        if(fun=="tanh"):
            return(1-np.power(self.tanh(z),2))

    def feedforward(self,input):
        """For each layer except the input one we compute the signal the neurons
        as the sum of the before layer signals times the weights of the concerned neuron 
        plus the bias of the concerned neurons and store these values
        """
        signal_layers=[np.array(input)]
        activation_layers=[np.array(input)]
        for i in range(self.number_layer-1):
            signal_layers.append(np.dot(self.weight[i],activation_layers[i])+self.bias[i])
            activation_layers.append(self.activation_apply((signal_layers[i+1]),self.function[i]))
        return(activation_layers)

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
        self.direction_ud=0
        self.direction_lr=1
        self.posX=0
        self.posY=0
        self.score=0
        self.body=np.array([[self.posX,self.posY]])
        self.tail=self.body[0]
        self.border=[map.number_col-self.posX,map.number_row-self.posY]
        map.map[self.posY,self.posX]=1
        map.add_apple()


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
            lose()
        else:
            map_value=map.map[self.posY,self.posX]
            if(map_value==1):
                lose()
            elif(map_value==0):
                map.map[self.posY,self.posX]=1
                self.body=np.vstack((self.body,[self.posX,self.posY]))
                map.map[self.body[0,1],self.body[0,0]]=0
            elif(map_value==2):
                self.score+=1
                map.map[self.posY,self.posX]=1
                self.body=np.vstack((self.body,[self.posX,self.posY]))

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

def lose():
    """
    Parameters:
    None

    Goal:
    When the snake does an error the canvas is cleared. And 
    a new instance of snake and map are created as way to set the game back
    to zero.
    """
    global snake
    global map
    map=Map(15,15)
    snake=Snake(map)

allowed_move=200
actual_move=0

map=Map(15,15)
snake=Snake(map)

net=Network([8,2,1],["reLu","reLu","reLu"])

while(actual_move<allowed_move):
    snake.move(map)
    data=np.array([[snake.posX],
                   [snake.posY],
                   [snake.tail[0]],
                   [snake.tail[1]],
                   [snake.border[0]],
                   [snake.border[1]],
                   [map.rdm_pos[0]],
                   [map.rdm_pos[1]]])
    output=net.feedforward(data)[-1][0]
    if(0<=output<1 and snake.direction_lr==0):
        snake.direction_lr=1
        snake.direction_ud=0
    if(1<=output<2 and snake.direction_lr==0):
        snake.direction_lr=-1
        snake.direction_ud=0
    if(2<=output<3 and snake.direction_ud==0):
        snake.direction_ud=-1
        snake.direction_lr=0
    if(3<=output<4 and snake.direction_ud==0):
        snake.direction_ud=1
        snake.direction_lr=0
    print(snake.posX,snake.posY)
    actual_move+=1