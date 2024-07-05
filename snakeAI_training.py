import numpy as np

class Snake_training(object):
    
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

class Map_training(object):

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
