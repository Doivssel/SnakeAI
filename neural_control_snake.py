import numpy as np
import tkinter as tk
from scipy.special import expit

class Network(object):

    def __init__(self,size,function):
        self.number_layer=len(size)
        self.size=size
        self.function=function
        self.weight=[np.array([[-7.21361131e-01, -1.70690304e+00,  3.91939369e+00,
        -2.44488881e+00, -2.36290694e-01, -2.97304233e+00,
         3.27335068e+00, -1.81951923e+00, -1.04940734e+00,
        -1.28919920e+00,  5.43527471e+00, -2.57277825e+00],
       [ 2.21607147e+00,  4.71793269e+00, -2.35401469e+00,
        -1.29020751e+00,  4.83532335e-01, -2.26609208e+00,
        -4.58911738e+00,  2.19813928e-01,  1.39294889e+00,
        -9.61122301e-01, -2.77378090e+00,  5.18553087e-01],
       [-7.26831897e+00, -7.75200143e-01,  3.04600575e+00,
        -2.38594166e+00, -3.04591413e+00, -1.05173846e-01,
        -1.62933348e+00, -3.16614338e-01,  3.27391373e+00,
         5.55669688e+00,  9.47985490e-01,  5.02777760e+00],
       [-5.08836311e+00,  4.79295354e-01, -7.19949418e+00,
        -1.27602389e-01, -8.12138859e+00,  2.51386857e+00,
        -6.10962993e-03,  4.81868333e+00,  1.67500379e+00,
        -1.63723956e+00, -3.87449125e+00,  9.68651365e-01],
       [ 2.06370031e+00, -7.32544959e-01,  3.08961489e-01,
         9.61590421e+00, -7.83434054e+00,  2.03673783e+00,
         1.17194829e+00,  7.30359928e-01, -3.06041328e+00,
        -7.81226056e+00, -1.10358747e+00, -1.87909575e+00],
       [ 5.22569699e+00, -1.81268055e+00, -5.26863255e-01,
        -5.19232099e+00, -5.84490527e+00,  8.50807640e+00,
        -5.26408196e-01,  3.94918999e+00,  4.59901458e+00,
        -3.31500332e+00,  4.01966614e-01, -1.45553769e+00],
       [ 1.72847078e+00, -5.20206537e+00,  2.81994144e+00,
         6.89573299e-01, -3.14625842e+00, -1.01761081e-01,
         4.57201141e-01, -1.73777547e+00, -5.39731262e+00,
         2.31190348e+00,  3.10274131e+00, -1.06103994e+00],
       [ 6.13144834e-01,  2.62575147e+00, -2.50568817e+00,
        -4.07070207e+00,  1.55946870e+00, -1.75022663e+00,
        -9.90880693e-01, -4.43773680e+00, -1.66465351e+00,
         2.02098324e+00, -6.65657887e+00, -1.08627489e+00],
       [-5.04615662e-01, -7.07407229e+00,  1.13782967e-01,
        -1.12680319e+00,  2.69022667e+00, -2.73379281e+00,
         3.43847548e+00,  3.92214000e-01,  2.75074920e+00,
        -4.22975808e+00,  2.24819835e+00,  4.87665653e+00],
       [ 5.73821781e+00,  1.15239854e+00, -2.79533998e+00,
         6.60134319e+00,  1.27374780e+00, -6.95625714e-01,
        -1.29481967e+00,  1.72969738e+00, -2.61010220e-01,
        -5.41496809e+00,  4.56833079e-02, -1.52563064e+00],
       [-3.13376389e+00, -1.53162760e-01, -3.72526052e+00,
         5.03750055e+00,  6.54129214e-01, -1.98746328e+00,
        -2.57853940e+00, -4.67690775e+00,  3.06146842e+00,
         2.16876544e+00, -5.95374331e+00, -9.88345625e-01],
       [ 1.79533947e+00, -2.95091032e+00, -2.43949745e+00,
        -9.60298610e-01,  8.88458740e-02,  7.52837440e+00,
        -3.99183960e-01,  8.25844915e-01, -8.02194883e+00,
         1.40958205e+00,  3.67553527e-01,  2.84680926e-02],
       [-5.50482764e-01,  1.20922464e+00, -6.54619107e-01,
         7.64727048e+00,  6.33148279e+00, -3.92729190e+00,
        -3.33688165e+00, -2.07711159e+00,  2.24918616e+00,
        -3.93382497e+00,  4.21453823e-01,  3.55548024e+00],
       [ 4.82063774e+00, -2.48200571e+00, -4.23650033e+00,
        -2.29289023e+00,  1.00307268e+01, -1.24564764e+00,
        -4.15506399e-03,  3.08390430e+00,  1.38562328e+00,
         5.39857539e-01, -1.37526392e-01, -2.10197741e+00],
       [-1.00201108e+00, -4.82713638e+00, -3.38111362e+00,
         2.79120585e+00,  2.82727982e+00, -7.33845077e+00,
        -4.24020897e+00,  4.36412477e+00,  5.61409755e+00,
         8.86497671e+00, -1.30360728e+00,  2.91978147e+00],
       [ 1.50812341e+00, -4.88834748e+00, -4.86772753e+00,
         1.17030434e+00,  3.05381574e-01, -9.35227962e-02,
         6.85001788e-01,  1.44663222e+00, -3.12321943e+00,
         3.52995187e+00,  2.10811374e+00,  1.24073209e-01],
       [-5.45537791e+00,  8.20139645e+00, -5.65979805e+00,
        -1.06084801e+00, -5.21351167e+00,  2.07053544e-01,
        -4.09470956e+00,  9.11814969e-01,  3.90082755e+00,
        -2.07212225e+00,  2.19358867e+00, -6.11410530e+00],
       [ 1.05103763e-01, -5.16515809e+00,  2.72405236e+00,
        -4.79994882e+00,  1.48239250e+00, -2.37430548e+00,
        -3.12042919e+00,  3.18561164e+00,  4.86855496e+00,
        -3.19756746e+00, -1.11463495e+00,  2.59234913e+00],
       [ 3.65872375e+00,  2.92467640e-01,  4.95050638e+00,
        -3.71537369e-01, -4.00113388e+00,  2.45648650e+00,
        -5.12626200e-01, -1.02291036e+00, -1.21312900e+00,
         2.75868206e+00,  8.16382959e+00, -1.67973891e+00],
       [-2.43387529e+00, -5.67095827e-01,  3.49338644e-01,
         3.02903427e+00,  1.50964067e+00, -2.82996505e+00,
         2.21621054e+00,  4.86662983e+00, -3.87800232e+00,
         6.79247633e-01,  6.39911554e-01,  2.08553867e+00]]), np.array([[ -5.27892736,   1.30140804,   5.79654253,  -0.47574761,
         13.9484327 ,   1.08528574,   1.19970423,  -9.59170047,
         -2.65896574,  -1.48196758,   4.41832712,  -0.97618052,
         -2.17515036,  -1.57869068,   0.70522146,  -3.61054492,
          3.21780346,   4.66670378,   6.45484488,  -2.51427007],
       [ -2.19177412,   4.60890347,   1.94451776,  -2.51999322,
          3.28483081,  -0.4368447 ,   6.49540883,   3.01951438,
         -2.22949561,  -1.58302647,  -3.01949927,  -2.49160637,
          5.28228204,  -0.76015801,  -4.27373315,  -5.5373581 ,
         -0.26481709, -11.07015123,  -5.02997724,  -4.80155965],
       [  0.04269306,   3.03410644,   3.21888292,  -2.24385549,
         -3.90776182,   1.10941624,  -5.72245205,   4.611783  ,
         -2.1048983 ,   2.5900966 ,  -3.56202338,  -4.29354339,
          1.06529424,   4.64978152,  -3.76150302,  -2.59649433,
         -8.42109647,  -1.4607465 ,   0.61997333,  -6.29706023],
       [ -0.18138891,   2.53344693,  -5.84528276,  -1.62567773,
         -1.69874227,  -3.60060062,   4.02726228,   1.97738409,
          2.30453085,  -0.42460184,   4.45072769,   3.1478776 ,
          6.4602628 ,   2.29616898,   1.09192969,  -5.18566275,
          4.86093196,   2.30120503,   0.25341449,  -5.01641785],
       [ -2.40042095,  -2.87241435,   2.36547348,   0.16153131,
          6.1620784 ,  -4.47942959,  -0.23591879,   5.0988589 ,
          2.28746572,  -1.13331474,  -1.26006218,  -8.47387286,
         -1.90232701,   1.03927298,  -2.16748997,   4.35399014,
          1.48243387,  -2.33224565,   0.3700468 ,  -1.21575388],
       [  2.76704355,  -3.79599564,   3.13094775,   0.11330894,
          3.10875266,   4.55553525,   1.66425418,   0.910768  ,
          5.16278928,  -1.47522815,  -1.02326949,   1.87377895,
          9.59253541,  -0.10399792,   2.8741019 ,   4.2781692 ,
          1.28796569,   0.29893127,   1.34613456,  -0.17308358],
       [  4.54658957,  -2.60804775,   1.5278855 ,  -1.02657568,
         -3.39674532,   7.32831383,   0.53261184,   2.05771686,
         -3.76985607,   0.86742501,  -2.38673129,   0.5360037 ,
          0.52466322,  -1.64714085,  -2.02530199,  -1.90716642,
         -0.23323651,  -0.64798253,   0.29009655,  -1.2735075 ],
       [  4.42139199,  -3.97369867,   1.36754341,   6.39181447,
          0.1940768 ,  -4.24683459,   2.93239753,   0.40739223,
          3.09095868,   2.61371744,  -4.85847924,  -1.4461497 ,
          4.13822478,  -5.2400551 ,   2.40262079,  -0.89317289,
         -2.99827613,  -4.2252123 ,  -6.70434266,  -4.21903724],
       [ -3.21349198,  -3.87379048,  -0.17772633,  -0.4478372 ,
         -6.95345854,   0.55525603,  -0.33551938,  -2.32678276,
         -1.34153764,  -0.93491457,   3.87059133,   7.02925466,
          4.76007186,  -1.71555829,  -6.69940023,   2.82960223,
         -0.21475536,   3.02688395,   4.11179861,   3.33314163],
       [ -2.62769309,   4.70534127,   5.80538087,   3.325333  ,
         -0.69820259,   3.92673963,   2.05524813,  -2.03910039,
          2.78351108,  -1.80509702,  -0.72404285,  -4.24598739,
         -3.43762231,   7.17576912,   1.10290496,   2.9492231 ,
         -0.35748706,   1.53159281,  -0.15320891,   2.9495063 ],
       [  0.78650809,  -3.41904347,  -3.01853795,   1.37255701,
          3.23978062,   1.0867222 ,  -2.75573267,  -4.44593761,
         -7.68054426,   1.23957667,   1.23082457,   1.97255287,
         -0.72680729,   2.69478192,  -0.60467656,   5.23394188,
         -3.48970331,   2.69618855,   2.85986256,   3.22584337],
       [ -2.87061283,   0.50693652,  -9.1862189 ,  -3.1950581 ,
          0.47458096,   7.79120117,  -1.44698783,   3.29802709,
         -4.5541752 ,   3.64097946,  -0.92396393,  -1.40317659,
          2.65398254,  -2.72155821,   1.37104757,   2.38872656,
          0.27537871,  -5.74109837,   7.10094906,  -0.57937866],
       [  0.31088499,   0.14356013,   1.3806906 ,  -6.79659784,
          0.14040164,   1.19431938,   2.78333047,  -1.49162563,
         -2.8273216 ,  10.26316622,  -5.30950651,   4.32733046,
         -4.19096602,  -6.60187195,  -7.01659622,  -4.79404701,
          1.60416284,   0.98866297,  -4.22097848,  -3.39226845],
       [  0.08437066,   2.41045516,   3.73017633,  -0.1796985 ,
          0.9168089 ,  -2.44783456,  -1.91800221,  -2.69725745,
         -4.53012407,  -4.16786191,   4.03154806,  -3.29741455,
          5.13140365,  -2.17273963,  -2.65144164,   1.25702084,
          1.28930962,   8.72182346,   0.54710516,  -5.26313025],
       [ -0.52085185,  -3.00177485,   7.57071238,   2.25495964,
          1.1817643 ,   6.90223412,   0.8930469 ,  -6.41735177,
         -2.16213591,  -7.58137049,  -2.53736395,   1.16203289,
         -1.35669303,  -0.40716039,   8.1323899 ,  -0.47183545,
         -3.53618566,   2.72198977,   0.63399389,   1.9475105 ],
       [ -0.70474042,  -3.85174051,  -3.61304413,  -1.86963967,
          3.38572613,  -5.05392214,  -2.40770418,  -6.38238388,
          2.10471374,   3.40138615,   2.52200483,   1.3326916 ,
          6.5444532 ,  -0.32637468,   0.87820569,   2.34510338,
          5.82194747,   1.75551696,   0.08905717,  -3.79071777],
       [ -5.14916827,   3.89060655,  -3.06958358,   2.64949559,
          0.79469578,  -1.84903665,   3.22409124,  -0.77196368,
         -1.83449837,  -0.60395047,  -2.20260629,  -1.83717626,
          2.44550871,   0.01941606,  -1.3263053 ,   1.18562693,
          4.29408292,   2.87553922,   0.82861877,   4.9322375 ],
       [ -0.16254467,  -4.19215322,   4.61944759,  -2.25885891,
          4.2637043 ,   4.55200713,   1.49623051,  -2.19279644,
          0.63528994,   6.97887853,   3.63156955,  -0.67498178,
          0.89568353,  -1.97502946,   2.62699758,  -1.43911569,
         -1.13791412,  -2.84591267,  -6.15119921,   1.85773575],
       [  0.13599247,  -2.19583093,  -0.5439769 ,   3.66316517,
          6.72657374,   4.44888671,   0.18788783,   1.16471173,
          1.49097929,  -1.19477548,   2.4588137 ,   0.27561713,
         -5.32780866,   5.05958866,  -2.63684231,   1.06357957,
         -3.04081019,   0.39097938,   4.16077817,  -5.85779917],
       [  0.82402432,  -5.36640716,  -2.59593745,   3.66262204,
         -2.794307  ,   2.43894215,  -4.64233222,  -0.63164828,
          0.83536652,   2.17922832,   2.39700374,  -0.63259394,
          3.78277019,   8.70398912,   2.21323687,   1.23663041,
          6.11139653,  -3.52176823,  -6.70180356,   2.24394098]]), np.array([[-2.21360619,  0.55836038,  4.8268305 ,  3.21793384, -0.15698091,
        -3.98743332, -4.6643802 ,  0.54475115, -0.27488144,  3.35175488,
         1.48325887, -3.35464297,  2.03860232,  1.00787346, -2.24810029,
        -0.18341637, -0.01018247, -4.66533453, -3.01373758,  1.89813256],
       [ 5.1473886 ,  2.61304051,  6.1475549 ,  0.51468039, -3.05411939,
        -4.67409234, -3.94788431,  5.78103129,  2.05889263, -3.53533914,
         2.18185574, -1.95683472, -4.64615538,  2.82308012, -0.32417806,
        -2.89762357,  0.45918483,  0.41071742,  2.32777908, -1.10484304],
       [-4.7331919 ,  0.64976169, -3.0930889 , -1.37806737,  4.96201164,
        -2.63118713,  5.64215343, -2.54507101,  3.58819821,  7.28243342,
        -0.26021901, -1.46781304,  1.936839  ,  5.24688103,  7.15881324,
        -2.6569155 , -1.75295528,  3.41661066,  1.25189767, -3.40073627],
       [-3.15827854, -0.71186747, -7.24018591,  7.70221286,  1.03089664,
         4.56986875, -1.80062223,  1.95473968, -2.99221026, -7.23152903,
        -3.46342076,  4.11992697, -1.20061623,  7.03177976,  0.51365509,
         0.25205406,  0.981746  ,  3.10021396, -2.80918876,  3.13714495]])]


    def sigmoid(self,z):
        return(1/(1+expit(-z)))
    
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
            signal_layers.append(np.dot(self.weight[i],activation_layers[i]))
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
        self.dir=["u","d","l","r"]
        self.direction_lr,self.direction_ud=self.init_dir(map)
        self.posX,self.posY=self.init_pos(map)
        self.score=0
        self.body=np.array([[self.posX,self.posY]])
        self.tail=self.body[0]
        self.border=[map.number_col-self.posX,map.number_row-self.posY]
        map.create_body_part(self)
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
            lose()
        else:
            map_value=map.map[self.posY,self.posX]
            if(map_value==1):
                lose()
            elif(map_value==0):
                map.map[self.posY,self.posX]=1
                self.body=np.vstack((self.body,[self.posX,self.posY]))
                map.delete_body_part(self)
                map.create_body_part(self)
                map.map[self.body[0,1],self.body[0,0]]=0
                self.body=np.delete(self.body,0,0)
            elif(map_value==2):
                self.score+=1
                map.add_apple()
                map.map[self.posY,self.posX]=1
                self.body=np.vstack((self.body,[self.posX,self.posY]))
                map.create_body_part(self)
                t_canvas.delete("temp_text")
                t_canvas.create_text(300,40,text="current score : "+str(self.score),font=("Helvetica 25 bold"),tags="temp_text")




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
        self.grid_Y=np.linspace(0,600,self.number_row+1)[0:self.number_row]
        self.grid_X=np.linspace(0,600,self.number_col+1)[0:self.number_col]


    def create_body_part(self,snake):
        """
        Parameters:
        snake: instance of the class Snake

        Goal: 
        Update the graphical representation of the snake by adding
        a body part.
        """
        canvas.create_rectangle(self.grid_X[snake.posX],self.grid_Y[snake.posY],self.grid_X[snake.posX]+600/self.number_col,self.grid_Y[snake.posY]+600/self.number_row,fill="blue")

    def delete_body_part(self,snake):  
        """
        Parameters:
        snake: instance of the class Snake

        Goal: 
        Update the graphical representation of the snake by removing
        a body part.

        Note:
        Could be fuse with the above function but I find it clearer this way
        """
        canvas.create_rectangle(self.grid_X[snake.body[0,0]],self.grid_Y[snake.body[0,1]],self.grid_X[snake.body[0,0]]+600/self.number_col,self.grid_Y[snake.body[0,1]]+600/self.number_row,fill="black")

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
        canvas.create_rectangle(self.grid_X[self.rdm_pos[0]],self.grid_Y[self.rdm_pos[1]],self.grid_X[self.rdm_pos[0]]+600/self.number_col,self.grid_Y[self.rdm_pos[1]]+600/self.number_row,fill="red")



def playing(net):
    snake.move(map)
    data=np.array([ [snake.tail[0]],
                        [snake.tail[1]],
                        #Distance to the wall in the cross direction centerd around the head
                        # [(snake.posX<1)*1],
                        # [(snake.posY<1)*1],
                        # [(map.number_col-snake.posX>9)*1],
                        # [(map.number_row-snake.posY>9)*1],
                        [snake.posX],
                        [snake.posY],
                        [map.number_col-snake.posX],
                        [map.number_row-snake.posY],
                        #Apple in the cross direction centerd around the head
                        # [(map.rdm_pos[0]==snake.posX and map.rdm_pos[1]>0 )*1],
                        # [(map.rdm_pos[0]==snake.posX and map.rdm_pos[1]<0 )*1],
                        # [(map.rdm_pos[1]==snake.posX and map.rdm_pos[0]>0 )*1],
                        # [(map.rdm_pos[1]==snake.posX and map.rdm_pos[0]<0 )*1],
                        [np.abs(snake.posX-map.rdm_pos[0])],
                        [np.abs(snake.posY-map.rdm_pos[1])],
                        #These are for the current direction 
                        [(snake.direction_lr<0)*1],
                        [(snake.direction_lr>0)*1],
                        [(snake.direction_ud<0)*1],
                        [(snake.direction_ud>1)*1]])
    output=net.feedforward(data)[-1]
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
    root.after(100,playing,net)

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
    canvas.delete("all")
    t_canvas.delete("all")
    map=Map(10,10)
    snake=Snake(map)


root=tk.Tk()
root.geometry("600x700")

canvas=tk.Canvas(root,width=600,height=600,bg="black")
canvas.pack()

t_canvas=tk.Canvas(root,width=600,height=100,bg="yellow")
t_canvas.pack()

map=Map(10,10)
snake=Snake(map)

net=Network([12,20,20,4],["reLu","reLu","sigmoid"])
playing(net)
root.mainloop()
