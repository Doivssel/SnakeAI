from snakeAI_training import *
from snakeAI_visual import *
from genetic_neural_nets import *

def score(weight,gen_net):
    map=Map_training(map_dim,map_dim)
    snake=Snake_training(map)
    while(snake.lost==False):
        snake.move(map)
        data=np.array([ [snake.tail[0]],
                        [snake.tail[1]],
                        #Distance to the wall in the cross direction centerd around the head
                        [snake.posX],
                        [snake.posY],
                        [map.number_col-snake.posX],
                        [map.number_row-snake.posY],
                        #Apple in the cross direction centerd around the head
                        [(map.rdm_pos[0]>snake.posX)*1],
                        [(map.rdm_pos[0]<snake.posX)*1],
                        [(map.rdm_pos[1]>snake.posY)*1],
                        [(map.rdm_pos[1]<snake.posY)*1],
                        #These are for the current direction 
                        [(snake.direction_lr<0)*1],
                        [(snake.direction_lr>0)*1],
                        [(snake.direction_ud<0)*1],
                        [(snake.direction_ud>1)*1]])
        output=gen_net.feedforward(data,weight)[-1]
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
    return(10*snake.score+snake.time)

map_dim,size,function=10,[14,20,20,4],["reLu","reLu","sigmoid"]

gen_net=Genetic_Network(size,function,2000)
gen_net.train(epoch=200,score=score,numb_ind=700,n_c=50,p=0.1,mu=0,sigma=2)
best_net=gen_net.get_best(score)


root=tk.Tk()
net=Network(size,function,best_net)
game=Game(root,map_dim,80,net)
game.root.mainloop()
