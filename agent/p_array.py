import numpy as np
import time

def func(original,new,car,maxCar,speedRange,agent):
    if car == maxCar:
        # compute p_product
        #print(original,new)
        p_action = 1/(speedRange[1]-speedRange[0]+1)
        p_product = 1
        for car in range(maxCar):
            p_product *= p_action
            if car != 0:
                if new[car] <= new[car-1]:
                    return 0
                elif new[car] == (new[car-1]+1):
                    p_product *= (new[car]-(original[car]+speedRange[0])+1)
        #print(p_product)
        return p_product
    else:
        sum_ = 0
        minSpeed = speedRange[1]
        maxSpeed = speedRange[0]
        for speed in range(maxSpeed,minSpeed+1):
            new[car] = original[car]+speed
            if agent >= new[car] and agent <= (original[car]+minSpeed):
                continue
            sum_ += func(original,new,car+1,maxCar,speedRange,agent)
        return sum_

def main(y):
    temp = np.array([0,0,0,1,0,1,0,0,0,1,0,1,1,0,0,0,0,0,1,0,1,0,0,0,1],dtype=np.intc)
    # y+3 the last grid to include
    # if y+3 > 50, then include remainders from the start
    cut = temp[y:y+4]
    if (y+4)>len(temp):
        cut = np.append(cut,temp[:(y+4)%len(temp)])
    original = []    
    for i in range(len(cut)):
        if cut[i] == 1:
            original.append(i)
    print(cut,original)
    original = np.array(original,dtype=np.intc)
    new = np.zeros(len(original),dtype=np.intc)
    speedRange = [-3,-1]
    print(func(original,new,0,len(original),speedRange,0),y)
    #func(original,new,0,len(original),speedRange,y)


for i in range(25):
    start = time.time()
    main(i)
    print(time.time()-start)

