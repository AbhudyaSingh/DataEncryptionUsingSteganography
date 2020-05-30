import numpy as np
def rebuild (block):
    global  hPath
    idx = 0
    blockx = [[0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            pos = hPath[idx] - 1
            posx = int(pos / 3)
            posy = int(pos % 3)
            val = block0[i][j]
            blockx[posx][posy] = val
            # print(blockx[posx][posy] , block0[i][j])
            # print(idx)
            idx += 1
    return blockx

hPath = [1, 2, 3, 6, 9, 8, 7, 4, 5]
windowsize= 3
block0=[[1 ,2, 3],
        [4 ,5 ,6],
        [7,8 ,9]]
block90 = [[3, 8, 5],
           [2,9,4],
           [1, 6, 7]]
block180=[[5 ,4, 7],
          [8 ,9 ,6],
          [3 ,2 ,1]]
block270=[[7 ,6, 1],
          [4 ,9 ,2],
          [5 ,8 ,3]]
s0   = "000000000"
s90  = "000000010"
s180 = "000010000"
s270 = "001100000"
for idx in range(9):
    posx = int((hPath[idx] - 1) / windowsize)
    posy = (hPath[idx] - 1) % windowsize
    #print('Embed ', s270[idx], 'to', block270[posx][posy])
num = 162
for i in range (8):
    n  = (num & (1<<i))
    if n>0 :
        n=1
    #print (n)
num = num ^ (1 << 2)
num =162
num =num ^ (1<<0)
# print (num)
q = [0 for _ in range(5)]
q[0]=125
#print (sum(q))
new_dict ={}
idx = 0
block0=[[1, 2, 3],
        [6, 9, 8],
        [7, 4, 5]]
print(block0)

block0 = rebuild(block0)
print(block0)
