from typing import List

import cv2 as cv
import numpy as np
import random
import math
from matplotlib import pyplot as plt
def wrap(s, w):
    return [s[i:i + w] for i in range(0, len(s), w)]
def rebuild (block):
    global  hPath
    idx = 0
    blockx = [[0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            pos = hPath[idx] - 1
            posx = int(pos / 3)
            posy = int(pos % 3)
            val = block[i][j]
            blockx[posx][posy] = val
            # print(blockx[posx][posy] , block0[i][j])
            # print(idx)
            idx += 1
    return blockx
def createBorder():
    global img
    global row
    global col

    borderType = cv.BORDER_CONSTANT
    TDLU = [0, 1, 0, 1]
    img = cv.copyMakeBorder(img, TDLU[0], TDLU[1], TDLU[2], TDLU[3], borderType, None, 255)
    row, col = img.shape
    #print(row, col)

def draw_hist(img, imgx):
    hist_img = cv.calcHist([img], [0], None, [256], [0, 256])
    hist_imgx = cv.calcHist([imgx], [0], None, [256], [0, 256])
    plt.subplot(221), plt.imshow(img, 'gray')
    plt.subplot(222), plt.imshow(imgx, 'gray')
    plt.subplot(223), plt.plot(hist_img)
    plt.subplot(224), plt.plot(hist_imgx)
    plt.plot(hist_imgx)
    plt.xlim([0, 256])
    plt.show()
    # plt.hist(imgx.ravel(), 256 ,[0,256])
    # plt.show()
def generateRotations(chunks):
    global dataRotations
    global windowSize
    for chunk in chunks:
        Sn = [['0', '0'] + chunk[:],
                chunk[:3 * windowSize-3]+['0', '1'] + chunk[3 * windowSize - 3:],
                chunk[:2 * windowSize - 2] + ['1', '0'] + chunk[2 * windowSize-2:],
                chunk[:windowSize-1] + ['1', '1'] + chunk[windowSize-1:]

             ]
        dataRotations.append(Sn)


def preProcess():
    global img
    global row
    global col
    global windowSize
    global hPath
    global hBlock
    blocks = []
    hBlock = []
   # print(row, col)
    for r in range(0, int(row / windowSize)):
        for c in range(0, int(col / windowSize)):
            # print('block', r, c)
            window = img[r * windowSize:r * windowSize + windowSize, c * windowSize:c * windowSize + windowSize]
            blocks.append(window)
    # print ('original block ', blocks[0])
    for win in blocks:
        block = np.zeros([windowSize, windowSize], dtype=int)
        for idx in range(0, windowSize * windowSize):
            posx = int((hPath[idx] - 1) / windowSize)
            posy = (hPath[idx] - 1) % windowSize
            px = int(idx / windowSize)
            py = idx % windowSize
            block[px][py] = win[posx][posy]
        hBlock.append(block)


def rotate90(block):
    row, col = block.shape

    for i in range(0, row):
        k = col - 1
        for j in range(0, int(col / 2)):
            block[i][j], block[i][k] = block[i][k], block[i][j]
            k -= 1

    for r in range(0, row):  # transpose
        for c in range(r, col):
            block[c][r], block[r][c] = block[r][c], block[c][r]

    return block
def PSNR(img , imgECO):
    MSE = 0
    mx = -1
    for i in range(row):
        for j in range(col):
            mx = max(mx , int(img[i][j]), (int (imgECO[i][j]) ))
            MSE += (int(img[i][j]) - (int (imgECO[i][j])))**2
    MSE = 1.0 * ( MSE /(row * col ))

    PSNR = 20 * math.log10(mx/ MSE)
    print (PSNR)

def Embed_ECO(block, pr, sd):
    global hBlock
    global plane
    for i in range(pr):
        block = rotate90(block)
    # print ('rotated  best ', pr* 90,block)
    for lsb in range(plane):
        for idx in range(0, windowSize*windowSize):
            posx = int((hPath[idx] - 1) / windowSize)
            posy = (hPath[idx] - 1) % windowSize
            Lblock = int(block[posx][posy])
            # print('Mapped ', idx, posx, posy)
            LblockBit = Lblock & (1 << lsb)
            if LblockBit > 0:
                LblockBit = 1
            strBit = int (sd[idx])
            if LblockBit != strBit:
               Lblock = Lblock ^ (1 << lsb)
            block[posx][posy] = Lblock
    # print('EHB', block)
    block = rebuild(block)
    # print('OEB', block)
    # print('fn ', block)
def calculate_ECO(block, rot, bcnt ):
    curCost = 0
    for i in range(0, plane):
        pow = 2 ** i
        for idx in range(0, (windowSize * windowSize)):
            posx = int((hPath[idx] - 1) / windowSize)
            posy = (hPath[idx] - 1) % windowSize
            blockBit = int(block[posx][posy]) & (1 << i)
            if blockBit > 0:
                blockBit = 1

            strBit = int(dataRotations[bcnt][rot][idx])
            # print (idx , blockBit,strBit, block[posx][posy])
            curCost += abs(blockBit - strBit)
        curCost = curCost * pow
    # print ("Current Cost of embedding ", curCost)
    return curCost

def ECO():
    global hBlock # WxW
    global dataRotations # (WxW rows)x 4 column x(WxW length)
    global plane
    global img
    global hPath
    copy_hblock = hBlock[:]
    imgECO = img[:]
    minCost = 1000000
    selectR = 0
    selectD = 0
    bcnt = 0
    # print('copy of hblock ', copy_hblock[0])
    for block in copy_hblock:
        # print('Block', block)
        for rot in range(4):
            # print ("Rotated block ", rot * 90 ,  block )#0 , 90 , 180, 270
            # print ("Corresponding Rotated bitstring  ", dataRotations[bcnt][rot])
            cur_Cost = calculate_ECO(block[:], rot, bcnt)
            # print('Current Cost ', cur_Cost)
            if cur_Cost < minCost:
                minCost = cur_Cost
                selectR = rot
                selectD = dataRotations[bcnt][rot]
            block = rotate90(block)  # 90,180,270,360
        # print ("Minimum Cost", minCost, selectR, selectD)
        # print('Before ',block) 0 degree
        Embed_ECO(block, selectR, selectD) # 0 degree block passed
        # print('After ',block)
        # print('copy_block after operation ', copy_hblock[bcnt])
        bcnt += 1


        # if bcnt == 1:
            # print('Embedded block ', block)
            # break

    cnt = 0
    for r in range(0, int(row / windowSize)):
        for c in range(0, int(col / windowSize)):
            # print('block', r, c)
            imgECO[r * windowSize:r * windowSize + windowSize, c * windowSize:c * windowSize + windowSize] = copy_hblock[cnt]
            cnt +=1

    # print('cnt ', cnt )
    # print("image x", imgx[: windowSize,:windowSize ] )
    draw_hist(img, imgECO)
    cv.imshow('Encrypted ECO Image ', imgECO)
    # cv.imwrite('Encrpted_ECO image.jpg', imgx)
    cv.waitKey(50000)
    cv.destroyAllWindows()
    # PSNR(img ,imgECO)
def calculate( block, bcnt , rot, hell):
    global plane
    global windowSize
    global hPath
    for i in range(0, plane):
        for idx in range(0,(windowSize * windowSize )):
            posx = int( ( hPath[idx] - 1) / windowSize)
            posy = ( hPath[idx] - 1) % windowSize
            blockValue = int(block[posx][posy])
            blockBit = blockValue & (1 << i)
            if blockBit > 0:
                blockBit = 1

            strBit = int(dataRotations[bcnt][rot][idx])
            # print(idx, blockBit, strBit, block[posx][posy])
            if blockBit != strBit:
                chblockValue = blockValue ^ (1 << i)
                hell[blockValue] -= 1
                hell[chblockValue] += 1
    # print('cost', sum(hell))
    return hell

def sum (x):
    s = 0
    for itr in x:
       s += abs(itr)
    return s


def Embed_HDM(block, selectR, selectD):
    global hBlock
    global plane
    for i in range(selectR):
        block = rotate90(block)
    # print ('rotated  best ', pr* 90,block)
    for lsb in range(plane):
        for idx in range(0, windowSize * windowSize):
            posx = int((hPath[idx] - 1) / windowSize)
            posy = (hPath[idx] - 1) % windowSize
            Lblock = int(block[posx][posy])
            # print('Mapped ', idx, posx, posy)
            LblockBit = Lblock & (1 << lsb)
            if LblockBit > 0:
                LblockBit = 1
            strBit = int(selectD[idx])
            if LblockBit != strBit:
                Lblock = Lblock ^ (1 << lsb)
            block[posx][posy] = Lblock
    block = rebuild(block)

def HDM():
    global hBlock  # WxW
    global dataRotations  # (WxW rows)x 4 column x(WxW length)
    global plane
    global hPath
    global img
    copy_block = hBlock[:]
    queue = [0] * 256
    imgHDM = img[:]
    selectR = 0
    selectD = 0
    bcnt = 0
    for block in copy_block:
        minW = 10000000
        queueS = [0]*256 # selected for block
        for rot in range(0, 4):
            # print("Rotated block ", rot*90 )#0 , 90 , 180, 270
            # print("Corresponding Rotated bitstring  ", dataRotations[bcnt][rot])
            curq = queue
            cur_x = calculate(block, bcnt, rot, curq[:])
            # print(sum(cur_x), sum(curq))
            if sum(cur_x) < minW:
                # print('Min Updated')
                minW = sum(cur_x)
                queueS = cur_x
                selectR = rot
                selectD = dataRotations[bcnt][rot]
            block = rotate90(block)  # 90,180,270,360

        # print('block ended',bcnt)
        queue = queueS
        # print(' hamiltoninan block' , block)
        Embed_HDM(block, selectR, selectD)
        # print(' embedded block', block)
        # block = rebuild(block)
        # print(' EH block', block)

        bcnt += 1
    #     if bcnt == 1:
    #         break
    # # print ('ca',copy_block[0] )
    cnt = 0
    for r in range(0, int(row / windowSize)):
        for c in range(0, int(col / windowSize)):
            # print('block', r, c)
            img[r * windowSize:r * windowSize + windowSize, c * windowSize:c * windowSize + windowSize] = copy_block[cnt]
            cnt += 1
    draw_hist(img , imgHDM)
    cv.imshow('Encrypted HDM Image ', imgHDM)
    # cv.imwrite('Encrpted_image.jpg', imgHDM)
    cv.waitKey(50000)
    cv.destroyAllWindows()

def main():
    global img
    global row
    global col
    global windowSize
    global plane
    global hPath
    global dataRotations
    global hBlock
    path = "lena.jpg"
    img = cv.imread(path, 0)
    row, col = img.shape
    windowSize = 3
    plane = 1
    hPath = [1, 2, 3, 6, 9, 8, 7, 4, 5]
    data = ['1', '0', '0', '1', '1', '1', '0']
    for i in range(plane*row *col -7):
        data.append(random.choice(['0', '1']))
    # data = [random.choice(['0', '1']) for _ in range(plane*row * col)]
    chunkSize = (plane*windowSize*windowSize)- 2
    chunks = wrap(data, chunkSize)
    if row % windowSize > 0:
        createBorder()
    dataRotations = []
    generateRotations(chunks)
    preProcess()
    ECO()
    # HDM()
if __name__ == "__main__":
    main()