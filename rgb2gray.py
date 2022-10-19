import cv2
import os

# path_ori='./data/train/ori/'
# path_gray='./data/train/gray/'
# lib=os.listdir(path_ori)

# for i in lib:
#     p=path_ori+i
#     # print(p)
#     img=cv2.imread(p)
#     img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     img_savepath=path_gray+i
#     cv2.imwrite(img_savepath,img)

def createGray(path_ori,path_gray):
    lis=os.listdir(path_ori)
    for i in lis:
        p=path_ori+i
        print(p)
        if p.endswith('.jpg'):
            img=cv2.imread(p)
            img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img_savepath=path_gray+i
            cv2.imwrite(img_savepath,img)
        else:
            continue
    
createGray('./data/validation/ori/','./data/validation/gray/')
createGray('./data/test/ori/','./data/test/gray/')
