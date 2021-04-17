from __future__ import print_function
import numpy as np
import os
import scipy.misc as misc
import tensorflow as tf
beijing = [0,0,0]
yuanzhuxiechilunzhou = [239,235,143]
zhou = [243,0,193]
yuanzhuxiechilun = [100,50,100]
xiangti = [255,0,0]
zhouchengduangai = [194,243,0]
guayouban = [64,191,175]
taotong = [0,243,97]
wolun = [96,255,223]
wogan = [43,0,215]
luoshuan= [96,128,255]
xiangjiaoquan = [0,188,0]
zhoucheng = [243,113,165]
yuanzhuixiechilun = [13,113,243]
zhoutao = [188,75,0]
yuanzhuzhichichilunzhou = [255,159,15]
yuanzhuzhichilun = [0,0,243]
classes = [ '箱体', '圆柱直齿轮', '轴承', '轴承端盖',
           '轴', '圆柱直齿齿轮轴', '圆锥斜齿轮', '轴套', '橡胶圈', '圆柱斜齿轮', '螺栓',
           '套筒', '蜗杆', '蜗轮', '刮油板', '圆柱斜齿轮轴','background']

colormap = [[255,0,0], [0, 0, 243], [243, 113, 165], [194, 243, 0], [243, 0, 193],
            [255, 159, 15], [13, 113, 243], [188, 75, 0], [0, 188, 0], [100, 50, 100],
            [96, 128, 255], [0, 243, 97], [43, 0, 215], [96, 255, 223],
            [64, 191, 175], [239, 235, 143],[0, 0, 0]]
COLOR_DICT = np.array([[255,0,0], [0, 0, 243], [243, 113, 165], [194, 243, 0], [243, 0, 193],
            [255, 159, 15], [13, 113, 243], [188, 75, 0], [0, 188, 0], [100, 50, 100],
            [96, 128, 255], [0, 243, 97], [43, 0, 215], [96, 255, 223],
            [64, 191, 175], [239, 235, 143],[0, 0, 0]])
def labelVisualize(num_class,color_dict,img):
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out



def saveResult(save_path,item,flag_multi_class = True,num_class = 17,i= None):
    img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
    misc.imsave(os.path.join(save_path,"%d_predict.png" % i), img)



