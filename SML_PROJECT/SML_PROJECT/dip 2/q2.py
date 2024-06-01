import cv2
import numpy as np
import random 
import math

camera_man = cv2.imread('cameraman.png', cv2.IMREAD_GRAYSCALE)
camera_man = camera_man[:512, :512]


n,m=camera_man.shape
totl_pixels=n*m

def Psnr(distorted_img,original_img):
    mse = np.mean((original_img.astype(np.float64) - distorted_img.astype(np.float64)) ** 2)
    psnr = 10*math.log10((255 ** 2) / mse)

    return psnr

def sinc(x):
    if x == 0:
        return 1.0
    return np.sin(np.pi * x) / (np.pi * x)

def downsize(input_image,dimensions):
    row,col=input_image.shape
    scale=row//dimensions
    output=np.zeros((dimensions,dimensions),dtype=int)

    for i in range(dimensions):
        for j in range(dimensions):
            x_original=i*scale
            y_original=j*scale
            output[i][j]=input_image[x_original][y_original]

    return output

def NN(input_image):
    row,col=input_image.shape
    output=np.zeros((row*8,col*8),dtype=int)
    newrow,newcol=row*8,col*8
    for i in range(newrow):
        for j in range(newcol):
            x_original=i//8
            y_original=j//8
            output[i][j]=input_image[x_original][y_original]

    return output

def linear(input_image, output_size):
    output = np.zeros((output_size[1], output_size[0]), dtype=int)
    wd_sc,ht_sc = input_image.shape[1] / output_size[0],input_image.shape[0] / output_size[1]

    for x in range(output_size[0]):
        for y in range(output_size[1]):
            input_x = int(x * wd_sc)
            input_y = int(y * ht_sc)
            #print(input_x," ",input_y)
            
            dx, dy = x * wd_sc - input_x, y * ht_sc - input_y
            new_pixel =0 
            p00 = input_image[input_x, input_y]
            new_pixel+=(1-dx)*(1-dy)*p00
            p01=0
            if(input_x<63):
                p01 = input_image[input_x+ 1, input_y]
            new_pixel+=dx * (1 - dy) * p01
            p10=0
            if(input_y<63):
                p10 = input_image[input_x, input_y + 1]
            new_pixel+=(1-dx) * ( dy) * p10
            p11=0
            if(input_x<63 and input_y<63):
                p11 = input_image[input_x + 1, input_y + 1]
            new_pixel+=(dx) * ( dy) * p11
            

            output[x,y] = new_pixel

    return output

def hermite(input_image, output_size):
    output = np.zeros((output_size[1], output_size[0]), dtype=int)
    wd_sc,ht_sc = input_image.shape[1] / output_size[0],input_image.shape[0] / output_size[1]

    for x in range(output_size[0]):
        for y in range(output_size[1]):
            input_x = int(x * wd_sc)
            input_y = int(y * ht_sc)
            #print(input_x," ",input_y)
            
            dx, dy = x * wd_sc - input_x, y * ht_sc - input_y

            # new_pixel = (
            #     (1 - 3 * dx**2 + 2 * dx**3) * input_image[y0, x0] +
            #     dx * (1 - dx) ** 2 * (wd_sc * input_image[y0, x0 + 1] - input_image[y0, x0]) +
            #     dx**2 * (3 - 2 * dx) * input_image[y0 + 1, x0] -
            #     dx**2 * (1 - dx) * (wd_sc * input_image[y0 + 1, x0 + 1] - input_image[y0 + 1, x0])
            # )
            new_pixel =0 
            p00 = input_image[input_x, input_y]
            new_pixel+=(1 - 3 * dx**2 + 2 * dx**3)*p00
            p01=0
            if(input_x<63):
                p01 = input_image[input_x+ 1, input_y]
                new_pixel+=dx**2 * (3 - 2 * dx) * p01
                new_pixel+=dx**2 * (1 - dx) *(p01)

            new_pixel-=dx * (1 - dx) ** 2 *(p00)
            p10=0
            if(input_y<63):
                p10 =+input_image[input_x, input_y + 1]
                new_pixel+=dx * (1 - dx) ** 2 *(wd_sc * p10)
            
            p11=0
            if(input_x<63 and input_y<63):
                p11 = input_image[input_x + 1, input_y + 1]
                new_pixel-=dx**2 * (1 - dx) *(wd_sc*p11)
            

            output[x,y] = new_pixel

    return output

def cubic_kernel(t):
                a = -0.5
                t = np.abs(t)
                if t <= 1:
                    return (a + 2) * t**3 - (a + 3) * t**2 + 1
                elif 1 < t <= 2:
                    return a * t**3 - 5 * a * t**2 + 8 * a * t - 4 * a
                else:
                    return 0
                
def bicubic(img, output_size):
    height, width = img.shape

    output = np.zeros((output_size[1], output_size[0]))

    wd_sc = width / output_size[0]
    ht_sc = height / output_size[1]

    for x in range(output_size[0]):
        for y in range(output_size[1]):
            input_x = x * wd_sc
            input_y = y * ht_sc

            x0, y0 = int(input_x), int(input_y)
            dx, dy = input_x - x0, input_y - y0

            new_pixel = 0

            for v in range(-1, 3):
                for u in range(-1, 3):
                    weight_x = cubic_kernel(dx - u)
                    weight_y = cubic_kernel(dy - v)

                    data_x = min(max(x0 + u, 0), width - 1)
                    data_y = min(max(y0 + v, 0), height - 1)

                    new_pixel += img[data_x, data_y] * weight_x * weight_y

            output[x, y] =int(np.round(new_pixel))

    return output


def lanczos(input_image, output_size):
    a=2
    input_height, input_width = input_image.shape
    [output_width, output_height] = output_size

    x_scale = input_width / output_width
    y_scale = input_height / output_height
    output_image = np.zeros((output_height, output_width))

    for x in range(output_height):
        for y in range(output_width):
            x_new = x * x_scale
            y_new = y * y_scale

            x0 = int(x_new - a + 1)
            x1 = int(x_new + a)
            y0 = int(y_new - a + 1)
            y1 = int(y_new + a)

            new_pixel = 0

            for i in range(x0, x1 + 1):
                for j in range(y0, y1 + 1):
                    if (0 <= i < input_width and 0 <= j < input_height):
                        weight_x = sinc(x_new - i)
                        weight_y = sinc(y_new - j)
                        new_pixel += input_image[i, j] * weight_x * weight_y

            output_image[x,y] = int(round(new_pixel))

    return output_image


camera_man_64x64=downsize(camera_man,64)
camera_man_NN=NN(camera_man_64x64)
camera_man_linear=linear(camera_man_64x64,[512,512])
camera_man_hermite=hermite(camera_man_64x64,[512,512])
camera_man_bicubic=bicubic(camera_man_64x64,[512,512])
camera_man_lanzcos=lanczos(camera_man_64x64,[512,512])

print("linear"," " ,Psnr(camera_man_linear,camera_man))
print("hermite"," " ,Psnr(camera_man_hermite,camera_man))
print("bicubic"," " ,Psnr(camera_man_bicubic,camera_man))
print("lanzcos"," " ,Psnr(camera_man_lanzcos,camera_man))
print("NN"," " ,Psnr(camera_man_NN,camera_man))

cv2.imwrite('camera_man_64x64.png', camera_man_64x64)
cv2.imwrite('camera_man_NN.png', camera_man_NN)
cv2.imwrite('camera_man_linear.png', camera_man_linear)
cv2.imwrite('camera_man_hermite.png', camera_man_hermite)
cv2.imwrite('camera_man_bicubic.png',camera_man_bicubic)
cv2.imwrite('camera_man_lanczos.png',camera_man_lanzcos)
