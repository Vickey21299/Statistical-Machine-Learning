
import cv2
import numpy as np
import random 
import math
def median_filter(input_image, filter_size):
    print(filter_size)
    (image_height, image_width) = input_image.shape
    output_image = np.zeros_like(input_image)

    
    for i in range(image_height):
        for j in range(image_width):
            neighborhood =np.array([])
            median_value=0
            for k in range(i-(filter_size//2),i+(filter_size//2)+1):
                for l in range(j-(filter_size//2),j+(filter_size//2)+1):
                    if(k>=0 and l>=0 and k<image_height and l<image_width):
                        neighborhood=np.append(neighborhood,input_image[k,l])
            median_value=np.median(neighborhood)
            output_image[i, j] = median_value

    return output_image
def Psnr(distorted_img,original_img):
    mse = np.mean((original_img.astype(np.float64) - distorted_img.astype(np.float64)) ** 2)
    psnr = 10*math.log10((255 ** 2) / mse)

    return psnr

def Best_denoised_image(distorted_img,original_image):
    best_denoised_image=np.copy(distorted_img)
    psnr=0
    best_size=0
    i=0
    row,col=distorted_img.shape
    for i in range(1,8):
        curr=median_filter(distorted_img,i)
        curr_psnr=Psnr(curr,original_image)
        if(curr_psnr>psnr):
            psnr=curr_psnr
            best_denoised_image=curr
            best_size=i
    return (best_denoised_image,psnr,best_size)



def add_salt_pepper_noise(img,distortion_perc):
    row,col=img.shape
    noisy_pixels=(distortion_perc/100)*row*col
    #print(row," ",col," ",noisy_pixels)
    
    output=np.copy(img)
    for i in range(0,int(noisy_pixels)):
        
        random_row_salt,random_col_salt=random.randint(0,row-1),random.randint(0,col-1)
        output[random_row_salt][random_col_salt]=255

        random_row_pepper,random_col_pepper=random.randint(0,row-1),random.randint(0,col-1)
        output[random_row_pepper][random_col_pepper]=0

    return output

barbara = cv2.imread('barbara_gray.bmp', cv2.IMREAD_GRAYSCALE)
#print(barbara)
n,m=barbara.shape
totl_pixels=n*m


noisy_5_perc=add_salt_pepper_noise(barbara,5)
noisy_15_perc=add_salt_pepper_noise(barbara,15)

noisy_20_perc=add_salt_pepper_noise(barbara,20)
noisy_25_perc=add_salt_pepper_noise(barbara,25)

(denoised_5,psnr_5,kernel_size_5)=Best_denoised_image(noisy_5_perc,barbara)
(denoised_15,psnr_15,kernel_size_15)=Best_denoised_image(noisy_15_perc,barbara)
(denoised_20,psnr_20,kernel_size_20)=Best_denoised_image(noisy_20_perc,barbara)
(denoised_25,psnr_25,kernel_size_25)=Best_denoised_image(noisy_25_perc,barbara)

print(psnr_5," ",kernel_size_5)
print(psnr_15," ",kernel_size_15)
print(psnr_20," ",kernel_size_20)
print(psnr_25," ",kernel_size_25)









cv2.imwrite('noisy_5_perc.png', noisy_5_perc)
cv2.imwrite('noisy_15_perc.png', noisy_15_perc)
cv2.imwrite('noisy_20_perc.png', noisy_20_perc)
cv2.imwrite('noisy_25_perc.png', noisy_25_perc)


cv2.imwrite('denoised_5.png', denoised_5)
cv2.imwrite('denoised_15.png', denoised_15)
cv2.imwrite('denoised_20.png', denoised_20)
cv2.imwrite('denoised_25.png', denoised_25)





        