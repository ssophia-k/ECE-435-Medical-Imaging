import cv2
import numpy as np

def add_salt_and_pepper_noise(img, salt_prob=0.01, pepper_prob=0.01):
    noisy_img = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rand_val = np.random.rand()
            if rand_val < salt_prob:
                noisy_img[i, j] = 255  # Salt
            elif rand_val < salt_prob + pepper_prob:
                noisy_img[i, j] = 0    # Pepper
    return noisy_img

def add_gaussian_noise(image, mean=0, sigma=25):
    # Generate random Gaussian noise
    # The noise must match the image shape (height, width, channels)
    row, col = image.shape
    gauss = np.random.normal(mean, sigma, (row, col))
    
    # Add the noise to the image
    noisy_image = image + gauss
    
    # Clip the values to stay between 0 and 255, then convert back to uint8
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image

