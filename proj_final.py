# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 13:18:09 2021

@author: Elliot
"""

import matplotlib.pyplot as plt
import numpy as np

def main():
    # Get image
    print("Getting decrypted image...")
    image = get_image()
    row, col, depth = len(image), len(image[0]), len(image[0][0])
    print('     row='+str(row),'col='+str(col),'depth='+str(depth))
    #plt.imshow(image[:,:,1], cmap='gray', vmin = 0, vmax = 255, interpolation='none')
    plt.imshow(image)
    plt.title("Original Encrypted Image")
    plt.ylabel("Row")
    plt.xlabel("Col")
    print("     [Exit image to continue]")
    plt.show()

    # Decrypt image using XOR cipher
    print("Decrypting image...")
    # Get key
    key = get_key()
    keylen = len(key)
    print('     key='+key,'len='+str(keylen))
    # Generate key
    keyarr = keygen(row, col, depth, key, keylen)
    decrypt_image = np.bitwise_xor(image, keyarr)
    filename = 'decryptedImage.tiff'
    plt.imshow(decrypt_image)
    plt.imsave(filename, decrypt_image)
    plt.title("Decrypted Image")
    plt.ylabel("Row")
    plt.xlabel("Col")
    plt.imsave(filename, decrypt_image)
    print("     [Exit image to continue]")
    plt.show()

    # Convert decrypted to gray scale and plot
    print("Grayscaling Decrypted Image...")
    grayscale = rgb2gray(decrypt_image)
    filename = 'grayImage.tiff'
    plt.imshow(grayscale)
    plt.title("Decrypted Image (Grayscale)")
    plt.ylabel("Row")
    plt.xlabel("Col")
    plt.imsave(filename, grayscale)
    print("     [Exit image to continue]")
    plt.show()

    # Convert to float64
    floatscale = np.float64(grayscale)

    # Apply gaussian filter
    print("Blurring image...")
    blurred = blur(floatscale)
    filename = 'blurImage.tiff'
    plt.imshow(blurred)
    plt.title("Blurred Image")
    plt.ylabel("Row")
    plt.xlabel("Col")
    plt.imsave(filename, blurred)
    print("     [Exit image to continue]")
    plt.show()

    # Apply sobel (edge detector)
    print("Applying Sobel (edge detector)... please be patient")
    edges = sobel_filter(blurred)
    filename = 'edgeImage.tiff'
    plt.imshow(edges)
    plt.title("Sobel Image")
    plt.ylabel("Row")
    plt.xlabel("Col")
    plt.imsave(filename, edges)
    print("     [Exit image to continue]")
    plt.show()

    # Crop image around peak pixel
    print("Cropping image around peak pixel...")
    # Search for peak pixel
    hi_row, hi_col = find_peak(edges)
    print('     peak at (' + str(hi_row) + ',' + str(hi_col) + ')')
    #cropped = crop_image(decrypt_image, hi_row, hi_col, depth)
    cropped = crop_image(edges, hi_row, hi_col)
    filename = 'croppedImage.tiff'
    plt.imshow(cropped)
    plt.title("Cropped Image")
    plt.yticks([0, 20, 40, 60, 80, 100], [str(hi_row-50),str(hi_row-50+20),str(hi_row-50+40),str(hi_row-50+60),str(hi_row-50+80),str(hi_row-50+100)])
    plt.ylabel("Row")
    plt.xticks([0, 20, 40, 60, 80, 100], [str(hi_col-50),str(hi_col-50+20),str(hi_col-50+40),str(hi_col-50+60),str(hi_col-50+80),str(hi_col-50+100)])
    plt.xlabel("Col")
    plt.imsave(filename, cropped)
    print("     [Exit image to continue]")
    plt.show()

    # Encrypt image using new key
    #encrypt_image = np.bitwise_xor(image, keyarr)
    print("Encrypting image with new key...")
    # Generate LFSR-based key
    seed = "-1"
    while int(seed) < 1 or int(seed) > 65535:
        seed = input('     Enter seed (1-65535)? ')
    print("     Generating key...")
    keyarr = lfsr_keygen(row, col, depth, int(seed))
    encrypt_image = decrypt_image ^ keyarr
    filename = 'encryptedNew.tiff'
    plt.imshow(encrypt_image)
    plt.title("     Encrypt using New Key and Seed="+seed)
    plt.ylabel("Row")
    plt.xlabel("Col")
    plt.imsave(filename, encrypt_image)
    print("     [Exit image to continue]")
    plt.show()

    # Decrypt image using new key but different seed
    # If seed not same as before, will not succesfully decrypt.  Try it.
    print("Decrypting image with new key but different seed...")
    seed2 = "-1"
    while int(seed2) < 1 or int(seed2) > 65535:
        seed2 = input('     Enter seed (1-65535)? ')
    print("     Generating key...")
    keyarr2 = lfsr_keygen(row, col, depth, int(seed2))
    decrypt_image2 = encrypt_image ^ keyarr2
    grayscale2 = rgb2gray(decrypt_image2)
    filename = 'decryptedNewDiffSeed.tiff'
    plt.imshow(grayscale2)
    plt.title("Decrypted using New Key and Seed="+seed2+" (Grayscale)")
    plt.ylabel("Row")
    plt.xlabel("Col")
    plt.imsave(filename, grayscale2)
    print("     [Exit image to continue]")
    plt.show()

    # Decrypt image using new key and seed
    print("Decrypting image with new key and original seed...")
    seed3 = "-1"
    while int(seed3) < 1 or int(seed3) > 65535:
        seed3 = input('     Enter seed (1-65535)? ')
    print("     Generating key...")
    keyarr3 = lfsr_keygen(row, col, depth, int(seed3))
    decrypt_image3 = encrypt_image ^ keyarr3
    grayscale3 = rgb2gray(decrypt_image3)
    filename = 'decryptedNewSameSeed.tiff'
    # Convert to gray scale and plot
    plt.imshow(grayscale3)
    #plt.imshow(grayscale, cmap='gray', vmin = 0, vmax = 255, interpolation='none')
    plt.title("Decrypted using New Key and Seed="+seed3+" (Grayscale)")
    plt.ylabel("Row")
    plt.xlabel("Col")
    plt.imsave(filename, grayscale3)
    print("     [Exit image to continue]")
    plt.show()
	
    # Plot histograms using grayscale
    yaxis = 0.05
    print("Plotting histograms using grayscale")
    print("     Origianl encrypted image")
    print("     [Exit image to continue]")
    grayscale_encrypted = rgb2gray(image)
    plot_histogram(grayscale_encrypted, "Histogram Encrypted Image (grayscale)"  , yaxis, "fig1")
    print("     Decrypted image")
    print("     [Exit image to continue]")
    plot_histogram(grayscale, "Histogram Decrypted Image (grayscale)" , yaxis, "fig2")
    print("     Encrypted image using new key")
    print("     [Exit image to continue]")
    grayscale_encrypted_lfsr = rgb2gray(encrypt_image)
    plot_histogram(grayscale_encrypted_lfsr, "Histogram Encrypted Image with New Key (grayscaled)"  , yaxis, "fig3")
# End Main


# Get Image Fileanme and Load into Array
def get_image():
    default_name = 'Pale_Blue_Dot_Encrypted.tiff'
    image_name = input('     What is the filename of the image? [' + default_name + ']')
    if len(image_name) == 0:
        image_name = default_name
    image = plt.imread(image_name)[:,:,:3]
    # Check Datatype and Convert if Needed
    if image.dtype == np.uint8:
        print("     Image is", image.dtype, "format")
    else:
        print("     Image is", image.dtype, " format. Converting to uint8...")
        image = image.astype(np.uint8)
    return image

# RGB to Grayscale
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    grayscale = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return grayscale

# Get Key
def get_key():
    default_key = "COME AND GET YOUR LOVE"
    key = str(input('     Enter key for decryption: [' + default_key + ']'))
    if len(key) == 0:
        key = default_key
    key=key.replace(" ","")  # Remove whitespace (strip only removes leading and trailing whitespaces including tabs (\t))
    return key

# Generate Symmetric Key
def keygen(row, col, depth, key, keylen):
    A = np.zeros((row, col, depth), dtype=np.uint8)
    for i in range(row):
        for j in range(col):
            for k in range(depth):
                a = (i*j) % keylen
                A[i][j][k] = a
    A = A*(256//keylen)
    return A

def lfsr_keygen(row, col, depth, seed):
    A = np.zeros((row, col, depth), dtype=np.uint8)
    state = seed
    for i in range(row):
        for j in range(col):
            for k in range(depth):
                new_8bit = 0
                for b in range(8):
                    newbit = (state ^ (state >> 4) ^ (state >> 11) ^ (state >> 14) ^ (state >> 15)) & 1
                    state = (state >> 1) | (newbit << 15)
                    new_8bit = new_8bit | (newbit << b)
                A[i][j][k] = new_8bit
    return A

def blur(a):
    kernel = np.array([[1.0,2.0,1.0], [2.0,4.0,2.0], [1.0,2.0,1.0]])
    kernel = kernel / np.sum(kernel)
    arraylist = []
    for y in range(3):
        temparray = np.copy(a)
        temparray = np.roll(temparray, y - 1, axis=0)
        for x in range(3):
            temparray_X = np.copy(temparray)
            temparray_X = np.roll(temparray_X, x - 1, axis=1)*kernel[y,x]
            arraylist.append(temparray_X)

    arraylist = np.array(arraylist)
    arraylist_sum = np.sum(arraylist, axis=0)
    return arraylist_sum

def sobel_filter(image):
    filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    new_image_x = convolution2d(image, filter, 1, 0)
    new_image_y = convolution2d(image, np.flip(filter.T, axis=0), 1, 0)
    gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))
    gradient_magnitude *= 255.0 / gradient_magnitude.max()
    return gradient_magnitude

def convolution2d(image, kernel, stride, padding):
    image = np.pad(image, [(padding, padding), (padding, padding)], mode='constant', constant_values=0)

    kernel_height, kernel_width = kernel.shape
    padded_height, padded_width = image.shape

    output_height = (padded_height - kernel_height) // stride + 1
    output_width = (padded_width - kernel_width) // stride + 1

    new_image = np.zeros((output_height, output_width)).astype(np.float32)

    for y in range(0, output_height):
        for x in range(0, output_width):
            new_image[y][x] = np.sum(image[y * stride:y * stride + kernel_height, x * stride:x * stride + kernel_width] * kernel).astype(np.float32)
    return new_image

def find_peak(image):
    row, col = len(image), len(image[0])
    peak_row = 0
    peak_col = 0
    peak_value = 0
    for i in range(row):
        for j in range(col):
            if image[i][j] > peak_value:
                peak_value = image[i][j]
                peak_row = i
                peak_col = j
    return peak_row, peak_col

def crop_image(image, hi_row, hi_col):
    A = np.zeros((101, 101), dtype=np.uint8)
    row, col = len(image), len(image[0])
    crop_row = hi_row - 50
    crop_col = hi_col - 50
#    if crop_row < 0:
#        crop_row = 0
#    if crop_col < 0:
#        crop_col = 0
#    if (crop_row + 100) >= row:
#        crop_row = row - 100
#    if (crop_col + 100) >= col:
#        crop_col = col - 100
    print('     crop row='+str(crop_row),'col='+str(crop_col))
    for i in range(101):
        tmp_col = crop_col
        for j in range(101):
            A[i][j] = image[crop_row][tmp_col]
            tmp_col = tmp_col + 1
        crop_row = crop_row + 1
    return A

def plot_histogram(x, title, yaxis, filename):
    plt.hist(x, density=True, bins=10, label="Data")
    #mn, mx = plt.xlim()
    #plt.xlim(mn, mx)
    #kde_xs = np.linspace(mn, mx, 300)
    #kde = st.gaussian_kde(x)
    #plt.plot(kde_xs, kde.pdf(kde_xs), label="PDF")
    #plt.legend(loc="upper left")
    #plt.ylabel("Probability")
    plt.xlabel("Data")
    plt.title(title)
    if ( yaxis > 0 ):
        plt.ylim((0,yaxis))
    plt.grid(True)
    plt.savefig(filename)
    plt.show()
    return 1



if __name__ == '__main__':
    main()
