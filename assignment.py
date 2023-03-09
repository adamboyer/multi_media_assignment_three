from scipy.fft import dct
import cv2
import numpy as np
import sys

def pretty_print(block):
    print("")
    for i in range(len(block)):
            print_line = ""
            for j in range(len(block[i])):
                add_me = " " + str(block[i][j])
                while len(add_me) < 5:
                    add_me += " " 
                print_line += add_me
            print(print_line)
    print("")

def compute_sad_and_dct(target_block, ref_gray_scale):

    #Here is the block size 16x16
    block_size = 16

    #largest possible float value set to minimum SAD and motion vector set to zero
    min_SAD = sys.float_info.max
    motion_vector = (0,0)
    best_match_block = None
    best_match_x1 = None
    best_match_y1 = None
    best_match_x2 = None 
    best_match_y2 = None
    #Find the best match in the reference image
    for i in range(block_size, ref_gray_scale.shape[0] - block_size, block_size):
        for j in range(block_size, ref_gray_scale.shape[1] - block_size, block_size):
            
            #Get current ref block
            x1 = j - block_size // 2
            y1 = i - block_size // 2
            x2 = j + block_size // 2
            y2 = i + block_size // 2
            ref_block = ref_gray_scale[y1:y2, x1:x2]

            #Calculate the Sum of Absolute Difference between the target and ref blocks
            current_SAD = np.sum(np.abs(ref_block - target_block))

            #Update best match
            if current_SAD < min_SAD:
                best_match_block = ref_block
                best_match_x1 = x1 
                best_match_y1 = y1 
                best_match_x2 = x2 
                best_match_y2 = y2 
                min_SAD = current_SAD
                motion_vector = (j - block_size, i - block_size)

    # Calculate the new position of the best matching block in the reference frame
    x, y = motion_vector
    new_x1 = best_match_x1 + x
    new_y1 = best_match_y1 + y
    new_x2 = best_match_x2 + x
    new_y2 = best_match_y2 + y
    #print(x,y,new_x1,new_y1,new_x2,new_y2,x1,y1,x2,y2)

    #while the block is out of bounds for ref image add padding of zeros
    while new_y2 > ref_gray_scale.shape[0] or new_x2 > ref_gray_scale.shape[1]:
        # Add zero padding to the image
        pad_size = 50
        ref_gray_scale = cv2.copyMakeBorder(ref_gray_scale, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=0)
    # # Extract the new block from the reference frame
    new_ref_block = ref_gray_scale[new_y1:new_y2, new_x1:new_x2]

    #Now find the difference block of the best match block and the target block
    
    diff_block = target_block - new_ref_block 

    #Compute the Discrete Cosine Transform of the diff_block
    dct_block = dct(dct(diff_block, axis=0, norm='ortho'), axis=1, norm='ortho')

    #Round values to the nearest integer
    dct_int = np.round(dct_block)
    dct_int = dct_int.astype(int)
    print("")
    print("Here DCT of the difference between the target block and the best match block after movement compensation")

    pretty_print(dct_int)
    
    print("motion vector: " + str(motion_vector))
    print("")
    
    print("")
    print("Here is just the DCT of the target block not the difference block")
    dct_target_block = dct(dct(target_block, axis=0, norm='ortho'), axis=1, norm='ortho')
    dct_int_target = np.round(dct_target_block)
    dct_int_target = dct_int_target.astype(int)
    pretty_print(dct_int_target)

#Read in the two frames
ref_img = cv2.imread("Frame1.png")
target_img = cv2.imread("Frame2.png")

#Convert the images to grayscale to reduce the computation cost
#We are really only concerned with the change in intensity levels
ref_gray_scale = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY) 
target_gray_scale = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY) 

#Here is the block size 16x16
block_size = 16

#Start at the top left corner in our target image
print("")
print("Here is the top left block")
print("")
top_left_block = target_gray_scale[:block_size, :block_size]
compute_sad_and_dct(top_left_block, ref_gray_scale)

#Next do the center of the image
print("")
print("Here is the center block")
print("")

width, height = target_gray_scale.shape
center_x = width // 2
center_y = height // 2
x1 = center_x - block_size // 2
y1 = center_y - block_size // 2
x2 = center_x + block_size // 2
y2 = center_y + block_size // 2
center_block = target_gray_scale[y1:y2, x1:x2]
compute_sad_and_dct(center_block, ref_gray_scale)

#Lastly do the bottom right block of the image
print("")
print("Here is the bottom right block")
print("")
height, width = target_gray_scale.shape
x1 = width - block_size
y1 = height - block_size
x2 = width
y2 = height
bottom_right_block = target_gray_scale[y1:y2, x1:x2]
compute_sad_and_dct(bottom_right_block, ref_gray_scale)
