NUM_CLASSES = 36 # 10 digits + 26 lowercase alphabets
CNN_CLASSES = 180 # (36 * 5) where 5 is total string length
ALL_CHAR_SET_LEN = 36
import numpy as np

ALL_CHAR_SET = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v', 'w', 'x', 'y', 'z']

def encoder(text, is_cnn=True):
    vector = np.zeros(CNN_CLASSES, dtype='float32') if is_cnn \
        else np.zeros(NUM_CLASSES, dtype='float32')

    def char2pos(char):
        k = ord(char) - 48
        if k > 9:
            k = ord(char) - 97 + 10
            if k > 35:
                raise ValueError('error')
        return (k - 1 if k < 24 else k - 2) 

    for i, char in enumerate(text):
        idx = i * ALL_CHAR_SET_LEN + char2pos(char)
        #print('idx=',idx)
        vector[idx] = 1.0
    return vector


def decoder(vec):
    label = ''
    len  = vec.shape[0]
    print('len=',len)
    #print(vec)
    print(vec.shape)
    for i in range(vec.shape[0] // ALL_CHAR_SET_LEN):
        start =ALL_CHAR_SET_LEN * i
        end = ALL_CHAR_SET_LEN * (i + 1)
        print('start=',start)
        print('end=',end)
        ats = np.argmax(vec[start:end])
        print(ats)
        label += ALL_CHAR_SET[np.argmax(vec[start:end])]

    return label

import numpy as np

def padded_img(img):
    desired_width = 224
    desired_height = 224

    # Assuming you have already loaded your 200x50x3 image as 'img' with numpy array
    original_height, original_width, channels = img.shape

    # Calculate the padding required on each side
    pad_width = desired_width - original_width
    pad_height = desired_height - original_height

    # Calculate the left, top, right, and bottom padding
    pad_left = pad_width // 2
    pad_top = pad_height // 2
    pad_right = pad_width - pad_left
    pad_bottom = pad_height - pad_top

    # Pad the image with black (you can change 'fill_value' to any other color)
    padded_img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)

    return padded_img

