import os
import cv2 as cv
import numpy as np
from PIL import Image
from variables import*
video = cv.VideoCapture(0)
while True:
        _, frame = video.read()

        im = Image.fromarray(frame, 'RGB')
        im = im.resize(target_size)
        img_array = np.array(im)
        
        img_list = os.listdir(generate_img_dir)
        if len(img_list) > 0:
            img_idx = [int(img_.split('.')[0]) for img_ in img_list]
            save_idx = max(img_idx) + 1
        else:
            save_idx = 1
        save_img = str(save_idx)+'.png'
        save_img = os.path.join(generate_img_dir, save_img)
        cv.imwrite(save_img, img_array) 

        cv.imshow("Capturing", frame)
        key=cv.waitKey(1)
        if key == ord('q'):
                break
video.release()
cv.destroyAllWindows()