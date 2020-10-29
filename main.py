import cv2 as cv
import numpy as np
from PIL import Image
from drowsiness_model import DrowsinessModel

from variables import*

if __name__ == "__main__":
    model = DrowsinessModel()
    model.run()
    video = cv.VideoCapture(0)

    while True:
            _, frame = video.read()

            im = Image.fromarray(frame, 'RGB')
            im = im.resize(target_size)
            img_array = np.array(im)
            img_array = np.expand_dims(img_array, axis=0)
            prediction = model.predition(img_array)
            # print(prediction)
            # #if prediction is 0, which means I am missing on the image, then show the frame in gray color.
            # if prediction == 0:
            #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv.imshow("Capturing", frame)
            key=cv.waitKey(1)
            if key == ord('q'):
                    break
    video.release()
    cv.destroyAllWindows()