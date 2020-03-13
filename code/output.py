import os
import cv2
import numpy as np
from numpy import ndarray
from keras.models import load_model
import functools
from numpy import indices
from cv2 import threshold, THRESH_BINARY, THRESH_OTSU


# resizing live feed to desired dimensions
def feed_preprocessing(frame):

    frame = np.array(frame, dtype=np.float32)
    frame = np.reshape(frame, (1, 224, 224, 1))
    return frame


def write_letter(frame, label_letter):

    # dirs is a list containing all letters in order (from default from the directory)
    dirs = [label for label in os.listdir('NewDataSet/Train') if os.path.isdir(os.path.join('NewDataSet/Train', label))]
    frame = cv2.putText(frame, dirs[label_letter], (270, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (30, 30, 255), 3)

    # print(type(dirs[label]))
    return frame


def recognition_func():

    # loading the pre-trained model
    model = load_model('SeqModel.h5')

    camera = cv2.VideoCapture(0)  # 0 -> index of camera

    camera.set(3, 1080)
    camera.set(4, 720)

    while True:

        check, frame1 = camera.read()

        cv2.rectangle(frame1, (20, 70), (245, 295), (0, 255, 0), thickness=1)

        gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        median_blurred = cv2.medianBlur(gray_frame, 9)

        gaus_blurred = cv2.GaussianBlur(median_blurred, (5, 5), 0)

        ret, thre_ots = cv2.threshold(gaus_blurred, 254, 510, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        cropped_frame = thre_ots[71:295, 21:245]

        cropFrame_array = feed_preprocessing(cropped_frame)

        # adding dimension
        # cropFrame_array = np.expand_dims(cropFrame_array, axis=0)

        # predict
        predictions_prob = model.predict(cropFrame_array)

        # print(predictions_prob)

        # take the value with the highest probability
        most_prob = ndarray.max(predictions_prob)

        # print(most_prob)

        # [0] -> A , [1] -> B , [2]-> C
        # print(type(predictions_label)) --> numpy.ndarray
        predictions_label = model.predict_classes(cropFrame_array)

        # convert [0] to 0 for future operations
        string_label = functools.reduce(lambda x, y: x + str(y), predictions_label, '')
        integer_label = int(string_label)

        # 0 -> A, 1 -> B, 2-> C
        # print(integer_label)

        frame1 = cv2.putText(frame1, 'Letter:', (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (30, 30, 255), 3)

        # matching the labels to the actual values
        # most_prob > number (where number determined based on measures) to avoid random results
        
        if most_prob > 0.85:

            # print(dirs[integer_label])

            write_letter(frame1, integer_label)


        if not check:
            break

        cv2.imshow('WINDOW', frame1)
        cv2.imshow('CROPPED', cropped_frame)

        if frame1 is None:
            break

        keyboard = cv2.waitKey(1) & 0xff
        if keyboard == 27:  # esc to terminate
            break

    camera.release()
    cv2.destroyAllWindows()


recognition_func()

