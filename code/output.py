import os
import cv2
import numpy as np
from numpy import ndarray
from keras.models import load_model
import functools
import time


# resizing live feed to desired dimensions
def feed_preprocessing(frame):

    frame = np.array(frame, dtype=np.float32)
    frame = np.reshape(frame, (1, 254, 254, 1))
    return frame


def write_letter(label_letter):

    # dirs is a list containing all letters in order (from default from the directory)
    dirs = [label for label in os.listdir('NewDataSet/Train') if os.path.isdir(os.path.join('NewDataSet/Train', label))]

    # print(type(dirs[label]))
    return dirs[label_letter]


def main_func():

    image = cv2.imread('asl_alphabet.jpg')

    resized_image = cv2.resize(image, (570, 720))

    # loading the pre-trained model
    model = load_model('SeqModel.h5')

    camera = cv2.VideoCapture(0)  # 0 -> index of camera

    camera.set(3, 1080)
    camera.set(4, 720)

    predict_letter_list = []
    word_str = ''
    word_list = []

    while True:

        check, frame1 = camera.read()

        cv2.rectangle(frame1, (20, 70), (275, 325), (0, 255, 0), thickness=1)

        gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        median_blurred = cv2.medianBlur(gray_frame, 3)  # blurring (median) the gray frame

        gaus_blurred = cv2.GaussianBlur(median_blurred, (3, 3), 0)  # blurring (gaussian) the already blurred gray frame

        canny_frame = cv2.Canny(gaus_blurred, 30, 30)

        cropped_frame = canny_frame[71:325, 21:275]

        cropFrame_array = feed_preprocessing(cropped_frame)

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

        # matching the labels to the actual values
        # most_prob > number (where number determined based on measures) to avoid random results
        if most_prob > 0.65:

            # print(f'accuracy: {most_prob}')
            # print(f'letter {write_letter(integer_label)}')

            letter = write_letter(integer_label)
            # append the predicted letter into the list
            predict_letter_list.append(letter)
            print(predict_letter_list)
            # every x letters
            if len(predict_letter_list) % 15 == 0:
                # check that ALL (x) the letters in the list are the same
                if all(x == predict_letter_list[0] for x in predict_letter_list):
                    if integer_label == 20:
                        word_list += ' '
                    elif integer_label == 4:  # trycatch should be added for empty list(deleting letter from empty word)
                        try:
                            word_list.pop()
                        except:
                            pass

                    else:
                        # add to the word that will be printed one of the identical items of the list
                        word_list.append(predict_letter_list[0])
                    time.sleep(.5)

                    print(word_list)

                # every x letter clear the list so the process repeats again from the start
                predict_letter_list.clear()
                # convert list into string for putText function
                word_str = "".join(word_list)

        frame1 = cv2.putText(frame1, 'Predicted Word: ' + word_str, (20, 60),
                             cv2.FONT_HERSHEY_SIMPLEX, 2, (128, 0, 0), 3)
        # frame1 = cv2.putText(frame1, 'Predicted Letter: ' + write_letter(integer_label), (20, 500),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 2, (128, 0, 0), 3)

        if not check:
            break

        cv2.imshow('WINDOW', frame1)
        cv2.imshow('CROPPED', cropped_frame)
        cv2.imshow('ALPHABET', resized_image)

        if frame1 is None:
            break

        keyboard = cv2.waitKey(1) & 0xff
        if keyboard == 27:  # esc to terminate
            break

    camera.release()
    cv2.destroyAllWindows()


main_func()
