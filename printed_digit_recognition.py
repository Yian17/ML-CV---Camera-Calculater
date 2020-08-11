import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


def get_data(path):
    np.random.seed(7)
    lst = []
    for image in os.listdir(path):
        if image[-3:] == "jpg":
            img = cv2.imread(path + "/" + image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (28, 28))
            lst.append(img)
    return np.array(lst)


def split_data():
    # path = "English/Img/GoodImg/Bmp/"
    path = "dataset/digits/"
    zero = path + "0"
    one = path + "1"
    two = path + "2"
    three = path + "3"
    four = path + "4"
    five = path + "5"
    six = path + "6"
    seven = path + "7"
    eight = path + "8"
    nine = path + "9"

    lst_0 = get_data(zero)
    lst_1 = get_data(one)
    lst_2 = get_data(two)
    lst_3 = get_data(three)
    lst_4 = get_data(four)
    lst_5 = get_data(five)
    lst_6 = get_data(six)
    lst_7 = get_data(seven)
    lst_8 = get_data(eight)
    lst_9 = get_data(nine)


    train_img = []
    train_label = []
    test_img = []
    test_label = []
    for i in range(10):
        train_img.extend(eval("lst_" + str(i))[:18])
        for j in range(18):
            train_label.append(i)
        test_img.extend(eval("lst_" + str(i))[18:])
        for k in range(6):
            test_label.append(i)

    train_img = np.array(train_img)
    train_label = np.array(train_label)
    test_img = np.array(test_img)
    test_label = np.array(test_label)

    return train_img, train_label, test_img, test_label


def split_data_symbol():
    path = "dataset/symbol/"
    plus = path + "plus"
    minus = path + "minus"
    multiply = path + "multiply"
    divide = path + "divide"
    left = path + "left"
    right = path + "right"
    equal = path + "equal"

    lst_0 = get_data(plus)
    lst_1 = get_data(minus)
    lst_2 = get_data(multiply)
    lst_3 = get_data(divide)
    lst_4 = get_data(left)
    lst_5 = get_data(right)
    lst_6 = get_data(equal)
    train_img = []
    train_label = []
    test_img = []
    test_label = []
    for i in range(7):
        train_img.extend(eval("lst_" + str(i))[:53])
        for j in range(53):
            train_label.append(i + 10)
        test_img.extend(eval("lst_" + str(i))[53:])
        for k in range(10):
            test_label.append(i + 10)

    train_img = np.array(train_img)
    train_label = np.array(train_label)
    test_img = np.array(test_img)
    test_label = np.array(test_label)

    return train_img, train_label, test_img, test_label

# This model is only used for analyzing the performance difference between
# digits and symbols. Not used in the detection and calculation.
def classify_digits():

    # np.random.seed(2)

    train_img, train_label, test_img, test_label = split_data()

    train_img = train_img / 255.0
    test_img = test_img / 255.0

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_img, train_label, batch_size=10, epochs=100)
    test_loss, test_acc = model.evaluate(test_img, test_label)

    print('Test accuracy:', test_acc)

# This model is only used for analyzing the performance difference between
# digits and symbols. Not used in the detection and calculation.
def classify_symbols():

    np.random.seed(7)

    train_img, train_label, test_img, test_label = split_data_symbol()

    # exit(0)
    train_img = train_img / 255.0
    test_img = test_img / 255.0

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_img, train_label, batch_size=45, epochs=200)
    test_loss, test_acc = model.evaluate(test_img, test_label)

    print('Test accuracy:', test_acc)


def classifier():

    train_img, train_label, test_img, test_label = split_data()

    train_symbol_img, train_symbol_label, test_symbol_img, test_symbol_label = split_data_symbol()

    train_img = np.vstack((train_img, train_symbol_img))
    train_label = np.hstack((train_label, train_symbol_label))
    test_img = np.vstack((test_img, test_symbol_img))
    test_label = np.hstack((test_label, test_symbol_label))

    train_img = train_img / 255.0
    test_img = test_img / 255.0

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(17, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_img, train_label, batch_size=40, epochs=150)
    test_loss, test_acc = model.evaluate(test_img, test_label)

    print('Test accuracy:', test_acc)

    return model



# classifier()


def detect_element_from_pic(img):

    copyed = img
    height, width = img.shape
    wid, hei = int(width / 4), int(height / 4)
    img = cv2.resize(img,(int(width / 4), int(height / 4)))

    image, contours, hier = cv2.findContours(img, cv2.RETR_TREE,
                                             cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 0, 255), 6)
    image, contours, hier = cv2.findContours(image, cv2.RETR_TREE,
                                             cv2.CHAIN_APPROX_SIMPLE)

    cropped_element = []
    x_position = []
    y_position = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        if x > 16 and y > 20 and w < wid-16 and h < hei-20:
            if w > 20 or h > 26:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                print(x, y, x+w, y+h)
                cropped = copyed[y * 4:(y+h) * 4, x * 4:(x+w) * 4]
                # cropped = copyed[y:(y+h), x:(x+w)]
                # cropped = copyed[x:x+w, y:y+h]
                cropped_element.append(cropped)
                x_position.append(x + w)
                y_position.append(y + h)

    result = [cropped_element for _, cropped_element in sorted(zip(x_position, cropped_element))]

    #
    cv2.imwrite("result2.jpg", img)

    x_position.sort()
    y_position.sort()

    return result, x_position[-1], y_position[-1]


def do_calculation(expression):

    detected_expr = ""
    for element in expression:
        detected = element
        if element == 10:
            detected = "+"
        elif element == 11:
            detected = "-"
        elif element == 12:
            detected = "*"
        elif element == 13:
            detected = "/"
        elif element == 14:
            detected = "("
        elif element == 15:
            detected = ")"
        elif element == 16:
            detected = "="
        detected_expr += str(detected)

    print(detected_expr, eval(detected_expr[:-1]))
    return eval(detected_expr[:-1])

# do_calculation()

def preprocessing_img(image):
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    ret, after = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)


    return after

# preprocessing_img("dataset/calculation/printed/expression1.jpg")


def do_homography(img, points):

    before_pts = points

    after_pts = [[0, 0], [350, 0], [350, 270], [0, 270]]
    h, status = cv2.findHomography(np.array(before_pts), np.array(after_pts))

    out = cv2.warpPerspective(img, h, (350, 270))
    out = cv2.resize(out, (out.shape[1] * 4, out.shape[0] * 4))

    return out


def recognize_expression(path, oblique, oblique_points):

    model = classifier()

    test_img = []
    for image in os.listdir(path):
        if image[-3:] == "jpg":
            test_img.append(image)

    for i in range(len(test_img)):
        image = test_img[i]
        oblique_img = None

        # get detected element, and prepocessing them
        if oblique:
            pts = oblique_points[i]
            img = cv2.imread(path + image, cv2.IMREAD_GRAYSCALE)
            oblique_img = do_homography(img, pts)
            ret, after = cv2.threshold(oblique_img, 130, 255, cv2.THRESH_BINARY)
            img = after

        else:
            img = preprocessing_img(path + image)

        elements, x, y = detect_element_from_pic(img)
        lst = []
        for e in elements:
            im = cv2.resize(e, (28, 28))
            lst.append(im)
        lst = np.array(lst)

        # get predictions
        predictions = model.predict(lst)
        predict_result = []

        # get label of predictions
        for predict in predictions:
            result = np.argmax(predict)
            predict_result.append(result)
        print(predict_result)

        # calculate, and paste to original image
        calculated_result = do_calculation(predict_result)

        if oblique:
            origin = oblique_img
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(origin, str(calculated_result), ((x+4) * 4, (y-17) * 4), font, 5,(0,0,0),18,cv2.LINE_AA)

            after = cv2.resize(origin, (int(origin.shape[1]/4), int(origin.shape[0]/4)))

        else:
            origin = cv2.imread(path + image)
        # print(orgin)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(origin, str(calculated_result), ((x+4) * 4, y * 4), font, 7,(0,0,0),18,cv2.LINE_AA)

            after = cv2.resize(origin, (int(origin.shape[1]/4), int(origin.shape[0]/4)))

        cv2.imwrite("result/calculated_"+image, after)


oblique = True

if oblique:
    path = "dataset/calculation/oblique/"
else:
    path = "dataset/calculation/printed/"

oblique_points = [[[56, 83], [924, 94], [1222, 404], [106, 518]]]
recognize_expression(path, oblique, oblique_points)


