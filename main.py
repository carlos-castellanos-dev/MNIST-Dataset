from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np


# Function Definition
# Part B (Function to Plot)
def plot_digits(x, y):
    for i in range(10):
        x_set_d = x[y == i, :, :]
        x_set_i = x_set_d[10, :, :]  # Selecting Digit From Set
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_set_i, cmap='gray')
        plt.title('Label: ' + str(i))
    plt.show()


# Main
# Part A (Display # of Images/Image Size in Sets)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("The number of images in training set is", x_train.shape[0])
print("The number of images in testing set is", x_test.shape[0])
print("Image size is", x_train.shape[1], "x", x_train.shape[2])

# Part D (Shuffling Training Set)
num_train_img = x_train.shape[0]
train_ind = np.arange(0, num_train_img)

train_ind_s = np.random.permutation(train_ind)

x_train = x_train[train_ind_s, :, :]
y_train = y_train[train_ind_s]

# Part D (Selecting 20% of Training Data)
x_valid = x_train[0:int(0.2 * num_train_img), :, :]
y_valid = y_train[0:int(0.2 * num_train_img)]

# Part D (Rest of Training Set)
x_train = x_train[int(0.2 * num_train_img):, :, :]
y_train = y_train[int(0.2 * num_train_img):]

# Part C (Call function to display images from each set)
print("Selecting 10 random images from training set")
plot_digits(x_train, y_train)
print("Selecting 10 random images from validation set")
plot_digits(x_valid, y_valid)
print("Selecting 10 random images from testing set")
plot_digits(x_test, y_test)

# Part D (Display # of Images/Image Size in Sets)
print("The number of images in training set is", x_train.shape[0])
print("The number of images in validation set is", x_valid.shape[0])
print("The number of images in testing set is", x_test.shape[0])
