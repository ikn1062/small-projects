import pygame
import sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import os.path
import cv2
from model import make_model


class Blackboard:
    def __init__(self, wsize_x, wsize_y, boundary=5):
        self.wsize_x = wsize_x
        self.wsize_y = wsize_y

        self.boundary = boundary

        self.model = load_model("mnist_model.model")

        self.labels = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9"}

        pygame.init()
        self.display = pygame.display.set_mode((wsize_x, wsize_y))
        pygame.display.set_caption("Blackboard")
        self.font = pygame.font.Font("arial.ttf", 25)

        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.red = (255, 0, 0)

    def board(self):
        x_arr, y_arr = [], []
        write = False

        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == MOUSEMOTION and write:
                    x, y = event.pos
                    pygame.draw.circle(self.display, self.white, (x, y), 4, 0)
                    x_arr.append(x)
                    y_arr.append(y)

                if event.type == MOUSEBUTTONDOWN:
                    write = True

                if event.type == MOUSEBUTTONUP:
                    write = False
                    self.predict(x_arr, y_arr)
                    x_arr, y_arr = [], []

                if event == KEYDOWN and event.unicode == "n":
                    self.display.fill(self.black)

                pygame.display.update()

    def predict(self, x_arr, y_arr):
        x_arr = sorted(x_arr)
        y_arr = sorted(y_arr)

        min_x, max_x = max(x_arr[0] - self.boundary, 0), min(x_arr[-1] + self.boundary, self.wsize_x)
        min_y, max_y = max(y_arr[0] - self.boundary, 0), min(y_arr[-1] + self.boundary, self.wsize_y)

        img_arr = np.array(pygame.PixelArray(self.display))[min_x:max_x, min_y:max_y].T.astype(float)

        image = cv2.resize(img_arr, (28, 28))
        image = np.pad(image, (10, 10), 'constant', constant_values=0)
        image = cv2.resize(image, (28, 28)) / 255

        label = str(self.labels[np.argmax(self.model.predict(image.reshape(1, 28, 28, 1)))])

        textSurface = self.font.render(label, True, self.white)
        textRecObj = textSurface.get_rect()
        textRecObj.left, textRecObj.bottom = min_x, min_y

        self.display.blit(textSurface, textRecObj)


if __name__ == "__main__":
    if not os.path.exists('mnist_model.model'):
        make_model()
    blackboard = Blackboard(1080, 720)
    blackboard.board()

