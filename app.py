import pygame
import sys
import cv2
import numpy as np
from keras.models import load_model
from pygame.locals import *

# Constants
WINDOWSIZEX = 640
WINDOWSIZEY = 480
BOUNDRYINC = 5
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
MODEL = load_model("my_trained_model.keras")
LABELS = {
    0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four",
    5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"
}

pygame.init()
FONT = pygame.font.Font(None, 24)  # Default system font with size 24
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
pygame.display.set_caption("Digit Board")

iswriting = False
number_xcord = []
number_ycord = []
image_cnt = 1

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)
            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False

            if number_xcord and number_ycord:  # Ensure lists are not empty
                rect_min_x = max(min(number_xcord) - BOUNDRYINC, 0)
                rect_max_x = min(max(number_xcord) + BOUNDRYINC, WINDOWSIZEX)
                rect_min_y = max(min(number_ycord) - BOUNDRYINC, 0)
                rect_max_y = min(max(number_ycord) + BOUNDRYINC, WINDOWSIZEY)

                img_arr = np.array(pygame.surfarray.array3d(DISPLAYSURF))
                img_arr = img_arr[rect_min_x:rect_max_x, rect_min_y:rect_max_y]
                img_arr = np.flip(np.rot90(img_arr, k=1), axis=1)  # Rotate and flip to correct orientation

                # Convert image to grayscale
                gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)

                # Resize the image to 28x28
                resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
                normalized = resized / 255.0  # Normalize image

                reshaped = np.reshape(normalized, (1, 28, 28, 1))  # Reshape for model input
                result = MODEL.predict(reshaped)
                label = LABELS[np.argmax(result[0])]

                # Render the label
                try:
                    text_surface = FONT.render(label, True, RED)
                except UnicodeEncodeError:
                    label = label.encode('ascii', 'ignore').decode('ascii')
                    text_surface = FONT.render(label, True, RED)

                text_rect = text_surface.get_rect()
                text_rect.center = (rect_min_x + (rect_max_x - rect_min_x) // 2, rect_max_y + 10)
                DISPLAYSURF.blit(text_surface, text_rect)

            number_xcord.clear()
            number_ycord.clear()

        if event.type == KEYDOWN:
            if event.unicode == 'n':
                DISPLAYSURF.fill(BLACK)

    pygame.display.update()
