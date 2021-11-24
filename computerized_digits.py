from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
fonts = ['calibril.ttf', 'consola.ttf', 'cour.ttf', 'arial.ttf', 'segoeuil.ttf', 'cambria.ttc']
digits = [str(x) for x in range(10)]

def make_image(center: Tuple[int, int], text: str, font: str, font_size: int):
    img = Image.new('L', (28,28))
    draw = ImageDraw.Draw(img)
    fonttt = ImageFont.truetype(font, font_size)
    draw.text(center,text,(255),font=fonttt, anchor='mm', align='center')
    # img.show()
    return img

def make_training_array(amnt: int):
    labels = {str(x):[] for x in range(10)}
    for i in range(amnt):
        for label in digits:
            font = np.random.choice(fonts)
            font_size = np.random.randint(18,22)
            x,y = np.random.randint(-4,4,2)
            img = make_image((14+x,14+y), label, font, font_size).rotate(np.random.randint(-15,15))
            labels[label].append(img.copy())
    return labels

def plot_examples(arr: List):
    plt.figure(figsize=(20,10))
    columns = 5
    for i, image in enumerate(arr):
        plt.subplot(len(arr) // columns + 1, columns, i + 1)
        plt.imshow(image, cmap='gray')
    plt.show()


if __name__ == '__main__':
    train = make_training_array(50)
    plot_examples(train['4'][:10])