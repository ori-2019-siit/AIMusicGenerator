from AIMusicGenerator.pixel_cnn_related.midi_to_img import get_input_paths
from PIL import Image
import numpy as np
import os


def mask_image(image):
    height, width, channels = image.shape
    third = int(width / 3)

    for row in image[:,third:]:
        for pixel in row:
            pixel.fill(0)

    return image


def load_image(path):
    img = Image.open(path)
    img.load()
    img = img.rotate(90)
    return np.array(img, dtype=np.uint8)


def save_image(image, path):
    img = Image.fromarray(image, 'RGB')
    img = img.rotate(-90)
    img.save(path)


def mask_all_images(path):
    paths = get_input_paths(path)

    for path in paths:
        img = load_image(path)
        img = mask_image(img)
        save_image(img, path)


if __name__ == '__main__':
    mask_all_images(os.path.join("..\\dataset_for_pixel_cnn", "test"))
