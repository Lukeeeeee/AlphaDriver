import itertools
from PIL import Image


def slice_queue(q, left, right):
    return list(itertools.islice(q, left, right))


def show_pic(img):
    im = Image.fromarray(img)
    im.show()
