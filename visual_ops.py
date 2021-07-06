from PIL import Image
from ProblemSet import ProblemSet

from PIL import ImageDraw, ImageChops, ImageStat, ImageOps
import numpy as np
import math
import operator
from functools import reduce
from itertools import product

def mse(f1, f2):
    n1 = np.array(list(f1.getdata()))
    n2 = np.array(list(f2.getdata()))
    err = np.sum((n1 - n2)**2)
    size = float(f1.size[0]**2)
    return math.sqrt(err/size)

def bk_insert(f1, f2):
    n1 = np.array(list(f1.getdata()))
    n2 = np.array(list(f2.getdata()))
    # err = (n1 - n2) < 0
    # print(err)
    err = np.sum((n1 - n2) < 0)
    size = float(f1.size[0]**2)
    return err / size

def similarity_score(im1, im2):
    """
    Pixelwise diff percent between im1 and im2
    :param im1:
    :param im2:
    :return: 0. ~ 1.
    """
    disturb = [ImageChops.offset(im1, dx, dy) for dx, dy in product(range(-2,3), range(-2,3))]
    outs = [ImageChops.difference(d1, im2) for d1 in disturb]
    residual = min([ImageStat.Stat(out).sum[0] for out in outs])
    score = 1. - (residual / 255.) / (im1.size[0] * im1.size[1])
    return score

def get_objects(im):
    width, height = im.size
    visited = set()
    objects = []
    delta = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]

    def _dfs(x, y):
        queue = [(x, y)]
        obj = []
        while queue:
            x, y = queue.pop()
            visited.add((x,y))
            obj.append((x, y))
            for dx, dy in delta:
                nx, ny = x+dx, y+dy
                if nx >= width or nx < 0 or ny >= height or ny < 0 or (nx, ny) in visited or im.getpixel((nx, ny)) != 0:
                    continue
                queue.append((nx, ny))
        return obj

    for x in range(width):
        if len(objects) >= 12:
            break
        for y in range(height):
            pix = im.getpixel((x, y))
            if pix != 0 or (x,y) in visited:
                continue
            o = _dfs(x, y)
            if len(o) > 20 and min(o[0]) < max(o[0]) + 10: # scattered pixels or too small
                objects.append(o)
    return objects

# def identify_objects(im):
#     width, height = im.size
#     objects = []
#     dark_fill_val = 1
#     light_fill_val = 254
#     for x in range(width):
#         for y in range(height):
#             xy = (x, y)
#             l_val = im.getpixel(xy)
#
#             if l_val == 0:
#                 ImageDraw.floodfill(im, xy, dark_fill_val)
#                 # im.show()
#                 objects.append(([xy], dark_fill_val))
#                 dark_fill_val += 1
#             elif l_val == 255:
#                 ImageDraw.floodfill(im, xy, light_fill_val)
#                 # im.show()
#                 objects.append(([xy], light_fill_val))
#                 light_fill_val -= 1
#             else:
#                 for obj in objects:
#                     if obj[1] == l_val:
#                         obj[0].append(xy)
#                         break
#     return objects

def black_start(im):
    width, height = im.size
    for x in range(width):
        for y in range(height):
            pix = im.getpixel((x, y))
            if pix != 255:
                return x,y
    return -1, -1

def shift_add(im1, im2):
    x1, y1 = black_start(im1)
    x2, y2 = black_start(im2)
    if x2 == -1:
        return None
    offx, offy = x1-x2, y1-y2
    return ImageChops.multiply(ImageChops.offset(im1, offx, offy), ImageChops.offset(im1, -offx, -offy))

def same_object(o1, o2):

    image1 = Image.new('L', (184,184), color=(255))
    d =  ImageDraw.Draw(image1)
    d.point(o1, fill=(0))
    image2 = Image.new('L', (184,184), color=(255))
    d = ImageDraw.Draw(image2)
    d.point(o2, fill=(0))
    return similarity_score(image1, image2)


def cat_all_figures(prompt):
    """

    :param prompt: given images len=8
    :return: cancatenated as 3 x3 x2 ==> with one missing part to be filled by the key
    """

    x, y = imsize

    whole = Image.new('L', (x * 3, y * 3))

    x_off, y_off = 0, 0
    for i in range(3):
        whole.paste(prompt[i], (x_off, y_off))
        x_off += x
    y_off += y
    x_off = 0
    for i in range(3, 6):
        whole.paste(prompt[i], (x_off, y_off))
        x_off += x
    y_off += y
    x_off = 0
    for i in range(6, 9):
        whole.paste(prompt[i], (x_off, y_off))
        x_off += x
    return whole

def paste_and_pad(img_a):
    # p 9 a->b
    leftb = (0, 0, imsize[0]/2, imsize[1])
    leftb = img_a.crop(leftb)
    rightb = (imsize[0]/2, 0, imsize[0], imsize[1])
    rightb = img_a.crop(rightb)

    new_size = dst.size
    delta_w = imsize[0] - new_size[0]
    delta_h = imsize[1] - new_size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    new_im = ImageOps.expand(dst, padding, fill=255)
    new_im.show()

def get_object_size(obj):
    """
    black pixel counts after floodfilling the object
    :param obj:
    :return:
    """
    xy = get_center(obj)

    new_im = Image.new('L', imsize, color=255)
    d = ImageDraw.Draw(new_im)
    d.point(obj, fill=(0))

    ImageDraw.floodfill(new_im, xy, 0)

    return new_im.histogram()[0]

def get_centroid(im):
    (X, Y) = im.size
    immat = im.load()
    m = np.zeros((X, Y))

    for x in range(X):
        for y in range(Y):
            m[x, y] = immat[(x, y)] != 255
    m = m / np.sum(np.sum(m))
    # marginal distributions
    dx = np.sum(m, 1)
    dy = np.sum(m, 0)

    # expected values
    cx = np.sum(dx * np.arange(X))
    cy = np.sum(dy * np.arange(Y))
    return (cx, cy)

def black_ratio_sim(im1, im2):
    total_pixel = im1.size[0] * im1.size[1]
    bk1 = im1.histogram()[0] / total_pixel
    bk2 = im2.histogram()[0] / total_pixel
    return 1 - abs(bk1 - bk2)



if __name__ == "__main__":
    from Utils import Shape, similarity_score
    problem_set = ProblemSet("Challenge Problems D")
    p = problem_set.problems[3]
    file_a = p.figures['A'].visualFilename
    file_b = p.figures['B'].visualFilename
    file_c = p.figures['C'].visualFilename
    file_d = p.figures['D'].visualFilename
    file_e = p.figures['E'].visualFilename
    file_f = p.figures['F'].visualFilename
    file_g = p.figures['G'].visualFilename
    file_h = p.figures['H'].visualFilename

    file_1 = p.figures['1'].visualFilename
    file_2 = p.figures['2'].visualFilename
    file_3 = p.figures['3'].visualFilename
    file_4 = p.figures['4'].visualFilename
    file_5 = p.figures['5'].visualFilename
    file_6 = p.figures['6'].visualFilename
    file_7 = p.figures['7'].visualFilename
    file_8 = p.figures['8'].visualFilename
    # https://pillow.readthedocs.io/en/stable/reference/Image.html

    thresh = 200
    fn = lambda x: 255 if x > thresh else 0

    img_a = Image.open(file_a).convert('L')

    img_b = Image.open(file_b).convert('L')
    img_c = Image.open(file_c).convert('L')
    img_d = Image.open(file_d).convert('L')
    img_e = Image.open(file_e).convert('L')
    img_f = Image.open(file_f).convert('L')
    img_g = Image.open(file_g).convert('L')
    img_h = Image.open(file_h).convert('L')

    img_1 = Image.open(file_1).convert('L')
    img_2 = Image.open(file_2).convert('L')
    img_3 = Image.open(file_3).convert('L')
    img_4 = Image.open(file_4).convert('L')
    img_5 = Image.open(file_5).convert('L')
    img_6 = Image.open(file_6).convert('L')
    img_7 = Image.open(file_7).convert('L')
    img_8 = Image.open(file_8).convert('L')



    out = img_h.rotate(315, fillcolor=(255))
    # out.show()
    s = similarity_score(out, img_5)
    print(s)
    # s = similarity_score(img_e, ImageOps.flip(img_e))
    # print(s)
    # s = similarity_score(ImageOps.mirror(img_a), img_c)
    # print(s)
    # s = similarity_score(ImageOps.flip(img_d), img_f)
    # print(s)



















#





















