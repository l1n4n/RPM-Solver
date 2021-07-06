from PIL import Image, ImageDraw, ImageChops, ImageStat, ImageOps
import numpy as np
import math
import operator
from functools import reduce
from itertools import product, combinations


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


def black_ratio_sim(im1, im2):
    total_pixel = im1.size[0] * im1.size[1]
    bk1 = im1.histogram()[0] / total_pixel
    bk2 = im2.histogram()[0] / total_pixel
    return 1 - abs(bk1 - bk2)


def shape_similarity(o1, o2):
    # - (residual ratio diff + number of residual blocks diff)
    n1, r1, h1 = o1
    n2, r2, h2 = o2
    s = int(n1 == n2)
    if n1 >= 12 and n2 >= 12:
        s = 1
    return s - (abs(r1 - r2)) - abs((h1 - h2))


def count_shapes(lo):
    shape_cluster = {}
    for o_shape in lo:
        o_shape = list(o_shape)
        o_shape = [round(i, 2) for i in o_shape]
        o_shape = tuple(o_shape)
        if not shape_cluster:
            shape_cluster[o_shape] = 1
        else:
            for k in shape_cluster.copy():
                if shape_similarity(k, o_shape) > 0.9:
                    shape_cluster[k] += 1
                    break
                else:
                    shape_cluster[o_shape] = 1
    return shape_cluster


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


def logical_ops(im1, im2):
    """
    return a-b, a|b, a+b
    """
    a = im1.convert(mode='1')
    b = im2.convert(mode='1')
    xor = ImageChops.logical_xor(a, b)
    _or = ImageChops.logical_or(a, b)
    _and = ImageChops.logical_and(a, b)
    return (ImageChops.invert(xor), _or, _and)


class Shape:
    def __init__(self, obj, imsize=(184,184)):
        self.data = obj
        self.imsize = imsize
        self.filled = False
        self.center = (92, 92)
        self.box = self.get_box()
        self.img = self.get_protoimg()
        self.size = self.get_size()
        self.corners = self.get_corners()
        self.shape = self.get_shape()

    def get_protoimg(self):
        imsize = self.imsize
        ob = Image.new('L', imsize, color=255)
        e = ImageDraw.Draw(ob)
        e.point(self.data, fill=(0))
        bk_before = ob.histogram()[0]
        # bk_before = len(self.data)
        self.center = self.get_center(ob)
        center_value = ob.getpixel(self.center)
        ImageDraw.floodfill(ob, self.center, 0)
        bk_after = ob.histogram()[0]
        self.filled = bk_before / bk_after > 0.5 and (center_value != 255) # get filled pixels ratio
        self.fill_ratio = bk_before/bk_after
        return ob

    def get_size(self):
        return self.img.histogram()[0] / (self.imsize[0] * self.imsize[1])

    def get_box(self):
        obj = self.data
        minx = min(obj)[0]
        maxx = max(obj)[0]
        obj_ = sorted(obj, key=lambda x: x[1])
        miny = obj_[0][1]
        maxy = obj_[-1][1]
        # return (minx, miny), (maxx, miny), (minx, maxy), (maxx, maxy)
        return minx, miny, maxx, maxy

    def get_center(self, im):
        # BEGIN CODE FROM https://stackoverflow.com/questions/37519238/python-find-center-of-object-in-an-image
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
        # END CODE FROM https://stackoverflow.com/questions/37519238/python-find-center-of-object-in-an-image
        return (cx, cy)

    def get_corners(self):
        I = self.img
        # BEGIN CODE FROM https://codegolf.stackexchange.com/questions/65280/count-the-number-of-sides-on-a-polygon
        w, h = I.size
        D = I.getdata()
        B = {i % w + i // w * 1j for i in range(w * h) if D[i] != 255}
        n = d = 1
        o = v = q = p = max(B, key=abs)
        while p - w:
            p += d * 1j
            e = 2 * ({p} < B) + ({p + d} < B)
            if e != 2:
                e %= 2
                d *= 1j - e * 2j
                p -= d / 1j ** e
            if abs(p - q) > 5: # could be tuned
                t = (q - v) * (p - q).conjugate()
                q = p
                w = o
                if .98 * abs(t) > t.real:
                    n += 1
                    v = p
        # END CODE FROM https://codegolf.stackexchange.com/questions/65280/count-the-number-of-sides-on-a-polygon
        # print(n)
        # if n >= 12:
        #     return 12 # circle
        return n

    def get_shape(self):
        new_im = Image.new('L', self.imsize, color=255)
        d = ImageDraw.Draw(new_im)
        d.rectangle(self.box, fill=(0))

        # residual
        out = ImageChops.invert(ImageChops.difference(new_im, self.img))
        # residual = get_objects(out)

        # residual blocks number
        # n = len(residual)
        n = self.corners

        # size of residual
        minx, miny, maxx, maxy = self.box
        rsize = out.histogram()[0] / ((maxx - minx) * (maxy - miny))

        # height width ratio
        hw = (maxy - miny) / (maxx - minx)

        if rsize < 0.003 and n == 5:
            n -= 1 # to correct square detection
        return (n, rsize, hw)


def get_img(p, title):
    f = p.figures[title].visualFilename
    org = Image.open(f).convert('L')
    return org


def xor_and_common(im1, im2, im3):
    a = im1.convert('1')
    b = im2.convert('1')
    c = im3.convert('1')
    xor = ImageChops.invert(ImageChops.logical_xor(a, b))
    common = ImageChops.add(ImageChops.add(a, b), c)
    return ImageChops.logical_and(xor, common)


def one_pattern_each(a,b,c,d,e,f,g,h):
    # resize and convert to 1
    # a_ = a.resize((64, 64), resample=Image.BICUBIC).convert('1')
    # b_ = b.resize((64, 64), resample=Image.BICUBIC).convert('1')
    # c_ = c.resize((64, 64), resample=Image.BICUBIC).convert('1')
    # d_ = d.resize((64, 64), resample=Image.BICUBIC).convert('1')
    # e_ = e.resize((64, 64), resample=Image.BICUBIC).convert('1')
    # f_ = f.resize((64, 64), resample=Image.BICUBIC).convert('1')
    # g_ = g.resize((64, 64), resample=Image.BICUBIC).convert('1')
    # h_ = h.resize((64, 64), resample=Image.BICUBIC).convert('1')
    # l = [a_, b_, c_, d_, e_, f_, g_, h_]
    l = [a,b,c,d,e,f,g,h]
    for im1, im2 in combinations(l, 2):
        if similarity_score(im1, im2) > 0.97:
            return None
    return l


def black_start(im):
    width, height = im.size
    for x in range(width):
        for y in range(height):
            pix = im.getpixel((x, y))
            if pix != 255:
                return x, y
    return -1, -1


def get_left_offsets(im1, im2):
    x1, y1 = black_start(im1)
    x2, y2 = black_start(im2)
    if x2 == -1:
        return None, None
    offx, offy = x1 - x2, y1 - y2
    return offx, offy


def black_end(im):
    width, height = im.size
    for x in range(width-1, -1, -1):
        for y in range(height-1, -1, -1):
            pix = im.getpixel((x, y))
            if pix != 255:
                return x, y
    return -1, -1


def get_right_offsets(im1, im2):
    x1, y1 = black_end(im1)
    x2, y2 = black_end(im2)
    if x2 == -1:
        return None, None
    offx, offy = x1 - x2, y1 - y2
    return offx, offy


def same_object(o1, o2, thred=0.985):
    image1 = Image.new('L', (184, 184), 255)
    image2 = Image.new('L', (184, 184), 255)
    e = ImageDraw.Draw(image1)
    e.point(o1, fill=(0))
    f = ImageDraw.Draw(image2)
    f.point(o2, fill=(0))
    return similarity_score(image1, image2) > thred


def fill_objects(objects, shapes, imsize):
    new_im = Image.new('L', imsize, color=255)
    d = ImageDraw.Draw(new_im)
    for _ in objects:
        d.point(_, fill=(0))
    for o in shapes:
        ImageDraw.floodfill(new_im, o.center, 0)
    return new_im


if __name__ == "__main__":
    from ProblemSet import ProblemSet

    problem_set = ProblemSet("Basic Problems E")
    p = problem_set.problems[8]
    a = get_img(p, 'A')
    b = get_img(p, 'B')
    c = get_img(p, 'C')
    d = get_img(p, 'D')
    e = get_img(p, 'E')
    f = get_img(p, 'F')
    g = get_img(p, 'G')
    h = get_img(p, 'H')
    keys = ['1', '2', '3', '4', '5', '6', '7', '8']
    kims = [get_img(p, k) for k in keys]

    objs_a = get_objects(a)
    objs_b = get_objects(b)
    objs_c = get_objects(c)
    objs_d = get_objects(d)
    objs_e = get_objects(e)
    objs_f = get_objects(f)
    objs_g = get_objects(g)
    objs_h = get_objects(h)

    kos = {}
    for i, k in enumerate(kims):
        ko = get_objects(k)
        kos[i] = ko

    if len(objs_a) == len(objs_b) == len(objs_c) == 2:
        parts = []
        for idx, o in enumerate(objs_a):
            if same_object(objs_c[0], o):
                parts.append(idx)
                break
        for idx, o in enumerate(objs_b):
            if same_object(objs_c[1], o):
                parts.append(idx)
                break
        if len(parts) == 2:
            out = Image.new('L', a.size, 255)
            e = ImageDraw.Draw(out)
            e.point(objs_g[parts[0]], fill=(0))
            e.point(objs_h[parts[1]], fill=(0))
            out.show()
            for k in kims:
                s = similarity_score(out, k)
                print(s)

    # shapes_a = [Shape(o) for o in objs_a]
    # shapes_b = [Shape(o) for o in objs_b]
    # shapes_c = [Shape(o) for o in objs_c]
    # shapes_d = [Shape(o) for o in objs_d]
    # shapes_e = [Shape(o) for o in objs_e]
    # shapes_f = [Shape(o) for o in objs_f]
    # shapes_g = [Shape(o) for o in objs_g]
    # shapes_h = [Shape(o) for o in objs_h]

    # kshapes = {}
    # for i, ko in kos.items():
    #     ks = [Shape(o) for o in ko]
    #     kshapes[i] = ks


    # offx, offy = get_left_offsets(a, b)
    # print(offx, offy)
    #
    # out = ImageChops.invert(ImageChops.logical_xor(a, ImageChops.offset(b.rotate(180), offx, 0)))
    # out.show()
    #
    # c = kims[6]
    # offx, offy = get_right_offsets(c, out)
    # print(offx, offy)
    # ImageChops.offset(out, offx, 0).show()
    # s = similarity_score(c, ImageChops.offset(out, offx, 0))
    # print(s)

    # objs_a = get_objects(a)
    # objs_b = get_objects(b)
    # objs_c = get_objects(c)
    # objs_d = get_objects(d)
    # objs_e = get_objects(e)
    # objs_f = get_objects(f)
    # objs_g = get_objects(g)
    # objs_h = get_objects(h)
    #
    # kos = {}
    # for i, k in enumerate(kims):
    #     ko = get_objects(k)
    #     kos[i] = ko
    #
    # shapes_a = [Shape(o) for o in objs_a]
    # shapes_b = [Shape(o) for o in objs_b]
    #
    # shapes_c = [Shape(o) for o in objs_c]
    # shapes_d = [Shape(o) for o in objs_d]
    # shapes_e = [Shape(o) for o in objs_e]
    # shapes_f = [Shape(o) for o in objs_f]
    # shapes_g = [Shape(o) for o in objs_g]
    # shapes_h = [Shape(o) for o in objs_h]














