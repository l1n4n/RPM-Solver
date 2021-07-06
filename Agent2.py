# Your Agent for solving Raven's Progressive Matrices. You MUST modify this file.
#
# You may also create and submit new files in addition to modifying this file.
#
# Make sure your file retains methods with the signatures:
# def __init__(self)
# def Solve(self,problem)
#
# These methods will be necessary for the project's main method to run.

# Install Pillow and uncomment this line to access image processing.
from PIL import Image, ImageChops, ImageStat, ImageDraw, ImageOps
import numpy as np
from collections import defaultdict
from itertools import product
from Utils import similarity_score, black_ratio_sim, shape_similarity, get_objects, Shape, count_shapes

# BEGIN CODE FROM https://stackoverflow.com/questions/366682/how-to-limit-execution-time-of-a-function-call-in-python
from contextlib import contextmanager
import threading
import _thread

class TimeoutException(Exception):
    def __init__(self, msg=''):
        self.msg = msg

@contextmanager
def time_limit(seconds, msg=''):
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        # raise TimeoutException("Timed out for operation {}".format(msg))
        print(msg)
        return
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()
# END CODE FROM https://stackoverflow.com/questions/366682/how-to-limit-execution-time-of-a-function-call-in-python

class Visual:
    def __init__(self):
        self.imsize = None
        self.TRANS = [Image.NONE, Image.FLIP_TOP_BOTTOM, Image.FLIP_LEFT_RIGHT, Image.ROTATE_270, Image.ROTATE_180,
                      Image.ROTATE_90]

    def look_same(self, im1, im2, im3, threds=0.98):
        if similarity_score(im1, im2)  > threds and similarity_score(im2, im3) > threds:
            return True
        return False

    def _get_img(self, p, title):
        f = p.figures[title].visualFilename
        org = Image.open(f).convert('L')
        if not self.imsize:
            self.imsize = (org.size[0], org.size[1])
        return org

    def black_start(self, im):
        width, height = im.size
        for x in range(width):
            for y in range(height):
                pix = im.getpixel((x, y))
                if pix != 255:
                    return x, y
        return -1, -1

    def get_offsets(self, im1, im2):
        x1, y1 = self.black_start(im1)
        x2, y2 = self.black_start(im2)
        if x2 == -1:
            return None
        offx, offy = x1 - x2, y1 - y2
        return offx, offy

    def shift_add(self, offx, offy, im1):
        return ImageChops.multiply(ImageChops.offset(im1, offx, offy), ImageChops.offset(im1, -offx, -offy))

    def object_move(self, objs, offs, im2, im3):
        # offx = x1 -x2
        o1 = objs[0]
        o2 = objs[1]
        offx, offy = offs[0], offs[1]
        middle = Image.new('L', self.imsize, color=(255))
        d = ImageDraw.Draw(middle)
        d.point([(x - offx, y - offy) for (x, y) in o1], fill=(0))
        d.point([(x + offx, y + offy) for (x, y) in o2], fill=(0))
        # middle.show()
        right = Image.new('L', self.imsize, color=(255))
        e = ImageDraw.Draw(right)
        e.point([(x - offx * 2, y - offy * 2) for (x, y) in o1], fill=(0))
        e.point([(x + offx * 2, y + offy * 2) for (x, y) in o2], fill=(0))
        # right.show()
        if similarity_score(middle, im2) > 0.9 and similarity_score(right, im3) > 0.9\
                and black_ratio_sim(middle, im2) > 0.98 and black_ratio_sim(right, im3) > 0.98:
            return True
        return False

    def same_object(self, o1, o2, thred=0.985):
        image1 = o1.img
        image2 = o2.img
        return similarity_score(image1, image2) > thred

    def find_unchanged_object(self, set1, set2):
        unchanged = []
        u1 = []
        u2 = []
        for i, oa in enumerate(set1):
            for j, ob in enumerate(set2):
                if self.same_object(oa, ob):
                    unchanged.append(oa)
                    u1.append(i)
                    u2.append(j)

        return unchanged, [set1[i] for i in range(len(set1)) if i not in u1], [set2[i] for i in range(len(set2)) if i not in u2]

    def cat_all_figures(self, prompt):
        """
        :param prompt: given images len=8
        :return: cancatenated as 3 x3 x2 ==> with one missing part to be filled by the key
        """
        x, y = self.imsize
        whole = Image.new('L', (x*3, y*3))

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
        for i in range(6, 8):
            whole.paste(prompt[i], (x_off, y_off))
            x_off += x
        return whole

    def symmetry_vertical(self, im):
        return similarity_score(im, ImageOps.mirror(im))

    def symmetry_horizontal(self, im):
        return similarity_score(im, ImageOps.flip(im))

    def symmetry_major_diagonal(self, im):
        return similarity_score(im, im.transpose(Image.TRANSPOSE))

    def symmetry_minor_diagonal(self, im):
        return similarity_score(im, im.transpose(Image.TRANSVERSE))

    def subtract_and_trans(self, diff12, diff23, thred=0.985):
        """
        diff12 ---trans---> diff23
        :param diff12:
        :param diff23:
        :param thred:
        :return:
        """
        for idx, t in enumerate(self.TRANS):
            res = diff12.transpose(method=t)
            if similarity_score(res, diff23) > thred:
                return idx
        return -1

    def get_answer(self, p):
        a = self._get_img(p, 'A')
        b = self._get_img(p, 'B')
        c = self._get_img(p, 'C')
        d = self._get_img(p, 'D')
        e = self._get_img(p, 'E')
        f = self._get_img(p, 'F')
        g = self._get_img(p, 'G')
        h = self._get_img(p, 'H')

        keys = ['1', '2', '3', '4', '5', '6', '7', '8']
        kims = [self._get_img(p, k) for k in keys]

        scores = [0 for i in range(len(keys))]

        # check holistic pattern
        missing_one = self.cat_all_figures([a,b,c,d,e,f,g,h])
        x_off, y_off = self.imsize
        x_off *= 2
        y_off *= 2
        # missing_one.save('whole.png')
        for i, k in enumerate(kims):
            whole = missing_one.copy()
            whole.paste(k, (x_off, y_off))
            # check symmetry
            h_score = self.symmetry_horizontal(whole)
            if h_score > 0.985:
                scores[i] += h_score
            v_score = self.symmetry_vertical(whole)
            if v_score > 0.985:
                scores[i] += v_score
            d1_score = self.symmetry_major_diagonal(whole)
            if d1_score > 0.985:
                scores[i] += d1_score
            d2_score = self.symmetry_minor_diagonal(whole)
            if d2_score > 0.985:
                scores[i] += d2_score

        # check row identity
        if self.look_same(a, b, c) and self.look_same(d,e,f):
            for i, k in enumerate(kims):
                sim = similarity_score(g, k)
                if sim > 0.98:
                    scores[i] += sim

        # check col identity
        if self.look_same(a, d, g) and self.look_same(b, e, h):
            for i, k in enumerate(kims):
                sim = similarity_score(c, k)
                if sim > 0.98:
                    scores[i] += sim

        # check bc shift and add
        offs1 = self.get_offsets(b, c)
        bc = self.shift_add(offs1[0], offs1[1], b)
        sim_bc = similarity_score(bc, c)

        offs2 = self.get_offsets(e, f)
        ef = self.shift_add(offs2[0], offs2[1], e)
        sim_ef = similarity_score(ef, f)
        if max(sim_bc, sim_ef) > 0.98:
            offx, offy = (offs1[0] + offs2[0])//2, (offs1[1] + offs2[1])//2
            gold = ImageChops.multiply(ImageChops.offset(h, offx, offy), ImageChops.offset(h, -offx, -offy))
            for i, k in enumerate(kims):
                sim = similarity_score(gold, k)
                if sim > 0.88:
                    scores[i] += sim

        # check ac shift and add
        offs3 = self.get_offsets(a, c)
        ac = self.shift_add(offs3[0], offs3[1], a)
        offs4 = self.get_offsets(d, f)
        df = self.shift_add(offs4[0], offs4[1], d)
        sim_ac = similarity_score(ac, c)
        sim_df = similarity_score(df, f)
        if max(sim_ac, sim_df) > 0.98:
            offx, offy = (offs3[0] + offs4[0]) // 2, (offs3[1] + offs4[1]) // 2
            gold = ImageChops.multiply(ImageChops.offset(g, offx, offy), ImageChops.offset(g, -offx, -offy))
            for i, k in enumerate(kims):
                sim = similarity_score(gold, k)
                if sim > 0.88:
                    scores[i] += sim

        # check unchanged part
        # row-wise
        same_ab = ImageChops.add(a, b)
        same_bc = ImageChops.add(b, c)
        bkcount = min(same_ab.histogram()[0], same_bc.histogram()[0]) / (self.imsize[0] * self.imsize[1])
        if similarity_score(same_ab, same_bc) > 0.985 and bkcount > 0.015: # something is unchanged
            same_gh = ImageChops.add(g, h)
            for i, k in enumerate(kims):
                same_hk = ImageChops.add(h, k)
                sim = similarity_score(same_gh, same_hk)
                if sim > 0.98:
                    scores[i] += sim

        # col-wise
        same_ad = ImageChops.add(a, d)
        same_dg = ImageChops.add(d, g)
        bkcount = min(same_ad.histogram()[0], same_dg.histogram()[0]) / (self.imsize[0] * self.imsize[1])
        if similarity_score(same_ad, same_dg) > 0.985 and bkcount > 0.015:
            same_cf = ImageChops.add(c, f)
            for i, k in enumerate(kims):
                same_fk = ImageChops.add(f, k)
                sim = similarity_score(same_cf, same_fk)
                if sim > 0.98:
                    scores[i] += sim

        # check differences with possible transformation
        # row-wise
        diff_ab = ImageChops.difference(a, b)
        diff_de = ImageChops.difference(d, e)
        diff_gh = ImageChops.difference(g, h)
        diff_bc = ImageChops.difference(b, c)

        if similarity_score(diff_ab, diff_de) > 0.98 and similarity_score(diff_de, diff_gh) > 0.98:
            gold = diff_bc
            gold2 = ImageChops.multiply(ImageChops.invert(gold), h)
            for i, k in enumerate(kims):
                diff_hk = ImageChops.difference(h, k)
                sim = similarity_score(gold, diff_hk)
                if sim > 0.98:
                    scores[i] += sim
                sim2 = similarity_score(gold2, k)
                sim3 = black_ratio_sim(gold2, k) # problem c-12 why choose 7 instead of 8?
                if sim2 > 0.98:
                    scores[i] += sim2 + sim3
        # col-wise
        diff_ad = ImageChops.difference(a, d)
        diff_be = ImageChops.difference(b, e)
        diff_cf = ImageChops.difference(c, f)
        diff_dg = ImageChops.difference(d, g)

        if similarity_score(diff_ad, diff_be) > 0.98 and similarity_score(diff_be, diff_cf) > 0.98:
            gold = diff_dg
            gold2 = ImageChops.multiply(ImageChops.invert(gold), f)
            for i, k in enumerate(kims):
                diff_fk = ImageChops.difference(f, k)
                sim = similarity_score(gold, diff_fk)
                if sim > 0.98:
                    scores[i] += sim
                sim2 = similarity_score(gold2, k)
                sim3 = black_ratio_sim(gold2, k) # problem c-12 why choose 7 instead of 8?
                if sim2 > 0.98:
                    scores[i] += sim2 + sim3

        # check subtract_and_trans
        # row-wise
        subtract_trans = self.subtract_and_trans(diff_ab, diff_bc)
        if subtract_trans >= 0:
            gold = ImageChops.subtract(g, h).transpose(self.TRANS[subtract_trans])
            for i, k in enumerate(kims):
                diffhk = ImageChops.subtract(h, k)
                sim = similarity_score(gold, diffhk)
                if sim > 0.98:
                    scores[i] += sim * 2
        # col-wise
        subtract_trans = self.subtract_and_trans(diff_ad, diff_dg)
        if subtract_trans >= 0:
            gold = ImageChops.subtract(c, f).transpose(self.TRANS[subtract_trans])
            for i, k in enumerate(kims):
                difffk = ImageChops.subtract(f, k)
                sim = similarity_score(gold, difffk)
                if sim > 0.98:
                    scores[i] += sim * 2

        # objects detection
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

        # cut-half and paste
        black_pixels_b = [_ for o in objs_b for _ in o]
        x_ords = [o[0] for o in black_pixels_b]
        y_ords = [o[1] for o in black_pixels_b]
        lx, rx = min(x_ords), max(x_ords)
        uy, by = min(y_ords), max(y_ords)
        leftbox = (lx, uy, (lx+rx)//2, by)
        # print(leftbox)
        rightbox = (self.imsize[0]//2, uy, self.imsize[0]//2+(rx-lx)//2, by)
        # print(rightbox)
        upbox = (0, 0, self.imsize[1], self.imsize[0] // 2)
        botbox = (0, self.imsize[0]//2, self.imsize[1],self.imsize[1])
        # upbox = (lx, uy, rx, (uy+by) // 2)
        # botbox = (lx, self.imsize[1]//2, rx, self.imsize[1]//2 + (by-uy)//2)

        rb = b.crop(rightbox)
        bd = d.crop(botbox)
        rh = h.crop(rightbox)
        bf = f.crop(botbox)

        outa = a.copy()
        outb = b.copy()

        outh = h.copy()
        outf = f.copy()

        # paste right then left
        outa.paste(rb, rightbox)
        if similarity_score(b, outa) > 0.9:
            outb.paste(rb, leftbox)
            # outb.show()
            if similarity_score(c, outb) > 0.9: # thredshold problem? challenge 08 >0.86, but basics cannot be too low
                outh.paste(rh, leftbox)
                for i, k in enumerate(kims):
                    sim = similarity_score(outh, k)
                    if sim > 0.9 and black_ratio_sim(outh, k) > 0.98:
                        scores[i] += sim
        # # paste bottom then upper =================== deprecate as it causes to wrong answers
        outa = a.copy()
        outd = d.copy()
        outa.paste(bd, botbox)
        if similarity_score(d, outa) > 0.9:
            outd.paste(bd, upbox)
            if similarity_score(g, outd) > 0.9:
                outf.paste(bf, upbox)
                for i, k in enumerate(kims):
                    sim = similarity_score(outf, k)
                    if sim > 0.9 and black_ratio_sim(outf, k) > 0.98:
                        scores[i] += sim

        # check object number pattern
        na, nb, nc, nd, ne, nf, ng, nh = len(objs_a), len(objs_b), len(objs_c), len(objs_d), len(objs_e), len(objs_f),\
        len(objs_g), len(objs_h)
        nks = [len(v) for v in kos.values()]
        # row-wise
        if (na-nb) == (nd-nd) == (ng-nh) and (nb-nc) == (ne-nf):
            for i, nk in enumerate(nks):
                if (nh-nk) == (ne-nf):
                    scores[i] += 1.
                else:
                    scores[i] -= 1.
        if (na+nb) == nc and (nd+ne) == nf:
            for i, nk in enumerate(nks):
                if (ng + nh) == nk:
                    scores[i] += 1.
                else:
                    scores[i] -= 1.
        # col-wise
        if (na-nd) == (nb-ne) == (nc-nf) and (nd-ng) == (ne-nh):
            for i, nk in enumerate(nks):
                if (nf-nk) == (nd-ng):
                    scores[i] += 1.
                else:
                    scores[i] -= 1.
        if (na+nd) == ng and (nb+ne) == nh:
            for i, nk in enumerate(nks):
                if (nc+nf) == nk:
                    scores[i] += 1.
                else:
                    scores[i] -= 1.

        # check object movement
        if len(objs_a) > 1:
            offab = self.get_offsets(a, b)
            res = self.object_move(objs_a, offab, b, c)
            if res:
                # apply the same trans on g to get h
                if len(objs_g) == 2:
                    gold = Image.new('L', self.imsize, color=(255))
                    d = ImageDraw.Draw(gold)
                    d.point([(x - offab[0] * 2, y - offab[1] * 2) for (x, y) in objs_g[0]], fill=(0))
                    d.point([(x + offab[0] * 2, y + offab[1] * 2) for (x, y) in objs_g[1]], fill=(0))
                    # gold.save('gold.png')
                    for i, k in enumerate(kims):
                        sim = similarity_score(gold, k)
                        # black pixel ratio similarity is used for problem c-09 answer 1 has higher similarity score, not sure why
                        bksim = black_ratio_sim(gold, k)
                        scores[i] += sim + bksim

        # size as sum of black pixels?
        sizes_a = [len(o) for o in objs_a]
        sizes_b = [len(o) for o in objs_b]
        sizes_c = [len(o) for o in objs_c]
        sizes_d = [len(o) for o in objs_d]
        sizes_e = [len(o) for o in objs_e]
        sizes_f = [len(o) for o in objs_f]
        sizes_g = [len(o) for o in objs_g]
        sizes_h = [len(o) for o in objs_h]

        sizes_ks = []
        for i, ko in kos.items():
            sizes_ks.append([len(o) for o in ko])

        # black pixel counts
        suma = sum(sizes_a)
        sumb = sum(sizes_b)
        sumc = sum(sizes_c)
        if abs(suma-sumb) >= 100 and abs(sumb-sumc) >= 100: # size changed
            sumg = sum(sizes_g)
            sumh = sum(sizes_h)
            if abs(sumg-sumh) >= 100:
                for i, k in enumerate(sizes_ks):
                    if abs(sum(k)-sumh) >= 100 and (sum(k) - sumh) *(sumc - sumb) > 0:
                        scores[i] += 0.5

        # with time_limit(30, 'Time Out'):
        shapes_a = [Shape(o) for o in objs_a]
        shapes_b = [Shape(o) for o in objs_b]
        shapes_c = [Shape(o) for o in objs_c]
        # shapes_d = [Shape(o) for o in objs_d]
        # shapes_e = [Shape(o) for o in objs_e]
        # shapes_f = [Shape(o) for o in objs_f]
        shapes_g = [Shape(o) for o in objs_g]
        shapes_h = [Shape(o) for o in objs_h]

        kshapes = {}
        for i, ko in kos.items():
            ks = [Shape(o) for o in ko]
            kshapes[i] = ks

        # shape counts
        na = count_shapes([_.shape for _ in shapes_a])
        nb = count_shapes([_.shape for _ in shapes_b])
        nc = count_shapes([_.shape for _ in shapes_c])

        ng = count_shapes([_.shape for _ in shapes_g])
        nh = count_shapes([_.shape for _ in shapes_h])
        kns = {}
        for i, ko in kshapes.items():
            nk = count_shapes([_.shape for _ in ko])
            kns[i] = nk

        # shape arithmetic
        if len(na) == len(nb) == len(nc) and len(ng) == len(nh):
            for i, k in kns.items():
                if len(k) == len(ng):
                    scores[i] += 1.

        # count total shapes in a row
        _shapes_row = list(na.keys()) + list(nb.keys()) + list(nc.keys())
        total_row = count_shapes(_shapes_row)
        _shapes_gh = list(ng.keys()) + list(nh.keys())
        for i, k in kns.items():
            _tmp = _shapes_gh + list(k.keys())
            _tmp_count = count_shapes(_tmp)
            if len(total_row) == len(_tmp_count):
                scores[i] += 1.

        # # filled arithmetic ===> cause wrong answer in test 4 and 8
        # total_filled_row = 0
        # total_unfilled_row = 0
        # for _s in shapes_a + shapes_b + shapes_c:
        #     if _s.filled:
        #         total_filled_row += 1
        #     else:
        #         total_unfilled_row += 1
        # filled_ratio = total_filled_row / (total_unfilled_row + total_filled_row)
        # for i, k in kshapes.items():
        #     _filled = 0
        #     _unfilled = 0
        #     for _t in shapes_g + shapes_h + k:
        #         if _t.filled:
        #             _filled += 1
        #         else:
        #             _unfilled += 1
        #     if _filled / (_unfilled + _filled) == filled_ratio:
        #         scores[i] += 0.5
        filled_a = [i.filled for i in shapes_a]
        filled_b = [i.filled for i in shapes_b]
        filled_c = [i.filled for i in shapes_c]
        if len(set(filled_a)) == len(set(filled_b)) == len(set(filled_c)) == 1:
            if filled_a[0] != filled_b[0] and filled_a[0] == filled_c[0]: # switch filled/unfilled pattern
                filled_g = [i.filled for i in shapes_g]
                filled_h = [i.filled for i in shapes_h]
                if len(set(filled_g)) == len(set(filled_h)) == 1 and filled_g[0] != filled_h[0]:
                    for i, k in kshapes.items():
                        filled_k = [i.filled for i in k]
                        if len(set(filled_k)) == 1 and filled_k[0] == filled_g[0]:
                            scores[i] += 0.5


        # find unchanged object
        unchanged, changed_a, changed_b = self.find_unchanged_object(shapes_a, shapes_b)
        unchanged_gh, changed_g, changed_h = self.find_unchanged_object(shapes_g, shapes_h)

        if len(unchanged) == len(unchanged_gh) and (len(changed_a) == len(changed_g) > 0):
            unchanged_c = set()
            for o in unchanged:
                for i, oc in enumerate(shapes_c):
                    if self.same_object(o, oc):
                        unchanged_c.add(i)
            changed_c = [shapes_c[i] for i in range(len(shapes_c)) if i not in unchanged_c]
            # find what has changed
            EXPAND = 0
            SAME_SHAPE = 0
            SHRINK = 0
            for oa, ob, oc in zip(changed_a, changed_b, changed_c):
                if shape_similarity(oa.shape, ob.shape) > 0.9 and shape_similarity(ob.shape, oc.shape) > 0.9: # same shape
                    SAME_SHAPE += 1
                    if oa.size * 1.1 < ob.size and ob.size * 1.1 < oc.size: # for visually difference
                        EXPAND += 1
                    elif ob.size * 1.1 < oa.size and oc.size * 1.1 < ob.size:
                        SHRINK += 1

            # check if unchanged in key
            kleft = {}
            for o in unchanged_gh:
                for i, k in kshapes.items():
                    _unchanged = []
                    for j, _o in enumerate(k):
                        if self.same_object(o, _o):
                            _unchanged.append(j)
                            scores[i] += 0.5
                            break
                    left = [k[_i] for _i in range(len(k)) if _i not in _unchanged]
                    kleft[i] = left

            # check if key contains object that has same shape/expand the changed object in h
            if SAME_SHAPE > 0 and (EXPAND > 0 or SHRINK > 0):
                for o in changed_h:
                    for i, k in kleft.items():
                        count_shape = 0
                        count_expand = 0
                        count_shrink = 0
                        for _o in k:
                            if not self.same_object(o, _o):
                                if shape_similarity(o.shape, _o.shape) > 0.9:
                                    count_shape += 1
                                    if o.size * 1.1 < _o.size:
                                        count_expand += 1
                                    elif _o.size * 1.1 < o.size:
                                        count_shrink += 1

                        scores[i] += int(count_shape == SAME_SHAPE) * 1. # shape should matters more, since there are misleading answers
                        scores[i] += int(count_expand == EXPAND) * .5
                        scores[i] += int(count_shrink == SHRINK) * .5

        ################ return highest score
        if max(scores) > 0:
            return scores.index(max(scores)) + 1
        return -1

class Agent:
    # The default constructor for your Agent. Make sure to execute any
    # processing necessary before your Agent starts solving problems here.
    #
    # Do not add any variables to this signature; they will not be used by
    # main().
    def __init__(self):
        # self.verbal_brain = SemanticNet()
        self.visual_brain = Visual()
    # The primary method for solving incoming Raven's Progressive Matrices.
    # For each problem, your Agent's Solve() method will be called. At the
    # conclusion of Solve(), your Agent should return an int representing its
    # answer to the question: 1, 2, 3, 4, 5, or 6. Strings of these ints
    # are also the Names of the individual RavensFigures, obtained through
    # RavensFigure.getName(). Return a negative number `to skip a problem.
    #
    # Make sure to return your answer *as an integer* at the end of Solve().
    # Returning your answer as a string may cause your program to crash.
    def Solve(self, problem):
        ans = self.visual_brain.get_answer(problem)
        return ans


if __name__ == "__main__":
    from ProblemSet import ProblemSet
    vv = Visual()
    # problem_set = ProblemSet("Basic Problems C")

    #### 3 4 4 8 3 7 2 5 2 7 4 8
    problem_set = ProblemSet("Challenge Problems C")
    ##### 7  7 3 8 4 7 3  1 7 3 4 2
    ##### -1 7 3 1 4 4 3 -1 7 5 4 1
    ##### 7 7 3 8 4 4 3 -1 5 5 4 2

    for i, problem in enumerate(problem_set.problems[5:6]):
        print(vv.get_answer(problem), end=' ')


