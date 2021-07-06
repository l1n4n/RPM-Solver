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
from Utils import similarity_score, black_ratio_sim, shape_similarity, get_objects, Shape, count_shapes, black_start,\
    black_end, get_left_offsets, get_right_offsets, xor_and_common, one_pattern_each, same_object, fill_objects


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

    def shift_add(self, offx, offy, im1):
        return ImageChops.multiply(ImageChops.offset(im1, offx, offy), ImageChops.offset(im1, -offx, -offy))

    def xor_with_offset(self, im1, im2, im3):
        a = im1.convert(mode='1')
        b = im2.convert(mode='1')
        c = im3.convert(mode='1')

        offx, offy = get_left_offsets(a, b)
        if not offx:
            return False

        out = ImageChops.invert(ImageChops.logical_xor(a, ImageChops.offset(b, offx, offy)))

        offx, offy = get_left_offsets(c, out)
        if not offx:
            return False
        s = similarity_score(c, ImageChops.offset(out, offx, offy))
        return s > 0.985

    def xor_with_rotate_offset(self, im1, im2, im3):
        a = im1.convert(mode='1')
        b = im2.convert(mode='1')
        c = im3.convert(mode='1')

        offx, offy = get_left_offsets(a, b)
        if not offx:
            return False

        out = ImageChops.invert(ImageChops.logical_xor(a, ImageChops.offset(b.rotate(180), offx, 0)))

        offx, offy = get_right_offsets(c, out)
        if not offx:
            return False
        s = similarity_score(c, ImageChops.offset(out, offx, 0))
        return s > 0.985

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

    def one_each_row(self, a,b,c,d,e,f):
        sim_dic = {}
        keys = ['d', 'e', 'f']
        for id, i in enumerate([d,e,f]):
            if similarity_score(i, a) > 0.98 and similarity_score(i, b) < 0.98 and similarity_score(i, c) < 0.98:
                sim_dic['a'] = keys[id]
            elif similarity_score(i, b) > 0.98 and similarity_score(i, a) < 0.98 and similarity_score(i, c) < 0.98:
                sim_dic['b'] = keys[id]
            elif similarity_score(i, c) > 0.98 and similarity_score(i, a) < 0.98 and similarity_score(i, b) < 0.98:
                sim_dic['c'] = keys[id]
        if sorted(list(sim_dic.values())) == keys:
            return True
        return False

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

        # check row identity
        if self.look_same(a, b, c) and self.look_same(d,e,f):
            for i, k in enumerate(kims):
                sim = similarity_score(g, k)
                if sim > 0.98:
                    scores[i] += sim
        else:
            # check one each in a row pattern
            if self.one_each_row(a,b,c,d,e,f):
                row_patterns = [a,b,c]
                matched = []
                for id, i in enumerate(row_patterns):
                    if similarity_score(i, g) > 0.98 or similarity_score(i, h) > 0.98:
                        matched.append(id)
                gold = [row_patterns[i] for i in range(len(row_patterns)) if i not in matched][0]
                for i, k in enumerate(kims):
                    sim = similarity_score(gold, k)
                    if sim > 0.98:
                        scores[i] += sim

            # sum of black pixels in each row --- help detect repeat row pattern d-02
            bk_row1 = a.histogram()[0] + b.histogram()[0] + c.histogram()[0]
            bk_row2 = d.histogram()[0] + e.histogram()[0] + f.histogram()[0]
            if abs(bk_row1 - bk_row2) <= 60:
                bk_gh = g.histogram()[0] + h.histogram()[0]
                for i, k in enumerate(kims):
                    tmp = bk_gh + k.histogram()[0]
                    # scores[i] -= abs(tmp - (bk_row1 + bk_row2)/2) / (self.imsize[0] ** 2)
                    if abs(tmp - (bk_row1 + bk_row2)/2) <= 60:
                        scores[i] += 0.25

        # check col identity
        if self.look_same(a, d, g) and self.look_same(b, e, h):
            for i, k in enumerate(kims):
                sim = similarity_score(c, k)
                if sim > 0.98:
                    scores[i] += sim
        else:
            # check one each in a col pattern
            if self.one_each_row(a,d,g,b,e,h):
                col_patterns = [a,d,g]
                matched = []
                for id, i in enumerate(col_patterns):
                    if similarity_score(i, c) > 0.98 or similarity_score(i, f) > 0.98:
                        matched.append(id)
                gold = [col_patterns[i] for i in range(len(col_patterns)) if i not in matched][0]
                for i, k in enumerate(kims):
                    sim = similarity_score(gold, k)
                    if sim > 0.98:
                        scores[i] += sim

            # sum of black pixels in each col --- help detect repeat row pattern d-02
            bk_col1 = a.histogram()[0] + d.histogram()[0] + g.histogram()[0]
            bk_col2 = b.histogram()[0] + e.histogram()[0] + h.histogram()[0]
            if abs(bk_col1 - bk_col2) <= 60:
                bk_cf = c.histogram()[0] + f.histogram()[0]
                for i, k in enumerate(kims):
                    tmp = bk_cf + k.histogram()[0]
                    # scores[i] -= abs(tmp - (bk_col1 + bk_col2) / 2) / (self.imsize[0] ** 2)
                    if abs(tmp - (bk_col1 + bk_col2) / 2) <= 60:
                        scores[i] += 0.25

        # check repeated row pattern D-02 03
        if similarity_score(b,f) > 0.98 and similarity_score(d,h) > 0.98 and similarity_score(a, e) > 0.98:
            for i, k in enumerate(kims):
                sim = similarity_score(e, k)
                if sim > 0.98:
                    scores[i] += sim/2
        elif similarity_score(b,d) > 0.98 and similarity_score(f,h) > 0.98 and similarity_score(c,e) > 0.98 and similarity_score(e,g) > 0.98:
            for i, k in enumerate(kims):
                sim = similarity_score(b, k)
                if sim > 0.98:
                    scores[i] += sim/2

        # logical and for basic e 1
        out = ImageChops.multiply(a, b)
        if similarity_score(out, c) > 0.985:
            gold = ImageChops.multiply(g, h)
            for i, k in enumerate(kims):
                sim = similarity_score(gold, k)
                if sim > 0.98:
                    scores[i] += sim
        out = ImageChops.multiply(a, d)
        if similarity_score(out, g) > 0.985:
            gold = ImageChops.multiply(c, f)
            for i, k in enumerate(kims):
                sim = similarity_score(gold, k)
                if sim > 0.98:
                    scores[i] += sim

        # logical xor with offset for e4
        if self.xor_with_offset(a, b, c):
            # apply to g, h
            offx, offy = get_left_offsets(g, h)
            g_ = g.convert('1')
            h_ = h.convert('1')
            gold = ImageChops.invert(ImageChops.logical_xor(g_, ImageChops.offset(h_, offx, offy)))
            for i, k in enumerate(kims):
                ox, oy = get_left_offsets(gold, k)
                sim = similarity_score(gold, ImageChops.offset(k, ox, oy))
                if sim > 0.98:
                    scores[i] += sim
        # logical xor with rotate e12
        if self.xor_with_rotate_offset(a, b, c):
            # apply to g, h
            offx, offy = get_left_offsets(g, h)
            g_ = g.convert('1')
            h_ = h.convert('1')
            gold = ImageChops.invert(ImageChops.logical_xor(g_, ImageChops.offset(h_.rotate(180), offx, 0)))
            for i, k in enumerate(kims):
                ox, oy = get_right_offsets(gold, k)
                sim = similarity_score(gold, ImageChops.offset(k, ox, 0))
                if sim > 0.985:
                    scores[i] += sim
        # logical ops basic e 5,6,7
        out = xor_and_common(a, b, c)
        if similarity_score(out, c) > 0.985:
            for i, k in enumerate(kims):
                _tmp = xor_and_common(g, h, k)
                sim = similarity_score(_tmp, k)
                if sim > 0.98:
                    scores[i] += sim + black_ratio_sim(_tmp, k)*.5
        out = xor_and_common(a, d, g)
        if similarity_score(out, g) > 0.985:
            for i, k in enumerate(kims):
                _tmp = xor_and_common(c, f, k)
                sim = similarity_score(_tmp, k)
                if sim > 0.98:
                    scores[i] += sim + black_ratio_sim(_tmp, k)*.5

        # intersection e-10
        if similarity_score(ImageChops.add(a, b), c) > 0.985 and similarity_score(ImageChops.add(d,e), f) > 0.985:
            gold = ImageChops.add(g, h)
            for i, k in enumerate(kims):
                sim = similarity_score(gold, k)
                if sim > 0.985:
                    scores[i] += sim
        if similarity_score(ImageChops.add(a, d), g) > 0.985 and similarity_score(ImageChops.add(b,e), h) > 0.985:
            gold = ImageChops.add(c, f)
            for i, k in enumerate(kims):
                sim = similarity_score(gold, k)
                if sim > 0.985:
                    scores[i] += sim

#        one pattern each, basic D 7,8,9,10
        detected = one_pattern_each(a,b,c,d,e,f,g,h)
        if detected is not None:
            for i, k in enumerate(kims):
                for j, o in enumerate(detected):
                    sim = similarity_score(o, k)
                    bk = black_ratio_sim(o, k)
                    # print((j, i), similarity_score(o, k), black_ratio_sim(o, k))
                    if (sim + bk) / 2 > 0.98:
                        scores[i] -= (sim + bk)/2
                        break
                    scores[i] -= (sim * 0.1 + bk * 0.5)*0.125

        # alternative way for basic d 10
        # take union
        u_abc = ImageChops.multiply(a, ImageChops.multiply(b, c))
        u_def = ImageChops.multiply(d, ImageChops.multiply(e, f))
        u_adg = ImageChops.multiply(a, ImageChops.multiply(d, g))
        u_beh = ImageChops.multiply(b, ImageChops.multiply(e, h))
        if similarity_score(u_abc, u_def) > 0.985 and similarity_score(u_adg, u_beh) > 0.985 and similarity_score(u_abc, u_adg) > 0.985 and similarity_score(u_def, u_beh) > 0.985:
            for i, k in enumerate(kims):
                u_ghk = ImageChops.multiply(g, ImageChops.multiply(h, k))
                sim = similarity_score(u_abc, u_ghk)
                if sim > 0.98:
                    scores[i] += sim

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

        # check object number pattern
        na, nb, nc, nd, ne, nf, ng, nh = len(objs_a), len(objs_b), len(objs_c), len(objs_d), len(objs_e), len(objs_f),\
        len(objs_g), len(objs_h)
        nks = [len(v) for v in kos.values()]
        if len(list(set(nks))) > 1:
            # row-wise
            if (na-nb) == (nd-ne) == (ng-nh) and (nb-nc) == (ne-nf):
                for i, nk in enumerate(nks):
                    if (nh-nk) == (ne-nf):
                        scores[i] += 0.5
            if (na+nb) == nc and (nd+ne) == nf:
                for i, nk in enumerate(nks):
                    if (ng + nh) == nk:
                        scores[i] += 0.5
            if (na+nb+nc) == (nd+ne+nf):
                for i, nk in enumerate(nks):
                    if (na+nb+nc) == (ng+nh+nk):
                        scores[i] += 0.5
            if (na - nb == nc) and (nd - ne == nf):
                for i, nk in enumerate(nks):
                    if (ng - nh == nk):
                        scores[i] += 0.5
        # # col-wise
        # if (na-nd) == (nb-ne) == (nc-nf) and (nd-ng) == (ne-nh):
        #     for i, nk in enumerate(nks):
        #         if (nf-nk) == (nd-ng):
        #             scores[i] += 1.
        # elif (na+nd) == ng and (nb+ne) == nh:
        #     for i, nk in enumerate(nks):
        #         if (nc+nf) == nk:
        #             scores[i] += 1.

        # e 9
        if na == nb == nc == 2:
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
                draw = ImageDraw.Draw(out)
                draw.point(objs_g[parts[0]], fill=(0))
                draw.point(objs_h[parts[1]], fill=(0))
                for i, k in enumerate(kims):
                    sim = similarity_score(out, k)
                    if sim > 0.98:
                        scores[i] += sim

        shapes_a = [Shape(o) for o in objs_a]
        shapes_b = [Shape(o) for o in objs_b]
        shapes_c = [Shape(o) for o in objs_c]
        shapes_d = [Shape(o) for o in objs_d]
        shapes_e = [Shape(o) for o in objs_e]
        shapes_f = [Shape(o) for o in objs_f]
        shapes_g = [Shape(o) for o in objs_g]
        shapes_h = [Shape(o) for o in objs_h]

        kshapes = {}
        for i, ko in kos.items():
            ks = [Shape(o) for o in ko]
            kshapes[i] = ks

        # shape counts
        sa = count_shapes([_.shape for _ in shapes_a])
        sb = count_shapes([_.shape for _ in shapes_b])
        sc = count_shapes([_.shape for _ in shapes_c])

        sg = count_shapes([_.shape for _ in shapes_g])
        sh = count_shapes([_.shape for _ in shapes_h])
        kns = {}
        for i, ko in kshapes.items():
            sk = count_shapes([_.shape for _ in ko])
            kns[i] = sk

        # shape arithmetic
        if len(sa) == len(sb) == len(sc) and len(sg) == len(sh):
            for i, k in kns.items():
                if len(k) == len(sg):
                    scores[i] += 0.5

        # count total shapes in a row
        _shapes_row = list(sa.keys()) + list(sb.keys()) + list(sc.keys())
        total_row = count_shapes(_shapes_row)
        _shapes_gh = list(sg.keys()) + list(sh.keys())
        for i, k in kns.items():
            _tmp = _shapes_gh + list(k.keys())
            _tmp_count = count_shapes(_tmp)
            if len(total_row) == len(_tmp_count):
                scores[i] += 0.5

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
        elif len(filled_a) == len(filled_b) == len(filled_c) and sum(filled_a) + sum(filled_b) == sum(filled_c):
            filled_d = [i.filled for i in shapes_d]
            filled_e = [i.filled for i in shapes_e]
            filled_f = [i.filled for i in shapes_f]
            if len(filled_d) == len(filled_e) == len(filled_f) and sum(filled_d) + sum(filled_e) == sum(filled_f):
                filled_g = [i.filled for i in shapes_g]
                filled_h = [i.filled for i in shapes_h]
                gold = min(len(filled_g), sum(filled_g) + sum(filled_h))
                for i, k in kshapes.items():
                    filled_k = [i.filled for i in k]
                    if len(filled_k) == len(filled_g) and gold == sum(filled_k):
                        scores[i] += 1.

        # challenge d-01
        diff_ab = ImageChops.invert(ImageChops.difference(b, a))
        diff_bc = ImageChops.invert(ImageChops.difference(c ,b))
        # apply b-a to d
        # apply c-b to e
        oute = ImageChops.multiply(diff_ab, d)
        outf = ImageChops.multiply(diff_bc, e)
        # apply b-a to g
        outh = ImageChops.multiply(diff_ab, g)
        if similarity_score(oute, e) > 0.985 and similarity_score(outf, f) > 0.985 and similarity_score(outh, h) > 0.985:
            gold = ImageChops.multiply(diff_bc, h)
            for i, k in enumerate(kims):
                sim = similarity_score(gold, k)
                if sim > 0.98:
                    scores[i] += sim

        # challenge e-02
        if similarity_score(ImageChops.multiply(a, b), c) > 0.93 and similarity_score(ImageChops.multiply(d, e),f) > 0.93:
            gold = ImageChops.multiply(g, h)
            for i, k in enumerate(kims):
                sim = similarity_score(gold, k)
                if sim > 0.93:
                    scores[i] += sim


        ANGS = [315, 270, 225, 180, 135, 90, 45, 0]
        ang = {}
        for i, t in enumerate(ANGS):
            out = a.rotate(t, fillcolor=(255))
            if 'b' not in ang and similarity_score(out, b) > 0.985:
                ang['b'] = t
            if 'c' not in ang and similarity_score(out, c) > 0.985:
                ang['c'] = t
        # challenge e-09
        if 'b' in ang and 'c' in ang and ang['b'] == ang['c']:
            for i, t in enumerate(ANGS):
                out = a.rotate(t, fillcolor=(255))
                if 'd' not in ang and similarity_score(out, d) > 0.985:
                    ang['d'] = t
                if 'e' not in ang and similarity_score(out, e) > 0.985:
                    ang['e'] = t
                if 'f' not in ang and similarity_score(out, f) > 0.985:
                    ang['f'] = t

            if 'd' in ang and 'e' in ang and 'f' in ang and (ang['d'] + ang['e']) % 360 == ang['f']:
                for i, t in enumerate(ANGS):
                    out = a.rotate(t, fillcolor=(255))
                    if 'g' not in ang and similarity_score(out, g) > 0.985:
                        ang['g'] = t
                    if 'h' not in ang and similarity_score(out, h) > 0.985:
                        ang['h'] = t
                if 'g' in ang and 'h' in ang:
                    rot = (ang['g'] + ang['h']) % 360
                    gold = a.rotate(rot, fillcolor=(255))
                    for i, k in enumerate(kims):
                        sim = similarity_score(gold, k)
                        if sim > 0.98:
                            scores[i] += sim
        # challenge d04
        elif 'b' in ang and 'c' in ang and ang['c'] == (ang['b'] * 2) % 360:
            gold_ang = ang['b']
            if similarity_score(d.rotate(gold_ang, fillcolor=(255)), e) > 0.985 and\
                similarity_score(e.rotate(gold_ang, fillcolor=(255)), f) > 0.985 and\
                similarity_score(g.rotate(gold_ang, fillcolor=(255)), h) > 0.985:
                gold = h.rotate(gold_ang, fillcolor=(255))
                for i, k in enumerate(kims):
                    sim = similarity_score(gold, k)
                    if sim > 0.98:
                        scores[i] += sim


        # challenge e 12
        if na == nb == nc == nd == ne == nf == ng == nh == 1:
            thresh = 200
            fn = lambda x: 255 if x > thresh else 0
            bka = a.point(fn, mode='1').histogram()[0]
            bkb = b.point(fn, mode='1').histogram()[0]
            bkc = c.point(fn, mode='1').histogram()[0]
            bkd = d.point(fn, mode='1').histogram()[0]
            bke = e.point(fn, mode='1').histogram()[0]
            bkf = f.point(fn, mode='1').histogram()[0]
            bkg = g.point(fn, mode='1').histogram()[0]
            bkh = h.point(fn, mode='1').histogram()[0]
            tt = self.imsize[0]**2
            if (bkb-bka)*(bkc-bkb) > 0 and (bkf-bke)*(bke-bkd) > 0 and abs((bkb-bka)/tt - (bkc-bkb)/tt) <= 0.002 and abs((bkf-bke)/tt - (bke-bkd)/tt) <= 0.002:
                gold = (bkh - bkg)/tt
                for i, k in enumerate(kims):
                    bkk = k.point(fn, mode='1').histogram()[0]
                    if (bkk-bkh)*(bkh-bkg)> 0 and abs(gold - (bkk - bkh)/tt) <= 0.002:
                        scores[i] += 0.5

        # challenge D-10
        ANGS = [270, 235, 180, 135, 90]
        flag = 0
        for ang in ANGS:
            if similarity_score(b.rotate(ang, fillcolor=(255)), c) > 0.985:
                flag += 1
                continue
            if similarity_score(e.rotate(ang, fillcolor=(255)), f) > 0.985:
                flag += 1
                continue
        if flag == 2:
            if similarity_score(ImageChops.multiply(a, c), a) > 0.95 and similarity_score(ImageChops.multiply(d, f), d) > 0.95:
                for i in range(8):
                    kim = kims[i]
                    for ang in ANGS:
                        if similarity_score(h.rotate(ang, fillcolor=(255)), kim) > 0.985:
                            new_k = fill_objects(kos[i], kshapes[i], self.imsize)
                            sim = similarity_score(ImageChops.multiply(g, new_k), g)
                            if sim > 0.9:
                                scores[i] += sim
                            continue

        # challenge e-05 & e-06 edge number
        eas = [o.shape[0] for o in shapes_a]
        ebs = [o.shape[0] for o in shapes_b]
        ecs = [o.shape[0] for o in shapes_c]
        eds = [o.shape[0] for o in shapes_d]
        ees = [o.shape[0] for o in shapes_e]
        efs = [o.shape[0] for o in shapes_f]
        egs = [o.shape[0] for o in shapes_g]
        ehs = [o.shape[0] for o in shapes_h]

        if (sum(eas) + sum(ecs)) // 2 == sum(ebs) and (sum(eds) + sum(efs)) // 2 == sum(ees):
            diff_edges = sum(ecs) - sum(eas)
            goal = sum(egs) + diff_edges
            for i, k in kshapes.items():
                eks = [o.shape[0] for o in k]
                if sum(eks) == goal:
                    scores[i] += 1.
        elif sum(eas) + sum(ebs) == sum(ecs) and sum(eds) + sum(ees) == sum(efs):
            goal = sum(egs) + sum(ehs)
            for i, k in kshapes.items():
                eks = [o.shape[0] for o in k]
                if sum(eks) == goal:
                    scores[i] += 1.

        # challenge d02
        # if the middle is symmetry horizontally, then a -> mirror -> c
        # if the middle is summetry vertically, then a -> flip -> c
        # ORIENT = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]
        if (similarity_score(b, ImageOps.mirror(b)) > 0.985 and similarity_score(ImageOps.mirror(a), c) > 0.985) or \
                (similarity_score(b, ImageOps.flip(b)) > 0.985 and similarity_score(ImageOps.flip(a), c) > 0.985):
            if (similarity_score(e, ImageOps.mirror(e)) > 0.985 and similarity_score(ImageOps.mirror(d), f) > 0.985) or \
                    (similarity_score(e, ImageOps.flip(e)) > 0.985 and similarity_score(ImageOps.flip(d), f) > 0.985):
                if similarity_score(h, ImageOps.flip(h)) > 0.985:
                    gold = ImageOps.flip(g)
                    for i, k in enumerate(kims):
                        sim = similarity_score(gold, k)
                        if sim > 0.985:
                            scores[i] += sim
                elif similarity_score(h, ImageOps.mirror(h)) > 0.985:
                    gold = ImageOps.mirror(g)
                    for i, k in enumerate(kims):
                        sim = similarity_score(gold, k)
                        if sim > 0.985:
                            scores[i] += sim


        ################ return highest score
        # if max(scores) > 0:
        return scores.index(max(scores)) + 1
        # return -1

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
    problem_set = ProblemSet("Challenge Problems D")

    for i, problem in enumerate(problem_set.problems[3:4]):
        print(vv.get_answer(problem), end=' ')


