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
from PIL import Image, ImageChops, ImageStat
import numpy
from collections import defaultdict
from itertools import product


class SemanticNet:
    def __init__(self):
        # ['shape', 'fill', 'size', 'angle', 'inside', 'alignment', 'above', 'overlaps', 'left-of', 'width', 'height']
        # [('shape', {'plus', 'heart', 'star', 'square', 'circle', 'triangle', 'rectangle', 'pentagon', 'right triangle', 'pac-man', 'diamond', 'octagon'}),
        # ('fill', {'no', 'right-half', 'top-half', 'bottom-half', 'left-half', 'yes'}),
        # ('size', {'large', 'medium', 'huge', 'very small', 'small', 'very large'}),
        # ('angle', {'225', '90', '45', '0', '270', '180', '315', '135'}),
        # ('alignment', {'bottom-right', 'top-right', 'top-left', 'bottom-left'}),
        # ('width', {'large', 'huge', 'small'}), # only to rectangular shape in problem C, may ignore for now
        # ('height', {'small', 'large', 'huge'})]
        # self.properties = set(['shape', 'size', 'fill', 'angle', 'alignment', 'width', 'height'])
        self.properties = set(['shape', 'size', 'fill', 'angle', 'alignment'])
        self.relations = set(['inside', 'above', 'overlaps', 'left-of'])
        self.reflection_mappings = dict(
            bottom="top",
            top="bottom",
            left="right",
            right="left"
        )

    # TODO
    # what if both a and b are inside c?
    # a   inside: c
    # b   inside: c
    # if all other attributes are identical, then cannot tell a and b
    # moreover, if a b in c, d e in f, then cannot tell which one inside which
    # but does it matter? as far as transformation detection?
    def renaming_by_relation(self, fig):
        """
        renaming an object by its relational attributes
        :param fig: a RavensFigure
        :return: {inside1: obj, inside0: obj}
        """
        naming = defaultdict(list)
        for name, object in fig.objects.items():
            for k, v in object.attributes.items():
                if k in self.relations:
                    parents = object.attributes[k]
                    level = len(parents.split(','))
                    key_str = k + str(level)
                    naming[key_str].append(object)
                    if level == 1:
                        # find the root node
                        parent = parents[0]
                        for o in fig.objects:
                            if o == parent:
                                key_str  = k + str(0)
                                naming[key_str].append(fig.objects[o])
                                break
        return naming


    def single_node(self, o1, o2):
        """
        Detect change between two objects
        :param o1:
        :param o2:
        :return:
        """
        transformation = {}
        for k in self.properties:
            if k in o1.attributes:
                if o1.attributes[k] != o2.attributes.get(k, ''):
                    # TODO separate functionality
                    if k == 'angle':
                        angle1 = int(o1.attributes[k])
                        angle2 = int(o2.attributes.get(k, 1))
                        if o1.attributes["shape"] == "pac-man" and angle1 + angle2 == 360:
                            transformation['reflection'] = True
                        elif o1.attributes["shape"] != "pac-man" and angle1 % 180 == (angle2 + 90) % 180:
                            transformation['reflection'] = True
                        elif (angle1 - angle2) % 5 == 0:
                            ans = str((abs(angle1 - angle2)))
                            if angle1 > angle2:
                                ans = '-' + ans
                            else:
                                ans = '+' + ans
                            transformation['rotation'] = ans
                    # seperate alignment change from reflection
                    elif k == 'alignment':
                        blocks1 = o1.attributes["alignment"].split("-")
                        v1 = blocks1[0]
                        h1 = blocks1[1]
                        blocks2 = o2.attributes.get("alignment", "").split("-")
                        v2 = blocks2[0]
                        h2 = blocks2[1]
                        if v1 == v2 and self.reflection_mappings[h1] == h2:
                            transformation['alignment'] = 'left-right'
                        if self.reflection_mappings[v1] == v2 and h1 == h2:
                            transformation['alignment'] = 'top-down'
                    else:
                        transformation[k] = o1.attributes[k] + '->' + o2.attributes.get(k, '')
        return transformation


    def represent_transformation(self, f1, f2):
        """
        compare two figures, get string representation of transformation
        :param f1:
        :param f2:
        :return:
        """
        # check number of objects
        num_obj1 = len(f1.objects)
        num_obj2 = len(f2.objects)
        if num_obj1 == num_obj2 == 1:
            o1 = list(f1.objects.values())[0]
            o2 = list(f2.objects.values())[0]
            return self.single_node(o1, o2)


        transformation = {}

        # check for deletion
        if num_obj1 > num_obj2:
            props1 = [self.get_property_set(o) for o in f1.objects.values()]
            props2 = [self.get_property_set(o) for o in f2.objects.values()]

            if len(props1) - len(props2) == num_obj1 - num_obj2:
                transformation['deleted'] = [i for i in props1 if i not in props2]
            else:
                # TODO need mapping
                pass

        # check for addition
        if num_obj1 < num_obj2:
            props1 = [self.get_property_set(o) for o in f1.objects.values()]
            props2 = [self.get_property_set(o) for o in f2.objects.values()]

            if len(props2) - len(props1) == num_obj2 - num_obj1:
                transformation['added'] = [i for i in props2 if i not in props1]
            else:
                # TODO need mapping
                pass

        intra_repr1 = self.renaming_by_relation(f1)
        intra_repr2 = self.renaming_by_relation(f2)

        # print(intra_repr1, intra_repr2)

        for new_name in intra_repr1:
            if new_name in intra_repr2:
                for o1, o2 in zip(intra_repr1[new_name], intra_repr2[new_name]):
                    single_trans = self.single_node(o1, o2)
                    # print(single_trans)
                    new_dict = {new_name+ '-' + k: v for k, v in single_trans.items()}
                    transformation.update(new_dict)
            # else:
            #     attr = self.get_property_set(intra_repr1[new_name][0])
            #
            #     if 'deleted' in transformation:
            #         transformation['deleted'].append(attr)
            #         # transformation['deleted'].append(new_name) # switch to properties?
            #     else:
            #         transformation['deleted'] = [attr]
            #         # transformation['deleted'] = [new_name]

        # print(transformation)
        return transformation


    def get_property_set(self, obj):
        prop = []
        for k, v in obj.attributes.items():
            if k in self.properties:
                prop.append(v)
        return prop


    def get_score(self, gold, pred):
        # unchanged case
        if not gold and not pred:
            return 1.
        counter = 0
        for k, v in gold.items():
            if k in pred and pred[k] == v:
                counter += 1
        return counter / (len(list(pred.values())) + 1e-8)


    def random_guess(self, size):
        """
        To deal with out of capacity problems, FOR FUN...
        :param size: number of keys
        :return:
        """
        guess = numpy.random.randint(0, size)
        # print(f'Guessed...{guess}')
        return guess


    # TODO support 3x3
    def compare_answers(self, problem):
        if problem.problemType == "3x3":
            return self.random_guess(9)

        row_repr = self.represent_transformation(problem.figures["A"], problem.figures["B"])
        # print('================================row=============================================')
        # print(row_repr)
        # print('------------------------')
        scores = [0 for _ in range(6)]

        for i, key in enumerate(["1", "2", "3", "4", "5", "6"]):
            candi_repr = self.represent_transformation(problem.figures["C"], problem.figures[key])
            # print(key, candi_repr)
            # if row_repr is not None and candi_repr == row_repr:
            #     return int(key)
            score = self.get_score(row_repr, candi_repr)
            # print(score)
            scores[i] = score

        # print('================================col=============================================')
        col_repr = self.represent_transformation(problem.figures["A"], problem.figures["C"])
        # print(col_repr)
        # print('------------------------')
        for i, key in enumerate(["1", "2", "3", "4", "5", "6"]):
            candi_repr = self.represent_transformation(problem.figures["B"], problem.figures[key])
            # print(key, candi_repr)
            # if col_repr is not None and candi_repr == col_repr:
            #     return int(key)
            score = self.get_score(col_repr, candi_repr)
            # print(score)
            scores[i] += score

        # print(scores)
        return numpy.argmax(numpy.array(scores)) + 1

############################### experiment with bundle theory ########################################
    def _experiment(self, f1, f2):
        # bundle theory
        # get flattened properties
        props1 = []
        for o in f1.objects.values():
            tmp = []
            for k, v in o.attributes.items():
                if k not in self.relations:
                    if k == 'angle':
                        tmp.extend([int(v), 360 - int(v), int(v) - 90])
                    elif k == 'alignment':
                        tmp.extend(v.split('-'))
                    else:
                        tmp.append(v)
            props1.append(tmp)

        props2 = []
        for o in f2.objects.values():
            tmp = []
            for k, v in o.attributes.items():
                if k not in self.relations:
                    if k == 'angle':
                        tmp.extend([int(v), 360 - int(v), int(v) - 90])
                    elif k == 'alignment':
                        tmp.extend(v.split('-'))
                    else:
                        tmp.append(v)
            props2.append(tmp)

        # print(props1, props2)
        return props1, props2

    def _compare_properties(self, a, b):
        pa, pb = self._experiment(a, b)
        unchanged, deleted, added = [], [], []
        for i, p1 in enumerate(pa):
            for j, p2 in enumerate(pb):
                if p1 == p2:
                    unchanged.append(p1)
                    pa.pop(i)
                    pb.pop(j)
        pa_all = [p for i in pa for p in i]
        pb_all = [p for i in pb for p in i]
        for p in pa_all:
            if p not in pb_all:
                deleted.append(p)

        for p in pb_all:
            if p not in pa_all:
                added.append(p)

        return {'same': unchanged, 'del': deleted, 'new': added}

    def _score(self, d1, d2):
        sam1 = [i for v in d1['same'] for i in v]
        sam2 = [i for v in d2['same'] for i in v]
        del1 = d1['del']
        del2 = d2['del']
        add1 = d1['new']
        add2 = d2['new']

        inter_same = [s for s in sam1 if s in sam2]
        # inter_del = [s for s in del1 if s in del2]
        # inter_add = [s for s in add1 if s in add2]
        inter_del = [s for s in del1 if s in del2 or isinstance(s, int) and abs(s) in del2]
        inter_add = [s for s in add1 if s in add2 or isinstance(s, int) and abs(s) in add2]

        score1 = (len(inter_same) + 1e-08) / (len(sam1 + sam2) + 1e-08)
        score2 = (len(inter_del) + 1e-08) / (len(del1 + del2) + 1e-08)
        score3 = (len(inter_add) + 1e-08) / (len(add1 + add2) + 1e-08)

        return score1 * 2 + score2 + score3

    def _compare_answers(self, problem):
        if problem.problemType == "3x3":
            return self.random_guess(9)
        a = problem.figures['A']
        b = problem.figures['B']
        c = problem.figures['C']

        scores = [0 for _ in range(6)]
        ab = self._compare_properties(a, b)
        # print(ab)
        for i, key in enumerate(['1', '2', '3', '4', '5', '6']):
            ck = self._compare_properties(c, problem.figures[key])
            # print(i+1, ck)
            # print(self._score(ab, ck))
            scores[i] = self._score(ab, ck)

        ac = self._compare_properties(a, c)
        # print(ac)
        for i, key in enumerate(['1', '2', '3', '4', '5', '6']):
            bk = self._compare_properties(b, problem.figures[key])
            # print(i+1, bk)
            # print(self._score(ab, ck))
            scores[i] += self._score(ac, bk)

        # print(scores)
        return numpy.argmax(numpy.array(scores)) + 1

######################################################################################################


class Visual:
    def __init__(self):
        # whole figure operations with priority, flip > rotation (challenge b6 both flip and rotation => flip)
        self.TRANS = [Image.NONE, Image.FLIP_TOP_BOTTOM, Image.FLIP_LEFT_RIGHT, Image.ROTATE_270, Image.ROTATE_180, Image.ROTATE_90]
        self.ANGS = [45, 135, 225, 315]

    def similarity_score(self, im1, im2):
        """
        Pixelwise diff percent between im1 and im2
        :param im1:
        :param im2:
        :return: 0. ~ 1.
        """
        disturb = [ImageChops.offset(im1, dx, dy) for dx, dy in product(range(-2, 3), range(-2, 3))]
        outs = [ImageChops.difference(d1, im2) for d1 in disturb]
        residual = min([ImageStat.Stat(out).sum[0] for out in outs])
        score = 1. - (residual / 255.) / (im1.size[0] * im1.size[1])
        return score

    def logical_ops(self, im1, im2):
        """
        return a-b, a|b, a+b
        """
        a = im1.convert(mode='1')
        b = im2.convert(mode='1')
        xor = ImageChops.logical_xor(a, b)
        _or = ImageChops.logical_or(a, b)
        _and = ImageChops.logical_and(a, b)
        return (xor, _or, _and)

    def macro_transformation(self, im1, im2, thred=0.98):
        """
        Indentity, Flip, Rotate on the whole figure
        :param im:
        :return:
        """
        for idx, t in enumerate(self.TRANS):
            res = im1.transpose(method=t)
            if self.similarity_score(res, im2) > thred:
                return idx + 1
        offset = len(self.TRANS)
        for idx, a in enumerate(self.ANGS):
            res = im1.rotate(a, fillcolor=(255))
            if self.similarity_score(res, im2) > thred:
                return idx + offset + 1
        return 0

    def generate_and_test(self, p):
        def _get_img(title):
            f = p.figures[title].visualFilename
            org = Image.open(f).convert('L')
            size = org.size[0]
            return org.resize((size//2, size//2), resample=Image.NEAREST)
        given = ['A', 'B', 'C']
        choices = ['1', '2', '3', '4', '5', '6']
        if p.problemType == '3x3':
            given.extend(['D', 'E', 'F', 'G', 'H'])
            choices.extend(['7', '8'])

        if p.problemType == '2x2':
            img_a = _get_img('A')
            img_b = _get_img('B')
            img_c = _get_img('C')
            scores = [0 for i in range(6)]
            macro_ab = self.macro_transformation(img_a, img_b)
            macro_ac = self.macro_transformation(img_a, img_c)
            offset = len(self.TRANS)
            if macro_ab:
                if macro_ab < offset:
                    trans = self.TRANS[macro_ab - 1]
                    gen = img_c.transpose(method=trans)
                else:
                    rot_ang = self.ANGS[macro_ab - offset -1]
                    gen = img_c.rotate(rot_ang, fillcolor=(255))
                for i, k in enumerate(choices):
                    candi = _get_img(k)
                    sim = self.similarity_score(gen, candi)
                    if sim > 0.97:
                        # gen.show()
                        # candi.show()
                        # print(sim)
                        return i+1
                    elif sim > 0.85:
                        scores[i] = sim

            if not macro_ab and macro_ac:
                if macro_ac < offset:
                    trans = self.TRANS[macro_ac - 1]
                    gen = img_b.transpose(method=trans)
                else:
                    rot_ang = self.ANGS[macro_ac - offset -1]
                    gen = img_b.rotate(rot_ang, fillcolor=(255))

                for i, k in enumerate(choices):
                    candi = _get_img(k)
                    sim = self.similarity_score(gen, candi)
                    if sim > 0.97:
                        # gen.show()
                        # candi.show()
                        return i+1
                    elif sim > 0.85:
                        scores[i] += sim

            ab_xor, ab_or, ab_and = self.logical_ops(img_a, img_b)
            tmp_c = img_c.convert(mode='1')
            gen_h = [ImageChops.logical_xor(tmp_c, ab_xor), ImageChops.logical_or(tmp_c, ab_or), ImageChops.logical_and(tmp_c, ab_and)]
            ac_xor, ac_or, ac_and = self.logical_ops(img_a, img_c)
            tmp_b = img_b.convert(mode='1')
            gen_v = [ImageChops.logical_xor(tmp_b, ac_xor), ImageChops.logical_or(tmp_b, ac_or), ImageChops.logical_and(tmp_b, ac_and)]
            for i, k in enumerate(choices):
                candi = _get_img(k).convert(mode='1')
                for x in gen_h + gen_v:
                    sim = self.similarity_score(x, candi)
                    if sim > 0.85:
                        scores[i] += sim

            if max(scores) > 0.9:
                return scores.index(max(scores)) + 1
            else:
                return -1


class Agent:
    # The default constructor for your Agent. Make sure to execute any
    # processing necessary before your Agent starts solving problems here.
    #
    # Do not add any variables to this signature; they will not be used by
    # main().
    def __init__(self):
        self.verbal_brain = SemanticNet()
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
        if not problem.hasVerbal:
            if problem.problemType == '2x2':
                # return self.verbal_brain.random_guess(7)
                return self.visual_brain.generate_and_test(problem)
            else:
                return self.verbal_brain.random_guess(9)

        ans = self.verbal_brain._compare_answers(problem)
        return ans


if __name__ == "__main__":
    # with Image.open("./Problems/Basic Problems B/Basic Problem B-01/1.png") as im:
    #     print(im.mode, im.size)
    from ProblemSet import ProblemSet
    vv = Visual()
    problem_set = ProblemSet("Challenge Problems B")

    for i, problem in enumerate(problem_set.problems):

        print(vv.generate_and_test(problem))
        # print(f'\nQ{i+1}')
        # res = sn.compare_answers(problem)
        # print(f'ANSWER for {i+1}: {res}')

