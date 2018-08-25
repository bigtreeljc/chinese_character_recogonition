import json
import re
import os
import pickle
import sys
import codecs
import time
import random
import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import cv2
import numpy as np
from math import cos, sin, pi
import numpy.ma as ma
import random as rand
from typing import List, Set, Dict, Tuple
from itertools import product
import logging

# from torchloop.util import tl_logging
logging.basicConfig(level=logging.DEBUG)
# logger = tl_logging.tl_logger()

class picture_pool:
    def __init__(self, bg_dir):
        self._bg_dir = os.path.abspath(bg_dir)
        self._bg_files = os.listdir(bg_dir)

    def sample_bg_file(self, sample_size: int=1) -> List[str]:
        selected_files = rand.sample(self._bg_files, sample_size)
        #### convert into abs path
        selected_files = list(map(lambda x: os.path.join(
            self._bg_dir, x), selected_files))
        return selected_files

class font_pool:
    def __init__(self, font_dir, file_list=None):
        self._font_dir = os.path.abspath(font_dir)
        if not file_list:
            self._font_files = os.listdir(font_dir)
        else:
            self._font_files = file_list

    def sample_font_file(self, sample_size: int=1) -> List[str]:
        selected_files = rand.sample(self._font_files, sample_size)
        #### convert into abs path
        selected_files = list(map(lambda x: os.path.join(
            self._font_dir, x), selected_files))
        return selected_files

    def font_files(self):
        return self._font_files

def random_scale(x,y):
    ''' 对x随机scale,生成x-y之间的一个数'''
    gray_out = random.randint(x, y)
    return gray_out

def text_gen_gray(bg_gray: int, line):
    gray_flag = np.random.randint(2)
    if bg_gray < line:
        text_gray = random_scale(bg_gray + line, 255)
    elif bg_gray > (255 - line):
        text_gray = random_scale(0, bg_gray - line)
    else:
        text_gray = gray_flag*random_scale(0, bg_gray - line) + (1 - gray_flag)*random_scale(bg_gray+line, 255)
    return text_gray

def text_gen_colored(bg_color: Tuple, line) -> Tuple:
    txt_color = [0, 0, 0]
    # 3 channels
    assert len(bg_color) == 3

    for c_ind, bg_gray in enumerate(bg_color):
        txt_gray = text_gen_gray(bg_gray, line)
        txt_color[c_ind] = txt_gray

    return tuple(txt_color)

def rot(img, angel, shape, max_angel, bg_gray):
    size_o = [shape[1], shape[0]]

    size = (shape[1] + int(shape[0]*cos((float(max_angel )/180) * pi)),shape[0])

    interval = abs(int(sin((float(angel) /180) * 3.14)* shape[0]))

    pts1 = np.float32([[0,0], [0,size_o[1]], [size_o[0],0], [size_o[0], size_o[1]]])

    if(angel>0):
        pts2 = np.float32([[interval,0],[0,size[1]  ],[size[0],0  ],[size[0]-interval,size_o[1]]])
    else:
        pts2 = np.float32([[0,0],[interval,size[1]  ],[size[0]-interval,0  ],[size[0],size_o[1]]])

    M  = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,size,borderValue=bg_gray)

    return dst

# def rotate(img, degree, 
        
class font_drawer:
    def __init__(self, font_file: str, bg_file: str, ch_size: int=16, 
            color_margin: int=60):
        self.image_font_ = ImageFont.truetype(font_file, ch_size)
        logging.debug("font file {}".format(font_file))
        self.bg_file = bg_file
        self.bg = Image.open(bg_file)
        self.color_margin = color_margin
        self.char_size = ch_size

    def draw_sample_gray_bg(self, txt_to_draw: str, 
            if_show: bool=False) -> None:
        '''
            draw chn unicode character on a picture 
            and returns the array representation
        '''
        bg_gray: int = random.randint(0, 255)
        text_gray: int = text_gen_gray(bg_gray, self.color_margin)
        logging.debug("grey bg {} txt grey {}".format(
            bg_gray, text_gray))
        txt_w, txt_h = self.image_font_.getsize(txt_to_draw)

        logging.debug("txt {} size {},{}".format(
            txt_to_draw, txt_w, txt_h))
        '''
            add some padding to img
        '''
        padding_w = random.randint(5, 15)
        padding_h = random.randint(5, 8)
        img_w, img_h = padding_w * 2 + txt_w, padding_h * 2 + txt_h
        pos_w, pos_h = padding_w, padding_h
        '''
            create img with params above 
        '''
        img: Image = Image.new("L", (img_w, img_h), bg_gray)

        draw = ImageDraw.Draw(img)
        draw.text((pos_w, pos_h), txt_to_draw, text_gray, 
                font=self.image_font_)
        if if_show:
            img.show()

    def draw_sample_colored(self, txt_to_draw: str, 
            if_show: bool=False) -> None:
        bg_color =\
               random.randint(0, 255), random.randint(0, 255), \
               random.randint(0, 255)
        text_color: int = text_gen_colored(bg_color, self.color_margin)
        logging.debug("bg color {} txt color {}".format(
            bg_color, text_color))
        txt_w, txt_h = self.image_font_.getsize(txt_to_draw)

        logging.debug("txt {} size {},{}".format(
            txt_to_draw, txt_w, txt_h))
        '''
            add some padding to img
        '''
        padding_w = random.randint(5, 15)
        padding_h = random.randint(5, 8)
        img_w, img_h = padding_w * 2 + txt_w, padding_h * 2 + txt_h
        pos_w, pos_h = padding_w, padding_h
        '''
            create img with params above 
        '''
        img: Image = Image.new("RGB", (img_w, img_h), bg_color)

        draw = ImageDraw.Draw(img)
        draw.text((pos_w, pos_h), txt_to_draw, text_color, 
                font=self.image_font_)
        if if_show:
            img.show()

    def _mk_colors(self, txt_to_draw: str) -> Tuple:
        bg_color =\
               random.randint(0, 255), random.randint(0, 255), \
               random.randint(0, 255)
        text_color: int = text_gen_colored(bg_color, self.color_margin)
        logging.debug("bg color {} txt color {}".format(
            bg_color, text_color))
        return [bg_color, text_color]

    def _mk_size_pos(self, txt_to_draw: str) -> Tuple:
        txt_w, txt_h = self.image_font_.getsize(txt_to_draw)

        logging.debug("txt {} size {},{}".format(
            txt_to_draw, txt_w, txt_h))
        '''
            add some padding to img
        '''
        padding_w = random.randint(5, 15)
        padding_h = random.randint(5, 8)
        img_w, img_h = padding_w * 2 + txt_w, padding_h * 2 + txt_h 
        pos_w, pos_h = padding_w, padding_h
        return img_w, img_h, pos_w, pos_h

    def _draw_text_on_img(self, txt_to_draw: str, img: Image,
            pos_w: int, pos_h: int, text_color: Tuple) -> None:
        draw = ImageDraw.Draw(img)
        draw.text((pos_w, pos_h), txt_to_draw, text_color, 
                font=self.image_font_)

    def draw_sample_colored_bg(self, txt_to_draw: str, 
            if_show: bool=False, show_bg: bool=False,
            show_cropped=False) -> Image:
        if show_bg:
            logging.debug("showing background picture {}".format(
                self.bg_file))
            self.bg.show()

        colors = self._mk_colors(txt_to_draw)
        bg_color, txt_color = colors[0], colors[1]
        txt_w, txt_h, txt_pos_w, txt_pos_h = \
                self._mk_size_pos(txt_to_draw)

        '''
            select a bbox in the bg picture
        '''
        bg_w, bg_h = self.bg.size
        # assert txt_w < bg_w and txt_h < bg_h, "font too large for bg"
        if not (txt_w < bg_w and txt_h < bg_h):
            print("warning font too large for bg")
            print("bg h {} w {} txt h {} w {}".format(
                bg_h, bg_w, txt_h, txt_w))
            return None
        available_area = bg_w - txt_w, bg_h - txt_h
        selected_w, selected_h = random.randint(0, available_area[0]), \
                    random.randint(0, available_area[0])
        area = (selected_w, selected_h, 
                selected_w + txt_w, selected_h + txt_h)
        cropped_img = self.bg.crop(area)
        if show_cropped:
            logging.debug("showing cropped img")
            cropped_img.show()
        self._draw_text_on_img(txt_to_draw, cropped_img, 
                txt_pos_w, txt_pos_h, txt_color)
        '''
            some other augmentation include perspective transformation
            color augmentation
        '''

        if if_show:
            cropped_img.show()

        return cropped_img

    ####
    # add some rotation and color augmentation
    ####
    def obfascate_sample(self):
        pass

def is_chn_char(ch):
    '''
        ch needs to be a 1 length str in python3
        check if that char is a chn char
    '''
    return u'\u4e00' <= ch <= u'\u9fff'

def check_contain_chinese(check_str):
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False


chn_nonchars = " ，。‘”“`;；：？！（）《》、……——一\""

class corpus:
    def __init__(self, corpus_dir, file_count=2):
        self.dir = corpus_dir
        self.count = file_count
        self.chn_dep = chn_nonchars
        self._read_in_memory()

    def _read_in_memory(self):
        files = os.listdir(self.dir)
        selected = rand.choices(files, k=self.count)
        for ff in selected:
            self._handle_file(ff)

    def _handle_file(self, fname):
        fname = os.path.join(self.dir, fname)
        logging.debug("\nfilename {}".format(fname))
        re_dep = " |，|。|‘|“|”|`|;|：|？|！|（|）|《|》|、|……|\"|；|——"

        with open(fname, 'r') as f:
            for line in f:
                line = line.strip()
                logging.debug(line+'\n')
                lines = re.split(re_dep, line)
                lines = list(filter(lambda x: len(x) > 0, lines))
                logging.debug(lines)
                logging.debug("-----")


    def sample_text(self):
        pass

class chn_subset:
    def __init__(self, chn_map_path):
        self.path = chn_map_path
        with open(self.path, 'r') as f:
            self.word_feq_map = json.load(f)

    def sample_words(self, sample_size=10):
        n_words = len(self.word_feq_map.keys())
        logging.info("num of chn chars is {}".format(n_words))
        sorted_map = self._sort_keys()
        assert len(sorted_map) > sample_size, "sample size {} bigger than vocab {}".\
            format(sample_size, len(sorted_map))
        
        ret = []
        for char, feq in sorted_map:
            if not char in chn_nonchars:
                ret.append((char, feq))
                if len(ret) == sample_size:
                    break
        return ret

    def _sort_keys(self):
        mydict = self.word_feq_map
        sorted_set = sorted(mydict.items(), key=lambda k: k[1], reverse=True)
        return sorted_set

class chn_ocr_generator:
    def __init__(self, subset, n_chars=10):
        self.char_set = subset
        self.chars_feq = subset.sample_words(n_chars)
        self.chars = list(map(lambda x: x[0], self.chars_feq))
        self.feq = list(map(lambda x: x[1], self.chars_feq))
        assert len(self.chars) == n_chars 

    @property
    def labels(self):
        return self.chars 

    def n_labels(self):
        return len(self.chars)

    def generate(self, dir_t, bg_dir, font_dir, n_bg, n_font, 
            n_pictures=3000, ch_h = 64, min_len=3, max_len=15):
        for __ in range(n_pictures):
            length = random.randint(min_len, max_len)
            label = random.choices(self.chars, k=length)
            label = ''.join(label)
            filename = os.path.join(dir_t, '.'.join([label, 'jpg']))
            print("saved to file {}".format(filename))

            pic_pool_o = picture_pool(bg_dir)
            sampled_bg = pic_pool_o.sample_bg_file(n_bg)
                          
            font_pool_o = font_pool(font_dir)
            sampled_font = font_pool_o.sample_font_file(n_font)
            
            bg_file: str = sampled_bg[0]
            font_file: str = sampled_font[0]

            text_drawer_o = font_drawer(font_file, bg_file, ch_size=ch_h)

            txt_to_draw = label
            img = None
            while not img:
                img = text_drawer_o.draw_sample_colored_bg(txt_to_draw,
                    if_show=False, show_bg=False, show_cropped=False)
                txt_to_draw = txt_to_draw[:-1]
            img.save(filename)
            # cv2.imshow('window', np.array(img))
            # cv2.waitKey(0)
            
    def _generate_product(self, min_len, max_len):
        pass
        # all_combs = []
        # for length in range(min_len, max_len):
        #     combs = list(product(self.chars, repeat=length))
        #     combs = list(map(lambda x: ''.join(x), combs))
        #     # print(combs)
        #     all_combs += combs

    def save_vocab(self, t_dir):
        filename = os.path.join(t_dir, "vocab.json") 
        with open(filename, 'w') as f:
            json.dump(self.chars, f)
