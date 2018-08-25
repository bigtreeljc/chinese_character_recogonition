import yaml
from easydict import EasyDict as edict
import generate_ocr as go
import os

def main():
    conf_file = "conf/generator.yml"
    with open(conf_file, 'r') as f:
        args = edict(yaml.load(f))

    n_bg, n_font = args.n_bg, args.n_font
    ch_h = args.ch_h
    n_pic = args.n_pic
    sample_size = args.sample_size
    min_len = args.min_len
    max_len = args.max_len
    bg_dir = args.bg_dir
    font_dir = args.font_dir

    target_dir = args.target_dir
    os.makedirs(target_dir, exist_ok=True)
    chn_map_path = args.chn_map_path

    chn_set = go.chn_subset(chn_map_path)
    generator = go.chn_ocr_generator(chn_set, n_chars=sample_size)
    generator.generate(target_dir, bg_dir, font_dir, n_bg,
        n_font, n_pictures=n_pic, ch_h=ch_h, min_len=min_len, 
        max_len=max_len)
    generator.save_vocab(target_dir)

if __name__ == "__main__":
    main()
