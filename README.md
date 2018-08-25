### Chinese text generator as well as crnn based ocr detection system
====================================================================

> # Generate Chinese Word image

- check parameters
```shell
vim conf/generate.yml
```
```yaml
n_bg: 1
n_font: 1
ch_h: 32
n_pic: 20
sample_size: 10
min_len: 4
max_len: 6
bg_dir: data/bgs
font_dir: data/chn_fonts
target_dir: data/ocr_dataset_train_20_10_val
chn_map_path: data/chn_corpus/word_freq.json
```
put your font files eg xyz.ttf to [font dir] attribute
put your background files eg xyz.jpg to [bg dir] attribute

- example generated file
![alt text][example_image]
[example_image]: example.jpg "example image"

> train with pytorch crnn model

- edit training configuration
```shell
vim conf/train.yml
```

- start the training
```
python train.py
```
- example output
TODO

> infer with trained model

- edit infer configuration
```shell
vim conf/infer.yml
```

- start infering
```
python infer.py
```

- example output
```shell 
INFO:root:raw output --------的-----个------他----人人---------是 real output 的个他人是
```
