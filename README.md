# RASSDL
<small> Tone mapping high dynamic range images based on region-adaptive self-supervised deep learning

Zhou, Fei, Guangsen Liao, Jiang Duan, Bozhi Liu, and Guoping Qiu. "Tone mapping high dynamic range images based on region-adaptive self-supervised deep learning." Signal Processing: Image Communication 102 (2022): 116595.

## environment
+ Python 3.8
+ PyTorch
## Data Download
We collected HDR images from four databases:

Fairchild, http://rit-mcsl.org/ fairchild//HDR.html

HDR-Eye, http://infoscience.epfl.ch/record/203873

Anyhere, http: //www.anyhere.com/gward/hdrenc/pages/originals.html

Stanford-HDRI, https://exhibits.stanford.edu/data/catalog/sz929jt3255
## Data Preprocessing
run dataset.py, prepare the train data list train.txt and test data list test.txt
## Train 
run main_train.py
## Test 
run main_test.py to reproduce the results on the paper
## Citation
@article{zhou2022tone,
  title={Tone mapping high dynamic range images based on region-adaptive self-supervised deep learning},
  author={Zhou, Fei and Liao, Guangsen and Duan, Jiang and Liu, Bozhi and Qiu, Guoping},
  journal={Signal Processing: Image Communication},
  volume={102},
  pages={116595},
  year={2022},
  publisher={Elsevier}
}
