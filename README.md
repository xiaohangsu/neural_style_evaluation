# Style Mixer Evaluation
===================
## Pre-requisite
===================
### 1. install [caffe](http://caffe.berkeleyvision.org/) in python3.5+ environment

The default installation of caffe will install Python Language Python 2.7. </br>

If you do not install **Python3.5 version** caffe successfully, you might not successfully run my Python script (Python3.5 version). If you install caffe Python2.7 accidentally But you still can modify Python script into Python 2.7 version.

### 2. follow examples and download the models and weights

Download the models, config files and weights below:</br>

* [Fine-tuning for style recognition](http://caffe.berkeleyvision.org/gathered/examples/finetune_flickr_style.html) and
* [Fine-tuning for Style Recognition](http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/02-fine-tuning.ipynb) 

The two examples is basicly the same. Those models and methodology are based on [Recognizing Style](http://sergeykarayev.com/files/1311.3715v3.pdf)




## Train
===============
The part is for style evaluation. Before you start evaluate style, you should train examples provided models. (scratch_models and imageNet models)

### 1. Download trainning Data using Script assemble\_data_python3.py
Usage: **python3 assemble\_data_python3.py -h** for details </br>

> -w workers to download images *default=your\_computer\_CPU\_core\_nums*
> 
> -s seeds random from set of images *useful when you want only train a part of your downloaded images due to time limited*

* put this file in caffe\_root/examples/finetune\_flickr\_style/ 
* Trainning Data download from Flickr, I provide a modified version script of downloading all those 85K images provided, or you can download just a smaller sets of images to process the trainning
*  **this script is important for tarinning because it will update the *train.txt* and *test.txt* after executed**

### 2.using finetune\_flickr_style.py
Usage: **python3 finetune\_flickr_style.py -h** for more details
>-i --iter iteration time to train models
>
>-b --batch BATCH each iteration time, run BATCH number images *default=64*
>
>-ei end-to-end train ImageNet models
>
>-es end-to-end train Scratch models
>
>-t start to test, run test.txt each line to run test

* before train and test, you should make sure your train.txt and test.txt is updated by assemble\_data\_python3.py or by update\_test.py
* Beware of batcth and iteration number. The total images number will be trainned is **batch * iteration**.

## Evaluation
====================================
### using update_test.py
Usage: **python3 update_test.py -with\_fileName**

> -with_fileName put images folder into *caffe\_root/data/flickr\_style/*
> 
> and this parameter should equal to the test images folder name

This script will update **caffe\_root/data/flickr\_style/test.txt**

### Content Evalution
-
#### using content_evalution.py
put content_evalution.py in *caffe\_root/examples/*
Before you run content_evaluation.py script, be sure you have updated your test.txt </br>
Usage: **python3 content_evalution.py**</br>
It will generate the content evalution

### Style Evalution
-
#### using style_evalution.py

put style_evalution.py in *caffe\_root/examples/*
Before you run style_evalution.py script, be sure you have updated your test.txt </br>
Usage: **python3 style_evalution.py**</br>
It will generate the style evalution

### Hint
you can use **>** and **tee** in your shell or terminal to generate log file for you.


## Notice

1.**train.txt test.txt** mention above are both in *caffe\_root/data/flickr_style/*</br>
2.All scirpts might have absolute paths inside, change them into your own paths.