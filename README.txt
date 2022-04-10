Maxime Mahdavian
Artem Chernigel

COMP432 Project
--------------------
We used Python 3.8 for this project, Python 3.7 should work, but we haven't tested it

Packages used:
torch
torchvision
matplotlibt pyplot
timeit
argparse
tensorflow Keras
sklearn

-------------------------------------------
The code for both models check if a cuda device is available to run the code on. Cuda drivers are required to make
this functionality work.
-----------------------------------------

This project is comprised of two python scripts, one for each model.

ResNet9
--------------------------------------------------
To run the ResNet9 model, simply run resnet_main.py

It requires one command line argument to specify the learning rate like so:

usage: resnet_main.py [-h] [-lr LR]

optional arguments:
  -h, --help  show this help message and exit
  -lr LR      learning rate

VGGNet16
---------------------------------------------
To run vggnet16 model, run the the file VGGNet16.py

Jupyter notebook
---------------------------------------------
There is a jupyter notebook which is a combination of all the python scripts used for both models