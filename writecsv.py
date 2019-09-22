# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 22:06:43 2018
@author: User
"""

import csv

datas = [['俄罗斯', '1707'],
         ['加拿大', 997],
         ['中国', 960]]

with open('D:/科研/体检/pytorch-CycleGAN-and-pix2pix/country.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in datas:
        writer.writerow(row)
