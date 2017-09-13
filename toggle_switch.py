#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 03:31:33 2017

@author: max
"""

import serial

arduino = serial.Serial('/dev/cu.usbserial-AL00UA1L', 9600, timeout=.1)
arduino.write('on'.encode())