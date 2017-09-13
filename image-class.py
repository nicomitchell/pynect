#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 15:41:49 2017

@author: max
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Simple image classification with Inception.
Run image classification with your model.
This script is usually used with retrain.py found in this same
directory.
This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. You are required
to pass in the graph file and the txt file.
It outputs human readable strings of the top 5 predictions along with
their probabilities.
Change the --image_file argument to any jpg image to compute a
classification of that image.
Example usage:
python label_image.py --graph=retrained_graph.pb
  --labels=retrained_labels.txt
  --image=flower_photos/daisy/54377391_15648e8d18.jpg
NOTE: To learn to use this file and retrain.py, please see:
https://codelabs.developers.google.com/codelabs/tensorflow-for-poets
"""


import argparse
import sys

import tensorflow as tf


def load_image(filename):
  """Read in the image_data to be classified."""
  return tf.gfile.FastGFile(filename, 'rb').read()


def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]


def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def run_graph(image_data, labels, input_layer_name, output_layer_name,
              num_top_predictions):
  with tf.Session() as sess:
    # Feed the image_data as input to the graph.
    #   predictions will contain a two-dimensional array, where one
    #   dimension represents the input image count, and the other has
    #   predictions per class
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    predictions, = sess.run(softmax_tensor, {input_layer_name: image_data})

    # Sort to show labels in order of confidence
    top_k = predictions.argsort()[-num_top_predictions:][::-1]
    for node_id in top_k:
      human_string = labels[node_id]
      score = predictions[node_id]
      print('%s (score = %.5f)' % (human_string, score))

    return 0


  
image_file = "/Users/max/Documents/tensorflow-for-poets-2/tf_files/flower_photos/roses/2414954629_3708a1a04d.jpg"
labels_file = "/Users/max/Documents/tensorflow-for-poets-2/tf_files/retrained_labels.txt"
graph_file = "/Users/max/Documents/tensorflow-for-poets-2/tf_files/retrained_graph.pb"


# load image
image_data = load_image(image_file)

# load labels
labels = load_labels(labels_file)

# load graph, which is stored in the default session
load_graph(graph_file)

output_layer = 'final_result:0'
input_layer = 'DecodeJpeg/contents:0'
num_top_predictions = 5
run_graph(image_data, labels, input_layer, output_layer, num_top_predictions)