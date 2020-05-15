""" Append paraphrases to original .csv file """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import pandas as pd 

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input_file", "", ".csv file of all paraphrased paragraphs")
flags.DEFINE_string(
    "orig_csv", "", "original .csv that was paraphrased")


def main(_):

  orig = pd.read_csv(FLAGS.orig_csv)
  if 'Unnamed: 0' in list(pd.read_csv(FLAGS.input_file).columns):
    data = pd.read_csv(FLAGS.input_file, index_col='Unnamed: 0')
  else:
    data = pd.read_csv(FLAGS.input_file)
  print(data.shape)

  # remove duplicated paraphrases
  data = data.drop_duplicates(subset=data.iloc[:,1].name)
  print(data.shape)

  # combine original .csv with paraphrases
  para = pd.concat([orig, data], ignore_index=True)
  print(para.shape)

  # remove any duplicates to original sentences
  para = para.drop_duplicates(subset=para.iloc[:,1].name)
  print(para.shape)
  
  # create paraphrase .csv and append new paraphrased data to the end
  para.to_csv(FLAGS.input_file)

if __name__ == '__main__':
  app.run(main)
