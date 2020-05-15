""" Send paraphrased paragraphs/sentences back to original .csv sructure """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import pandas as pd 

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input_file", "", ".txt file of paraphrased paragraphs")
flags.DEFINE_string(
    "orig_csv", "", "original .csv that was paraphrased")
flags.DEFINE_string(
    "output_file", "", "output .csv with paraphrases")

def main(_):
  with open(FLAGS.input_file) as inf:
    para_list = inf.readlines()

  orig = pd.read_csv(FLAGS.orig_csv)
  data = orig.copy()
  data.iloc[:,1] = para_list   # replace values of second column with paraphrased list
  data = data.drop_duplicates(subset=data.iloc[:,1].name)   # remove duplicate paraphrases

  # create paraphrase .csv and append new paraphrased data to the end
  with open(FLAGS.output_file, 'a') as ouf:
    data.to_csv(ouf, mode='a', header=ouf.tell()==0, index=False)

if __name__ == '__main__':
  app.run(main)
