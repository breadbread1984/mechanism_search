#!/usr/bin/python3

from absl import flags, app
from mechanism_finder import MechanismFinder
import cv2

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_pdf', default = None, help = 'path to input pdf')

def main(unused_argv):
  mf = MechanismFinder()
  results = mf.process_pdf(FLAGS.input_pdf)
  for result in results:
    print(f"figure #{result['figure_num']} on page {result['page_num']} at position {result['position']} is a mechanism diagram")
    cv2.imshow('', result['image'])
    cv2.waitKey()

if __name__ == "__main__":
  add_options()
  app.run(main)
