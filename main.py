#!/usr/bin/python3

from absl import flags, app
import numpy as np
import fitz # pymupdf
from mechanism_finder import MechanismFinder

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_pdf', default = None, help = 'path to input pdf')

def main(unused_argv):
  mf = MechanismFinder()
  pdf = fitz.open(FLAGS.input_pdf)
  for idx, page in enumerate(pdf):
    pix = page.get_pixmap(dpi=100)
    img = np.frombuffer(pix.samples, dtype = np.uint8).reshape(pix.height, pix.width, -1)
    if pix.n == 1: img = img[:,:,0]
    elif pix.n == 3: img = img.reshape(pix.height, pix.width, 3)
    elif pix.n == 4: img = img.reshape(pix.height, pix.width, 4)
    results = mf.process(img)
    if results['has_mechanism'] == True:
      print(f"figure #{results['figure']} on page {idx + 1} is mechanism diagram")

if __name__ == "__main__":
  add_options()
  app.run(main)
