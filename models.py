#!/usr/bin/python3

from abc import ABC, abstractmethod
from os import environ
import base64
import numpy as np
import cv2
from prompt import Prompt

class VLM(ABC):
  def encode_img(self, image):
    if type(image) is str:
      # image's url is given
      return image
    elif type(image) is np.ndarray:
      success, encoded_image = cv2.imencode('.png', image)
      assert success, "failed to encode numpy to png image!"
      png_bytes = encoded_image.tobytes()
      png_b64 = base64.b64encode(png_bytes).decode('utf-8')
      return f"data:image/png;base64,{png_b64}"
    else:
      raise RuntimeError('image can only be given in url or np.ndarray format!')
  @abstractmethod
  def inference(self, prompt: Prompt):
    pass

class Qwen25VL7B_dashscope(VLM):
  def __init__(self, api_key):
    from openai import OpenAI
    self.client = OpenAI(
      api_key = api_key,
      base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
  def inference(self, prompt: Prompt):
    messages = prompt.to_json()
    response = self.client.chat.completions.create(
      model = 'qwen2.5-vl-7b-instruct',
      messages = messages,
    )
    return response.choices[0].message.content
