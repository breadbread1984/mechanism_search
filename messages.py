#!/usr/bin/python3

from abc import ABC, abstractmethod
import base64
import numpy as np
import cv2

class Message(ABC):
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
  def to_json(self,):
    raise NotImplementedError

class SystemMessage(Message):
  def __init__(self, content):
    assert type(content) is str
    self.content = content
  def to_json(self,):
    return {'role': 'system', 'content': self.content}

class HumanMessage(Message):
  def __init__(self, content, image = None):
    assert type(content) is str
    assert type(image) is np.ndarray
    self.content = content
    self.image = image
  def to_json(self,):
    content = list()
    content.append({'type': 'text', 'text': self.content})
    if self.image is not None:
      content.append({'type': 'image_url', "image_url": {'url': self.encode_img(self.image)}})
    return {'role': 'user', 'content': content}

