#!/usr/bin/python3

from abc import ABC, abstractmethod
from os import environ
import base64
import numpy as np
import cv2
from prompt import Prompt

class VLM(ABC):
  @abstractmethod
  def inference(self, prompt: Prompt):
    raise NotImplementedError

class Qwen25VL7B_dashscope(VLM):
  def __init__(self, configs):
    from openai import OpenAI
    self.client = OpenAI(
      api_key = configs.dashscope_api_key,
      base_url = configs.dashscope_host
    )
  def inference(self, prompt: Prompt):
    messages = prompt.to_json()
    response = self.client.chat.completions.create(
      model = 'qwen3-vl-plus',
      messages = messages,
    )
    return response.choices[0].message.content

class PPOCRVL_vllm(VLM):
  def __init__(self, configs):
    from openai import OpenAI
    self.client = OpenAI(
      api_key = configs.vllm_api_key,
      base_url = configs.vllm_host,
    )
  def inference(self, prompt: Prompt):
    messages = prompt.to_json()
    response = self.client.chat.completions.create(
      model = 'PaddlePaddle/PaddleOCR-VL',
      messages = messages,
    )

