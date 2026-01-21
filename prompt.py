#!/usr/bin/python3

from typing import List
from messages import *

class Prompt(object):
  def __init__(self, messages: List[Message]):
    self.messages = messages
  def to_json(self,):
    return [message.to_json() for message in self.messages]
