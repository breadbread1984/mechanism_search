#!/usr/bin/python3

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from prompt import Prompt
from messages import *
from models import *
import configs

class MechanismFinder(object):
  def __init__(self, ):
    self.model = Qwen25VL7B_dashscope(api_key = configs.dashscope_api_key)
    class Output(BaseModel):
      has_mechanism: bool = Field(..., description = "whether current paper snippet contains a drug mechaism diagram.")
      figure: Optional[str] = Field(None, description = "figure number in format such as #1, #4 and so on (if has_mechanism=true)")
    self.parser = JsonOutputParser(pydantic_object = Output)
    self.instruction = self.parser.get_format_instructions()
    self.instruction = self.instruction.replace('{', '{{')
    self.instruction = self.instruction.replace('}', '}}')
  def get_prompt(self, image):
    return Prompt(messages = [
      HumanMessage(content = f"""# Instruction

Please determine whether the current paper snippet contains a drug mechanism diagram.

# Output format

{self.instruction}""", image = image)
    ])
  def process(self, image):
    prompt = self.get_prompt(image)
    output = self.model.inference(prompt)
    results = self.parser.parse(output)
    return results
