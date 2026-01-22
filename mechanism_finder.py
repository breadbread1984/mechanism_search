#!/usr/bin/python3

from typing import Optional, Literal
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
      has_mechanism: bool = Field(..., description = "whether current picture contains a drug mechaism diagram.")
      figure: Optional[int] = Field(None, description = "figure number in integer format (if has_mechanism=true)")
      subfigure: Optional[Literal['a','b','c','d','e','f','g','h','i','j','k','l','m','n']] = Field(None, description = "subfigure number if appliable (is has_mechanism=true)")
    self.parser = JsonOutputParser(pydantic_object = Output)
    self.instruction = self.parser.get_format_instructions()
    self.instruction = self.instruction.replace('{', '{{')
    self.instruction = self.instruction.replace('}', '}}')
  def get_prompt(self, image):
    return Prompt(messages = [
      SystemMessage(content = """任务：判断图片是否为"药物机理图（Drug Mechanism of Action Diagram）"

定义：展示药物如何作用的示意图，包含：
1. 药物分子结构/名称
2. 靶点蛋白（如kinase、receptor）
3. 箭头表示抑制/激活
4. 下游信号通路（PI3K/AKT、MAPK等）
5. 最终效应（凋亡、增殖抑制）

排除类型：
- 数据图（Western blot、IC50曲线、flow cytometry）
- 显微镜照片
- 纯柱状图/表格"""),
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
