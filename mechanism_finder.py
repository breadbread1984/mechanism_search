#!/usr/bin/python3

from typing import Optional, Literal, Tuple
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
import numpy as np
import fitz # pymupdf
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
      position: Optional[Tuple[int,int,int,int]] = Field(None, description = "position (top left x1,y1, bottom right x2,y2) of the figure or subfigure in sequence x1,y1,x2,y2")
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
  def process_image(self, image):
    prompt = self.get_prompt(image)
    output = self.model.inference(prompt)
    results = self.parser.parse(output)
    return results
  def process_pdf(self, pdf_path):
    pdf = fitz.open(pdf_path)
    pics = list()
    for idx, page in enumerate(pdf):
      pix = page.get_pixmap(dpi=100)
      img = np.frombuffer(pix.samples, dtype = np.uint8).reshape(pix.height, pix.width, -1)
      if pix.n == 1: img = img[:,:,0]
      elif pix.n == 3: img = img.reshape(pix.height, pix.width, 3)
      elif pix.n == 4: img = img.reshape(pix.height, pix.width, 4)
      results = self.process_image(img)
      if results['has_mechanism'] == True:
        x1,y1,x2,y2 = results['position']
        pics.append({
          'page_num': idx + 1,
          'figure_num': results['figure'] + '' if results['subfigure'] is None else results['subfigure'],
          'position': results['position'],
          'image': img[y1:y2,x1:x2],
        })
    return pics
