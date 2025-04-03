import math
import re
from typing import Dict, Any
from loguru import logger

from ..agent.mcp import Tool

class CalculatorTool(Tool):
    """计算器工具，用于执行数学计算"""
    
    def __init__(self):
        """初始化计算器工具"""
        super().__init__(
            name="calculator",
            description="执行数学计算，支持基本的数学运算符和函数",
            parameters={
                "expression": {
                    "type": "string",
                    "description": "要计算的数学表达式，例如 '2 + 2' 或 'sin(0.5) * 5'"
                }
            }
        )
        
    async def execute(self, expression: str) -> Dict[str, Any]:
        """执行数学计算
        
        Args:
            expression: 要计算的数学表达式
            
        Returns:
            Dict: 计算结果
        """
        logger.info(f"执行计算表达式: {expression}")
        
        # 安全检查：去除所有非数学表达式内容
        # 允许数字、小数点、运算符、括号和部分函数名
        sanitized = re.sub(r'[^0-9.+\-*/().sinexpsqrtalogcp ]', '', expression)
        
        try:
            # 定义安全的数学函数
            safe_dict = {
                'sin': math.sin,
                'cos': math.cos,
                'tan': math.tan,
                'asin': math.asin,
                'acos': math.acos,
                'atan': math.atan,
                'sqrt': math.sqrt,
                'log': math.log,
                'log10': math.log10,
                'exp': math.exp,
                'pi': math.pi,
                'e': math.e,
                'abs': abs,
                'pow': pow,
                'round': round
            }
            
            # 使用安全环境执行表达式
            result = eval(sanitized, {"__builtins__": {}}, safe_dict)
            
            return {
                "result": result,
                "expression": expression
            }
        except Exception as e:
            logger.error(f"计算表达式错误: {str(e)}")
            return {
                "error": f"计算错误: {str(e)}",
                "expression": expression
            } 