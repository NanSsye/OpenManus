import time
import datetime
import re
from typing import Dict, Any
from loguru import logger

from ..agent.mcp import Tool

class DateTimeTool(Tool):
    """日期时间工具，用于获取当前日期时间或执行日期计算"""
    
    def __init__(self):
        """初始化日期时间工具"""
        super().__init__(
            name="datetime",
            description="获取当前日期时间信息或执行日期计算",
            parameters={
                "format": {
                    "type": "string",
                    "description": "日期时间格式化字符串，例如 '%Y-%m-%d %H:%M:%S'，不提供则使用默认格式",
                    "default": "%Y-%m-%d %H:%M:%S"
                },
                "timezone": {
                    "type": "string",
                    "description": "时区信息，例如 'Asia/Shanghai'，不提供则使用系统时区",
                    "default": "local"
                },
                "operation": {
                    "type": "string",
                    "description": "日期计算操作，例如 '+1d'（加1天）, '-2h'（减2小时）等，不提供则返回当前时间",
                    "default": ""
                }
            }
        )
        
    async def execute(self, format: str = "%Y-%m-%d %H:%M:%S", 
                    timezone: str = "local", 
                    operation: str = "") -> Dict[str, Any]:
        """执行日期时间操作
        
        Args:
            format: 日期时间格式化字符串
            timezone: 时区信息
            operation: 日期计算操作 (e.g., '+1d', '-2h', '+6 days')
            
        Returns:
            Dict: 操作结果
        """
        logger.info(f"执行日期时间操作: format={format}, timezone={timezone}, operation={operation}")
        
        # 注意: 时区处理在此简化，实际可能需要pytz库来精确处理
        now = datetime.datetime.now()
        
        # 处理日期计算操作
        if operation:
            operation = operation.strip() # Remove leading/trailing whitespace
            try:
                # 使用更灵活的正则匹配操作: 允许空格，识别多种单位
                match = re.match(r'([+\-])\s*(\d+)\s*(days?|d|hours?|h|minutes?|mins?|m|seconds?|secs?|s)\b', 
                                 operation, re.IGNORECASE)
                if match:
                    op, value_str, unit_str = match.groups()
                    value = int(value_str)
                    unit_str = unit_str.lower()
                    
                    if op == '-':
                        value = -value
                        
                    delta = None
                    if unit_str.startswith('d'):
                        delta = datetime.timedelta(days=value)
                    elif unit_str.startswith('h'):
                        delta = datetime.timedelta(hours=value)
                    elif unit_str.startswith('m'):
                        delta = datetime.timedelta(minutes=value)
                    elif unit_str.startswith('s'):
                        delta = datetime.timedelta(seconds=value)
                        
                    if delta:
                        now += delta
                    else:
                        # Should not happen if regex is correct, but handle defensively
                        return {"error": f"无法识别日期单位: {unit_str} in {operation}"}
                else:
                    logger.warning(f"无法解析日期计算操作: {operation}")
                    return {"error": f"无法解析日期计算操作: {operation}"}
            except ValueError as ve:
                 logger.error(f"日期计算值错误: {ve} in {operation}")
                 return {"error": f"日期计算值错误: {ve} in {operation}"}
            except Exception as e:
                logger.exception(f"日期计算时发生意外错误: {str(e)}")
                return {"error": f"日期计算错误: {str(e)}"}
        
        # 格式化日期时间
        try:
            formatted = now.strftime(format)
            
            return {
                "datetime": formatted,
                "timestamp": int(now.timestamp()), # Use timestamp of calculated 'now'
                "iso_format": now.isoformat()
            }
        except ValueError as ve:
            logger.error(f"日期格式化错误: 无效的格式 '{format}'? ({ve})")
            return {"error": f"日期格式化错误: 无效的格式 '{format}'?"}
        except Exception as e:
            logger.exception(f"日期格式化时发生意外错误: {str(e)}")
            return {"error": f"日期格式化错误: {str(e)}"} 