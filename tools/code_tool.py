import sys
import io
import traceback
import asyncio
import textwrap
import builtins
import re
from typing import Dict, Any, Optional
from loguru import logger

from ..agent.mcp import Tool

class CodeTool(Tool):
    """代码工具，用于执行和生成代码"""
    
    def __init__(self, timeout: int = 10, max_output_length: int = 2000, enable_exec: bool = True):
        """初始化代码工具
        
        Args:
            timeout: 代码执行超时时间(秒)
            max_output_length: 最大输出长度
            enable_exec: 是否允许执行代码
        """
        super().__init__(
            name="code",
            description="执行或生成Python代码",
            parameters={
                "code": {
                    "type": "string",
                    "description": "要执行的Python代码"
                },
                "mode": {
                    "type": "string",
                    "description": "执行模式: execute(执行代码) 或 generate(生成代码)",
                    "default": "execute"
                },
                "description": {
                    "type": "string",
                    "description": "代码的功能描述，用于生成代码时",
                    "default": ""
                }
            }
        )
        self.timeout = timeout
        self.max_output_length = max_output_length
        self.enable_exec = enable_exec
        
    async def execute(self, code: str, mode: str = "execute", description: str = "") -> Dict[str, Any]:
        """执行代码工具
        
        Args:
            code: 要执行的Python代码
            mode: 执行模式: execute(执行代码) 或 generate(生成代码)
            description: 代码的功能描述，用于生成代码时
            
        Returns:
            Dict: 执行结果
        """
        if mode == "generate":
            return await self._generate_code(description)
        else:
            if not self.enable_exec:
                return {
                    "code": code,
                    "error": "代码执行功能已禁用，只能生成代码",
                    "result": None,
                    "stdout": "",
                    "stderr": ""
                }
            return await self._execute_code(code)
    
    async def _execute_code(self, code: str) -> Dict[str, Any]:
        """执行Python代码
        
        Args:
            code: 要执行的Python代码
            
        Returns:
            Dict: 执行结果
        """
        logger.info(f"执行代码: {code[:100]}...")
        
        # 预处理代码，检查是否包含input函数
        code = self._preprocess_code(code)
        
        # 创建输出缓冲区
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        stdin_simulation = io.StringIO("50\n模拟用户输入\n" * 10)  # 预填充一些模拟输入
        
        # 保存原始stdout/stderr/stdin
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        original_stdin = sys.stdin
        
        # 重定向输出和输入到缓冲区
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        sys.stdin = stdin_simulation
        
        result = None
        error = None
        
        try:
            # 使用asyncio.wait_for实现超时机制
            code_obj = compile(code, "<string>", "exec")
            
            # 创建安全的执行环境
            safe_globals = self._create_safe_globals()
            
            # 创建局部作用域
            local_vars = {}
            
            # 执行代码，带超时控制
            await asyncio.wait_for(self._run_code(code_obj, safe_globals, local_vars), timeout=self.timeout)
            
            # 获取可能的返回值
            if "result" in local_vars:
                result = local_vars["result"]
                
        except asyncio.TimeoutError:
            error = f"代码执行超时 (>{self.timeout}秒)"
            logger.warning(f"代码执行超时: {code[:50]}...")
        except Exception as e:
            error = f"执行出错: {str(e)}\n{traceback.format_exc()}"
            logger.error(f"代码执行错误: {e}")
        finally:
            # 恢复原始stdout/stderr/stdin
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            sys.stdin = original_stdin
        
        # 获取输出
        stdout = stdout_capture.getvalue()
        stderr = stderr_capture.getvalue()
        
        # 截断过长的输出
        if len(stdout) > self.max_output_length:
            stdout = stdout[:self.max_output_length] + "... (输出已截断)"
        
        if len(stderr) > self.max_output_length:
            stderr = stderr[:self.max_output_length] + "... (输出已截断)"
        
        return {
            "result": result,
            "stdout": stdout,
            "stderr": stderr,
            "error": error
        }
    
    def _preprocess_code(self, code: str) -> str:
        """预处理代码，替换input函数等交互操作"""
        # 替换input函数调用为模拟值
        input_pattern = re.compile(r'input\s*\(\s*["\']([^"\']*)["\']?\s*\)')
        
        def input_replacer(match):
            prompt = match.group(1) if match.group(1) else ""
            # 根据提示内容决定返回什么模拟值
            if "猜测" in prompt or "数字" in prompt:
                return "'50'"  # 猜数字游戏可以返回中间值50
            return "'模拟用户输入'"
        
        return input_pattern.sub(input_replacer, code)
        
    def _create_safe_globals(self) -> Dict:
        """创建安全的执行环境，预导入常用模块"""
        safe_modules = {
            # 标准库
            "random": __import__("random"),
            "math": __import__("math"),
            "datetime": __import__("datetime"),
            "time": __import__("time"),
            "json": __import__("json"),
            "re": __import__("re"),
            "os": __import__("os"),
            "sys": __import__("sys"),
            "io": __import__("io"),
            "collections": __import__("collections"),
            "itertools": __import__("itertools"),
        }
        
        # 创建安全的内置函数
        safe_builtins = dict(__builtins__)
        
        # 模拟input函数
        def safe_input(prompt=""):
            print(prompt, end="")
            return "模拟用户输入"
            
        safe_builtins["input"] = safe_input
        
        # 合并为全局环境
        globals_dict = {"__builtins__": safe_builtins}
        globals_dict.update(safe_modules)
        
        return globals_dict
    
    async def _run_code(self, code_obj, globals_dict, local_vars):
        """在事件循环中执行代码对象"""
        # 这里可以使用exec直接执行
        # 但要注意安全性，可以添加更多限制
        exec(code_obj, globals_dict, local_vars)
        
        # 支持await异步代码
        if "main" in local_vars and asyncio.iscoroutinefunction(local_vars["main"]):
            await local_vars["main"]()
    
    async def _generate_code(self, description: str) -> Dict[str, Any]:
        """生成Python代码
        
        Args:
            description: 代码的功能描述
            
        Returns:
            Dict: 生成的代码
        """
        logger.info(f"生成代码: {description}")
        
        if not description:
            return {
                "code": "",
                "error": "请提供代码功能描述"
            }
        
        # 实际上代码生成应该由上层的MCPAgent处理
        # 这里只提供一个简单的示例框架
        example_code = textwrap.dedent(f'''
        # {description}
        
        def main():
            """实现{description}的主函数"""
            # TODO: 实现功能
            result = "这里是结果"
            return result
            
        # 如果直接运行脚本，执行main函数
        if __name__ == "__main__":
            result = main()
            print(result)
        ''')
        
        return {
            "code": example_code,
            "message": "这是代码示例框架，需要由Gemini完成具体实现"
        } 