import os
import re
import asyncio
import base64
import json
import time
import platform
import importlib.util
from typing import Dict, List, Any, Optional, Tuple, Union

from loguru import logger

from ..api_client import GeminiClient

class Tool:
    """工具基类"""
    def __init__(self, name: str, description: str, parameters: Dict = None):
        self.name = name
        self.description = description
        # Store parameters, ensuring defaults are handled if needed by the caller
        self.parameters = parameters or {}
        
    def to_dict(self) -> Dict:
        """将工具转换为API格式的字典 (修正 required 逻辑)"""
        # Identify required parameters: those that DO NOT have a 'default' key in their definition
        required_params = [
            name for name, details in self.parameters.items() 
            if 'default' not in details
        ]
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": required_params # Use the correctly identified list
                }
            }
        }
        
    async def execute(self, **kwargs) -> Dict:
        """执行工具
        
        Args:
            **kwargs: 工具参数
            
        Returns:
            Dict: 执行结果
        """
        raise NotImplementedError("工具子类必须实现execute方法")

class MCPAgent:
    """MCP代理，实现多步骤思考过程"""
    
    def __init__(self, client: GeminiClient, model: str, 
                 max_tokens: int = 8192, temperature: float = 0.7,
                 max_steps: int = 5, thinking_steps: int = 3,
                 force_thinking: bool = True,
                 thinking_prompt: str = "请深入思考这个问题，分析多个角度并考虑是否需要查询额外信息，然后提供具体的解决方案。思考要全面但不要在最终回答中展示思考过程。",
                 system_prompt: Optional[str] = None):
        """初始化MCP代理
        
        Args:
            client: 已初始化的 GeminiClient 实例
            model: 使用的模型名称
            max_tokens: 最大生成token数
            temperature: 温度参数
            max_steps: 最大执行步骤数
            thinking_steps: 思考步骤数
            force_thinking: 是否强制执行思考步骤
            thinking_prompt: 思考提示词
            system_prompt: 自定义系统提示词，如果为None则使用默认提示词
        """
        if not isinstance(client, GeminiClient):
             raise TypeError("client must be an instance of GeminiClient")
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_steps = max_steps
        self.thinking_steps = thinking_steps
        self.force_thinking = force_thinking
        self.thinking_prompt = thinking_prompt
        self.tools = {}
        self.thinking_history = []
        self.conversation_history = []
        self.system_prompt = system_prompt
        
    def register_tool(self, tool: Tool) -> None:
        """注册工具
        
        Args:
            tool: 工具实例
        """
        self.tools[tool.name] = tool
        logger.info(f"工具已注册: {tool.name}")
        
    def register_tools(self, tools: List[Tool]) -> None:
        """批量注册工具
        
        Args:
            tools: 工具实例列表
        """
        for tool in tools:
            self.register_tool(tool)
            
    def get_tool_definitions(self) -> List[Dict]:
        """获取所有工具定义
        
        Returns:
            List[Dict]: 工具定义列表
        """
        return [tool.to_dict() for tool in self.tools.values()]
        
    async def execute_tool(self, tool_name: str, **kwargs) -> Dict:
        """执行工具
        
        Args:
            tool_name: 工具名称
            **kwargs: 工具参数
            
        Returns:
            Dict: 执行结果
        """
        if tool_name not in self.tools:
            return {"error": f"未找到工具: {tool_name}"}
        
        tool = self.tools[tool_name]
        
        try:
            start_time = time.time()
            logger.debug(f"Executing tool '{tool_name}' with args: {kwargs}")
            result = await tool.execute(**kwargs)
            elapsed = time.time() - start_time
            logger.info(f"工具 {tool_name} 执行完成，耗时 {elapsed:.2f}s")
            return result
        except TypeError as te:
             logger.error(f"工具 '{tool_name}' 参数错误: {te}. Provided args: {kwargs}")
             return {"error": f"工具 '{tool_name}' 参数错误: {te}"}
        except Exception as e:
            logger.exception(f"工具 {tool_name} 执行异常")
            return {"error": f"工具执行异常: {str(e)}"}
    
    def _extract_text_from_gemini_response(self, response: Dict) -> str:
        """从Gemini API响应中安全地提取文本内容"""
        try:
            # Standard non-streaming or chunk structure
            candidates = response.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if parts and "text" in parts[0]:
                    return parts[0]["text"]
            # Handle potential streaming chunk format if different (adjust if needed)
            # Placeholder - refine based on actual stream chunk structure if necessary
            if "text" in response: # Direct text in chunk?
                return response["text"]
                
        except (KeyError, IndexError, TypeError) as e:
            logger.warning(f"无法从Gemini响应中提取文本: {e}, 响应: {response}")
        return "" # Return empty string if text cannot be extracted
        
    async def run(self, instruction: str, history: List[Dict[str, str]] = None) -> Dict:
        """执行代理
        
        Args:
            instruction: 用户指令
            history: 历史对话记录，可选
            
        Returns:
            Dict: 执行结果
        """
        # 记录MCP模式设置
        logger.info(f"MCPAgent运行模式: thinking_steps={self.thinking_steps}, force_thinking={self.force_thinking}")
        
        # 快速路径：检查是否已禁用MCP（thinking_steps=0）
        if self.thinking_steps <= 0:
            logger.info("MCP模式已禁用 (thinking_steps=0)，将直接处理请求而不进行多步思考")
            return await self._direct_response(instruction, history)
        
        # 如果提供了历史记录，使用历史记录初始化会话
        # 否则创建新的会话历史
        if history:
            self.conversation_history = history.copy()
            logger.debug(f"使用提供的历史记录，共 {len(history)} 条")
        else:
            self.conversation_history = []
        
        # 添加用户当前指令
        self.conversation_history.append({"role": "user", "content": instruction})
        
        system_prompt = self.system_prompt or "你是一个能力强大的AI助手，可以使用各种工具来解决问题。请仔细分析用户的问题，决定是否需要使用工具，并生成最终的详细回答。"
        
        messages_for_gemini = self.conversation_history.copy() # Use a copy
        
        results_log = [] # Log actions taken
        tool_results_map = {} # Store results from tool executions
        tools_execution_failed = {} # Store tool execution failures
        
        # 记录当前已完成的思考步骤数
        completed_thinking_steps = 0
        
        for step in range(self.max_steps):
            logger.info(f"执行步骤 {step+1}/{self.max_steps}")
            
            # 检查是否已完成足够的思考步骤
            if self.force_thinking and completed_thinking_steps < self.thinking_steps:
                logger.info(f"当前已完成思考步骤: {completed_thinking_steps}/{self.thinking_steps}")
            
            # 获取工具定义 (OpenAI format)
            tool_definitions = self.get_tool_definitions()
            logger.trace(f"工具定义 (传递给GeminiClient): {json.dumps(tool_definitions, ensure_ascii=False)}")
            
            # 调用 Gemini 进行函数/工具调用决策
            # GeminiClient.function_calling handles message and tool format conversion
            logger.debug(f"向Gemini发送函数调用请求 (第 {step+1} 步)")
            function_decision_result = await self.client.function_calling(
                model=self.model,
                messages=messages_for_gemini, # Pass current history
                tools=tool_definitions,
                system_prompt=system_prompt,
                temperature=self.temperature # Pass temperature
            )
            
            # 检查API调用是否出错
            if "error" in function_decision_result:
                error_msg = function_decision_result["error"]
                logger.error(f"Gemini函数调用API错误: {error_msg}")
                results_log.append(f"步骤 {step+1} 错误: Gemini API 失败 - {error_msg}")
                # Decide whether to break or try generating a text response
                # For now, let's break and generate final answer based on failure
                tools_execution_failed["api_error"] = f"无法调用Gemini进行工具决策: {error_msg}"
                break 
                
            # 处理Gemini的决策结果
            llm_message = function_decision_result.get("message", "")
            tool_calls = function_decision_result.get("tool_calls", [])
            
            # 将模型的文本思考或回复（如果有）添加到历史记录
            if llm_message:
                logger.debug(f"Gemini文本回复 (步骤 {step+1}): {llm_message[:200]}...")
                # 不再使用content格式，统一使用parts格式，避免格式不一致
                # messages_for_gemini.append({"role": "assistant", "content": llm_message})
                results_log.append({
                    "step": step + 1,
                    "type": "thinking",
                    "content": llm_message[:100] + "...",
                    "result_summary": f"步骤 {step+1} 思考: {llm_message[:100]}..."
                })
                # 更新已完成的思考步骤计数
                completed_thinking_steps += 1
            
            # 如果没有工具调用，并且有文本回复，判断是继续思考还是返回最终答案
            if not tool_calls and llm_message:
                # 检查是否已经执行了足够的思考步骤
                if not self.force_thinking or completed_thinking_steps >= self.thinking_steps:
                    logger.info(f"已完成 {completed_thinking_steps} 步思考，达到或超过所需的 {self.thinking_steps} 步，返回最终回复。")
                    final_answer = llm_message
                    # 使用标准格式添加到历史
                    messages_for_gemini.append({"role": "assistant", "parts": [{"text": llm_message}]})
                    self.conversation_history = messages_for_gemini # Update main history
                    return {"answer": final_answer}
                else:
                    # 强制继续思考过程
                    logger.info(f"【强制思考模式】只完成了 {completed_thinking_steps}/{self.thinking_steps} 步思考，继续思考过程...")
                    # 将思考结果添加到历史
                    messages_for_gemini.append({"role": "assistant", "parts": [{"text": llm_message}]})
                    
                    # 添加一个提示，要求继续思考
                    next_prompt = self.thinking_prompt
                    
                    # 添加针对性的思考指示，确保模型考虑前面获取的信息
                    if self.check_results_for_tool_usage(results_log, "weather"):
                        next_prompt += "\n\n请记得考虑已经获取的天气信息，并将其作为规划的重要依据。"
                    
                    # 提示模型基于之前的思考继续深入
                    next_prompt += "\n\n请基于前面的思考内容继续深入分析，保持连贯性，并逐步完善解决方案。不要重新开始分析，而是承接前面的思考结果。"
                    
                    logger.debug(f"添加思考提示: '{next_prompt}'")
                    messages_for_gemini.append({"role": "user", "parts": [{"text": next_prompt}]})
                    continue  # 继续下一轮思考

            # 如果没有工具调用，也没有文本回复（异常情况），跳出循环生成通用回复
            if not tool_calls and not llm_message:
                 logger.warning("Gemini既未要求工具调用，也未生成文本回复。")
                 results_log.append(f"步骤 {step+1}: Gemini未返回有效操作。")
                 break # Exit loop, will generate final answer based on context
                 
            # --- 执行工具调用 --- 
            if tool_calls:
                 # Construct the single assistant turn containing both text (if any) and function calls
                 assistant_parts = []
                 if llm_message:
                     assistant_parts.append({"text": llm_message})
                 for tool_call in tool_calls:
                     assistant_parts.append({"functionCall": {
                         "name": tool_call["name"],
                         "args": tool_call["arguments"]
                     }})
                 # 确保使用正确的格式
                 messages_for_gemini.append({"role": "assistant", "parts": assistant_parts})

                 # Execute tools and collect all response parts for this step
                 tool_response_parts = [] 
                 all_tools_succeeded = True
                 for tool_call in tool_calls:
                     tool_name = tool_call["name"]
                     tool_args = tool_call["arguments"]
                     
                     logger.info(f"执行工具: {tool_name}, 参数: {json.dumps(tool_args, ensure_ascii=False)}")
                     tool_result = await self.execute_tool(tool_name, **tool_args)
                     
                     # Prepare the content for the functionResponse part
                     response_content = {}
                     if "error" in tool_result and tool_result["error"]:
                         tools_execution_failed[tool_name] = tool_result["error"]
                         logger.warning(f"工具 {tool_name} 执行失败: {tool_result['error']}")
                         response_content = {"error": tool_result["error"]}
                         all_tools_succeeded = False
                     else:
                         # 根据工具名称处理不同的响应格式
                         if tool_name == "code":
                             # 代码工具返回可能包含多个字段，统一为一个结构化响应
                             response_content = {
                                 "result": tool_result.get("result"),
                                 "stdout": tool_result.get("stdout", ""),
                                 "stderr": tool_result.get("stderr", ""),
                                 "error": tool_result.get("error")
                             }
                         else:
                             response_content = tool_result # 其他工具使用完整结果字典
                         
                     # Create the individual functionResponse part and add to list
                     tool_response_parts.append({
                         "functionResponse": {
                             "name": tool_name,
                             "response": { 
                                 "content": response_content 
                             }
                         }
                     })
                     
                     # Log the result
                     try:
                        # 确保结果可以序列化为JSON，裁剪过长内容
                        tool_result_copy = {}
                        for k, v in tool_result.items():
                            if isinstance(v, str) and len(v) > 500:
                                tool_result_copy[k] = v[:500] + "...(截断)"
                            else:
                                tool_result_copy[k] = v
                                
                        tool_result_str = json.dumps(tool_result_copy, ensure_ascii=False)
                        results_log.append({
                            "step": step + 1,
                            "tool": tool_name,
                            "result_summary": f"工具 {tool_name} 结果: {tool_result_str[:200]}..."
                        })
                     except Exception as e:
                        logger.warning(f"工具结果序列化失败: {e}")
                        results_log.append({
                            "step": step + 1,
                            "tool": tool_name, 
                            "result_summary": f"工具 {tool_name} 结果: (无法序列化为JSON)"
                        })
                     tool_results_map[tool_name] = tool_result

                 # Append ONE SINGLE message with role 'tool' containing ALL collected response parts
                 if tool_response_parts:
                     # 确保添加的工具响应与对应的函数调用数量一致
                     if len(tool_response_parts) != len(assistant_parts) - (1 if llm_message else 0):
                         logger.warning(f"工具响应数量({len(tool_response_parts)})与函数调用数量({len(assistant_parts) - (1 if llm_message else 0)})不匹配")
                         # 如果数量不匹配，可能需要修复工具响应
                         for i in range(len(assistant_parts) - (1 if llm_message else 0) - len(tool_response_parts)):
                             # 添加空响应以匹配函数调用数量
                             tool_response_parts.append({
                                 "functionResponse": {
                                     "name": "unknown",
                                     "response": { 
                                         "content": {"error": "工具响应丢失"}
                                     }
                                 }
                             })
                     
                     messages_for_gemini.append({"role": "tool", "parts": tool_response_parts})
                     logger.trace(f"Appended single tool message with {len(tool_response_parts)} response part(s).")
            else:
                 logger.info("没有工具调用请求，进入下一步思考或生成最终答案。")
                 # 注意：此处不直接break，让循环继续判断是否需要更多思考步骤
        
        logger.info(f"已完成的思考步骤总数: {completed_thinking_steps}/{self.thinking_steps}")
        
        # --- 生成最终答案 --- 
        logger.info("工具调用循环结束或达到最大步骤，开始生成最终答案...")
        
        # 准备最终生成请求的消息历史 (包含所有思考、工具调用和结果)
        final_messages = messages_for_gemini
        
        # Add a final prompt instructing the model to summarize and answer
        final_prompt_text = """基于以上所有思考和工具执行结果，请生成最终的、具体的、实用的回答。
请注意：
1. 不要重复你的思考过程，而是直接提供最终结论和实际可执行的建议
2. 提供具体的、有针对性的信息，避免泛泛而谈
3. 对于规划类问题，应该包含具体的时间安排、地点推荐、预算建议等实际内容
4. 回答应该是用户可以直接采取行动的，而不是分析框架或方法论
5. 确保你的回答简洁清晰、直接解决用户的问题
6. 如果使用了工具获取信息（如天气），请在回答的开头明确提及这些关键信息"""

        # 针对特定工具结果添加额外提示
        if self.check_results_for_tool_usage(results_log, "weather") or any(tool_name == "weather" for tool_name in tool_results_map.keys()):
            final_prompt_text += "\n\n重要：请在回答的开头明确总结天气信息，如温度、天气状况等，并据此给出具体建议。"
            
        if tools_execution_failed:
             error_details = "\n".join([f"- 工具 '{name}' 失败: {reason}" for name, reason in tools_execution_failed.items()])
             final_prompt_text = f"在生成最终回答时，请注意以下工具执行失败了:\n{error_details}\n请告知用户相关信息无法获取，并根据可用的信息和对话历史给出具体的、实用的最终回答。不要包含你的思考过程，而是直接提供结论和建议。"
             
        final_messages.append({"role": "user", "content": final_prompt_text})
        
        final_answer = ""
        error_generating_final = False
        
        # 使用 chat_completion 生成最终答案 (非流式)
        try:
            logger.debug("向Gemini发送最终答案生成请求 (非流式)")
            # Get the async generator
            response_generator = self.client.chat_completion(
                model=self.model,
                messages=final_messages,
                system_prompt=system_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False # Still False
            )
            
            # Await the first (and only) item from the generator
            response = await anext(response_generator, None) 

            if response is None:
                logger.error("生成最终答案时未收到任何响应 (非流式)")
                final_answer = "抱歉，在生成最终回复时未收到响应。"
                error_generating_final = True
            elif "error" in response:
                logger.error(f"生成最终答案时出错: {response['error']}")
                final_answer = f"抱歉，在生成最终回复时遇到错误: {response['error'].get('message', '未知错误')}"
                error_generating_final = True
            else:
                # Extract text from the complete response
                final_answer = self._extract_text_from_gemini_response(response)
                if not final_answer: # Handle case where extraction fails even on full response
                     logger.warning("无法从非流式Gemini响应中提取最终文本答案。")
                     final_answer = "抱歉，我无法生成有效的回复（解析错误）。"
                     error_generating_final = True # Treat as error if extraction failed

        except StopAsyncIteration: # Handle case where generator finishes unexpectedly
             logger.error("生成最终答案时响应生成器意外结束 (非流式)")
             final_answer = "抱歉，生成最终回复时发生内部错误（响应中断）。"
             error_generating_final = True
        except Exception as e:
             logger.exception("生成最终答案时发生意外错误 (非流式)")
             final_answer = f"抱歉，生成最终回复时发生内部错误。"
             error_generating_final = True
             
        if not final_answer and not error_generating_final:
             final_answer = "抱歉，我无法生成有效的回复。"
             logger.warning("Gemini未生成任何最终文本答案。")
             
        # 更新主对话历史记录 (用最后一次生成请求前的历史)
        self.conversation_history = final_messages[:-1] # Remove the final prompt we added
        self.conversation_history.append({"role": "assistant", "content": final_answer})
        
        return {
            "steps": results_log, # Log of actions taken
            "answer": final_answer.strip()
        } 

    def check_results_for_tool_usage(self, results_log, tool_name):
        """检查结果日志中是否使用了特定工具
        
        Args:
            results_log: 结果日志
            tool_name: 工具名称
            
        Returns:
            bool: 是否使用了该工具
        """
        return any(
            (isinstance(result, dict) and result.get("tool") == tool_name) or
            (isinstance(result, str) and f"工具 {tool_name}" in result)
            for result in results_log
        ) 

    async def _direct_response(self, instruction: str, history: List[Dict[str, str]] = None) -> Dict:
        """当MCP禁用时，直接生成回复而不进行多步思考
        
        Args:
            instruction: 用户指令
            history: 历史对话记录，可选
            
        Returns:
            Dict: 执行结果
        """
        # 初始化会话历史
        if history:
            self.conversation_history = history.copy()
        else:
            self.conversation_history = []
        
        # 添加用户当前指令
        self.conversation_history.append({"role": "user", "content": instruction})
        
        system_prompt = self.system_prompt or "你是一个能力强大的AI助手，可以使用各种工具来解决问题。请仔细分析用户的问题，决定是否需要使用工具，并生成最终的详细回答。"
        
        # 获取工具定义
        tool_definitions = self.get_tool_definitions()
        logger.trace(f"工具定义 (传递给GeminiClient): {json.dumps(tool_definitions, ensure_ascii=False)}")
        
        # 调用 Gemini 进行单次函数/工具调用决策
        logger.debug("向Gemini发送单次函数调用请求 (MCP禁用模式)")
        function_decision_result = await self.client.function_calling(
            model=self.model,
            messages=self.conversation_history,
            tools=tool_definitions,
            system_prompt=system_prompt,
            temperature=self.temperature
        )
        
        # 检查API调用是否出错
        if "error" in function_decision_result:
            error_msg = function_decision_result["error"]
            logger.error(f"Gemini函数调用API错误: {error_msg}")
            return {"answer": f"抱歉，我无法处理您的请求。(API错误: {error_msg})"}
            
        # 处理Gemini的决策结果
        llm_message = function_decision_result.get("message", "")
        tool_calls = function_decision_result.get("tool_calls", [])
        
        # 如果没有工具调用，直接返回文本回复
        if not tool_calls and llm_message:
            logger.info("Gemini未要求工具调用，直接返回文本回复")
            # 添加回复到历史记录
            self.conversation_history.append({"role": "assistant", "content": llm_message})
            return {"answer": llm_message}
            
        # 如果有工具调用，执行工具并生成最终回复
        if tool_calls:
            results_log = []  # 记录工具执行结果
            tool_results_map = {}  # 存储工具执行结果
            tools_execution_failed = {}  # 记录工具执行失败信息
            
            # 构建助手消息，包含文本和函数调用
            assistant_parts = []
            if llm_message:
                assistant_parts.append({"text": llm_message})
            for tool_call in tool_calls:
                assistant_parts.append({"functionCall": {
                    "name": tool_call["name"],
                    "args": tool_call["arguments"]
                }})
                
            # 添加到历史记录
            self.conversation_history.append({"role": "assistant", "parts": assistant_parts})
            
            # 执行工具
            tool_response_parts = []
            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["arguments"]
                
                logger.info(f"执行工具: {tool_name}, 参数: {json.dumps(tool_args, ensure_ascii=False)}")
                tool_result = await self.execute_tool(tool_name, **tool_args)
                
                # 准备工具响应
                response_content = {}
                if "error" in tool_result and tool_result["error"]:
                    tools_execution_failed[tool_name] = tool_result["error"]
                    logger.warning(f"工具 {tool_name} 执行失败: {tool_result['error']}")
                    response_content = {"error": tool_result["error"]}
                else:
                    # 处理不同工具的响应格式
                    if tool_name == "code":
                        response_content = {
                            "result": tool_result.get("result"),
                            "stdout": tool_result.get("stdout", ""),
                            "stderr": tool_result.get("stderr", ""),
                            "error": tool_result.get("error")
                        }
                    else:
                        response_content = tool_result
                
                # 添加工具响应
                tool_response_parts.append({
                    "functionResponse": {
                        "name": tool_name,
                        "response": {
                            "content": response_content
                        }
                    }
                })
                
                # 记录工具执行结果
                try:
                    tool_result_copy = {}
                    for k, v in tool_result.items():
                        if isinstance(v, str) and len(v) > 500:
                            tool_result_copy[k] = v[:500] + "...(截断)"
                        else:
                            tool_result_copy[k] = v
                    
                    tool_result_str = json.dumps(tool_result_copy, ensure_ascii=False)
                    results_log.append({
                        "step": 1,
                        "tool": tool_name,
                        "result_summary": f"工具 {tool_name} 结果: {tool_result_str[:200]}..."
                    })
                except Exception as e:
                    logger.warning(f"工具结果序列化失败: {e}")
                    results_log.append({
                        "step": 1,
                        "tool": tool_name,
                        "result_summary": f"工具 {tool_name} 结果: (无法序列化为JSON)"
                    })
                tool_results_map[tool_name] = tool_result
            
            # 添加工具响应到历史记录
            if tool_response_parts:
                self.conversation_history.append({"role": "tool", "parts": tool_response_parts})
            
            # 生成最终回复
            logger.info("生成包含工具执行结果的最终回复")
            
            # 添加最终提示，指导模型生成回复
            final_prompt_text = """根据以上工具执行结果，请生成最终回答。
请直接提供用户可以采取行动的具体回答，不要重复工具执行的过程。
如果工具执行失败，请告知用户相关信息无法获取，并根据可用信息给出建议。"""

            # 针对特定工具结果添加提示
            if any(result.get("tool") == "weather" for result in results_log if isinstance(result, dict)):
                final_prompt_text += "\n\n如果获取了天气信息，请在回答的开头明确总结天气信息，并据此给出具体建议。"
            
            # 添加提示到历史记录
            self.conversation_history.append({"role": "user", "content": final_prompt_text})
            
            # 获取最终回复
            try:
                logger.debug("向Gemini发送最终回复生成请求")
                response_generator = self.client.chat_completion(
                    model=self.model,
                    messages=self.conversation_history,
                    system_prompt=system_prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=False
                )
                
                response = await anext(response_generator, None)
                
                if response is None:
                    logger.error("生成最终回复时未收到响应")
                    return {"answer": "抱歉，在生成最终回复时未收到响应。"}
                elif "error" in response:
                    logger.error(f"生成最终回复时出错: {response['error']}")
                    return {"answer": f"抱歉，在生成最终回复时遇到错误: {response['error'].get('message', '未知错误')}"}
                else:
                    final_answer = self._extract_text_from_gemini_response(response)
                    if not final_answer:
                        logger.warning("无法从Gemini响应中提取最终回复")
                        return {"answer": "抱歉，我无法生成有效的回复。"}
                    
                    # 添加回复到历史记录
                    self.conversation_history.append({"role": "assistant", "content": final_answer})
                    return {"steps": results_log, "answer": final_answer.strip()}
                    
            except Exception as e:
                logger.exception("生成最终回复时发生错误")
                return {"answer": "抱歉，在生成最终回复时发生错误。"}
                
        # 异常情况：既没有文本回复也没有工具调用
        logger.warning("Gemini既未生成文本回复，也未要求工具调用")
        return {"answer": "抱歉，我无法理解您的请求。请尝试重新表述您的问题。"} 