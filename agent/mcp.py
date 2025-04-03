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
                 system_prompt: Optional[str] = None):
        """初始化MCP代理
        
        Args:
            client: 已初始化的 GeminiClient 实例
            model: 使用的模型名称
            max_tokens: 最大生成token数
            temperature: 温度参数
            max_steps: 最大执行步骤数
            thinking_steps: 思考步骤数
            system_prompt: 自定义系统提示词，如果为None则使用默认提示词
        """
        if not isinstance(client, GeminiClient):
             raise TypeError("client must be an instance of GeminiClient")
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_steps = max_steps
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
        
        for step in range(self.max_steps):
            logger.info(f"执行步骤 {step+1}/{self.max_steps}")
            
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
                 messages_for_gemini.append({"role": "assistant", "content": llm_message})
                 results_log.append(f"步骤 {step+1} 思考: {llm_message[:100]}...")
            
            # 如果没有工具调用，并且有文本回复，我们认为这是最终答案
            if not tool_calls and llm_message:
                 logger.info("Gemini未要求工具调用，直接返回文本回复。")
                 final_answer = llm_message
                 self.conversation_history = messages_for_gemini # Update main history
                 return {"answer": final_answer}
                 
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
                     if "error" in tool_result:
                         tools_execution_failed[tool_name] = tool_result["error"]
                         logger.warning(f"工具 {tool_name} 执行失败: {tool_result['error']}")
                         response_content = {"error": tool_result["error"]}
                         all_tools_succeeded = False
                     else:
                         response_content = tool_result # Use the full result dict
                         
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
                        tool_result_str = json.dumps(tool_result, ensure_ascii=False)
                        results_log.append(f"工具 {tool_name} 结果: {tool_result_str[:200]}...")
                     except Exception:
                        results_log.append(f"工具 {tool_name} 结果: (无法序列化为JSON)")
                     tool_results_map[tool_name] = tool_result

                 # Append ONE SINGLE message with role 'tool' containing ALL collected response parts
                 if tool_response_parts:
                     messages_for_gemini.append({"role": "tool", "parts": tool_response_parts})
                     logger.trace(f"Appended single tool message with {len(tool_response_parts)} response part(s).")
                 
                 # Optional: Break loop if any tool failed
                 # if not all_tools_succeeded:
                 #    logger.warning("由于工具执行失败，提前结束步骤循环。")
                 #    break 
            else:
                 logger.info("没有工具调用请求，结束当前步骤循环。")
                 break
                 
        # --- 生成最终答案 --- 
        logger.info("工具调用循环结束或达到最大步骤，开始生成最终答案...")
        
        # 准备最终生成请求的消息历史 (包含所有思考、工具调用和结果)
        final_messages = messages_for_gemini
        
        # Add a final prompt instructing the model to summarize and answer
        final_prompt_text = "基于以上对话和工具执行结果，请生成最终的、完整的、对用户友好的回答。"
        if tools_execution_failed:
             error_details = "\n".join([f"- 工具 '{name}' 失败: {reason}" for name, reason in tools_execution_failed.items()])
             final_prompt_text = f"在生成最终回答时，请注意以下工具执行失败了:\n{error_details}\n请告知用户相关信息无法获取，并根据可用的信息和对话历史给出最终回答。"
             
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