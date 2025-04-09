import aiohttp
import json
import asyncio
from typing import Dict, List, Optional, AsyncGenerator, Any, Tuple, Union
from loguru import logger
import time
import re

# --- Add Fish Audio SDK imports ---
from fish_audio_sdk import Session as FishSession, TTSRequest
import io

# --- 添加MiniMax相关的导入 ---
import base64
import requests

# --- Helper function to convert OpenAI format messages to Gemini format ---
def convert_messages_to_gemini(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Converts internal message history to Gemini's 'contents' format.
    Ensures strict user/model alternation, correctly placing 'tool' messages.
    Handles potential errors in the input history structure.
    """
    processed_contents = []
    last_role_added = None # Tracks the last role added ('user' or 'model')

    system_instruction = None
    if messages and messages[0].get("role") == "system":
        system_instruction = {"role": "system", "parts": [{"text": messages[0].get("content", "")}]}
        messages = messages[1:]

    for i, msg in enumerate(messages):
        role = msg.get("role")
        parts = msg.get("parts")
        content = msg.get("content") # Fallback for user message

        # Determine the role and parts for the current message
        current_role = None
        current_parts = []

        if role == "user":
            current_role = "user"
            if parts:
                current_parts = parts
            elif content:
                current_parts = [{"text": content}]
            else:
                logger.warning(f"User message at index {i} has no parts/content. Skipping.")
                continue

        elif role == "assistant" or role == "model":
            current_role = "model"
            if parts:
                current_parts = parts
            elif content:
                # 转换content为parts格式
                current_parts = [{"text": content}]
            else:
                logger.warning(f"Model/Assistant message at index {i} missing 'parts' and 'content'. Skipping.")
                continue

        elif role == "tool":
            current_role = "tool"
            if parts:
                current_parts = parts
            else:
                logger.warning(f"Tool message at index {i} missing 'parts'. Skipping.")
                continue
        else:
            logger.warning(f"Unsupported role '{role}' at index {i}. Skipping.")
            continue

        # --- Validate and Append based on Role Sequence ---
        if current_role == "user":
            if last_role_added == "user":
                logger.error(f"History Error: Consecutive user roles at index {i}. Skipping current user message.")
                continue
            processed_contents.append({"role": "user", "parts": current_parts})
            last_role_added = "user"

        elif current_role == "model":
            if last_role_added == "model":
                logger.error(f"History Error: Consecutive model roles at index {i}. Skipping current model message.")
                continue
            # Model role MUST follow a user role
            if last_role_added != "user":
                 logger.error(f"History Error: Model role at index {i} does not follow user role (last was {last_role_added}). Skipping model message.")
                 continue
            processed_contents.append({"role": "model", "parts": current_parts})
            last_role_added = "model"

        elif current_role == "tool":
            # Tool role MUST follow a model role
            if last_role_added != "model":
                 logger.error(f"History Error: Tool role at index {i} does not follow model role (last was {last_role_added}). Skipping tool message.")
                 continue
            processed_contents.append({"role": "tool", "parts": current_parts})
            # IMPORTANT: Do *not* update last_role_added for 'tool' roles
            # The next role expected is still 'user' after the model->tool sequence.
            
    # --- Final Check: Ensure last message is user --- 
    if processed_contents and processed_contents[-1]['role'] != 'user':
        logger.warning("History does not end with 'user' role. Appending placeholder user message.")
        processed_contents.append({"role": "user", "parts": [{"text": "(Summarize or continue)"}]})
        
    if system_instruction:
        return processed_contents, system_instruction
    else:
        return processed_contents, None


# --- Helper function to convert OpenAI function schema to Gemini Tool format ---
def convert_tools_to_gemini(tools: List[Dict]) -> Optional[List[Dict]]:
    """Converts OpenAI function definitions to Gemini's 'tools' format."""
    gemini_tools = []
    function_declarations = []
    for tool_dict in tools:
        func_data = tool_dict.get("function")
        if func_data:
            # Basic type mapping (can be expanded)
            param_properties = {}
            required_params = func_data.get("parameters", {}).get("required", [])
            for name, schema in func_data.get("parameters", {}).get("properties", {}).items():
                param_properties[name] = {
                    "type": schema.get("type", "string").upper(), # Convert to uppercase e.g., STRING
                    "description": schema.get("description", "")
                }
                # Handle enums if present
                if "enum" in schema:
                    param_properties[name]["enum"] = schema["enum"]

            declaration = {
                "name": func_data.get("name"),
                "description": func_data.get("description"),
                "parameters": {
                    "type": "OBJECT", # Assuming top-level parameters are always an object
                    "properties": param_properties,
                    "required": required_params
                }
            }
            function_declarations.append(declaration)

    if function_declarations:
        # Gemini API expects a list containing one tool object with functionDeclarations
        gemini_tools.append({"functionDeclarations": function_declarations})

    return gemini_tools

# --- Main API Client Class ---
class GeminiClient:
    """与 Google Gemini API 交互的客户端"""

    def __init__(self, api_key: str, base_url: str):
        """初始化 Gemini 客户端

        Args:
            api_key: Google AI Studio or Vertex AI API Key
            base_url: Gemini API Endpoint (e.g., https://generativelanguage.googleapis.com/v1beta)
        """
        if not api_key:
            raise ValueError("Gemini API Key is required.")
        self.api_key = api_key
        # Ensure base_url ends correctly depending on its structure
        if "googleapis.com" in base_url:
            # Standard Google AI base URL structure
            self.base_url = base_url.rstrip('/') # e.g. https://generativelanguage.googleapis.com/v1beta
        else:
            # Potentially Vertex AI or other structure - adjust as needed
            self.base_url = base_url.rstrip('/')
            logger.warning(f"Base URL '{base_url}' doesn't look like standard Google AI URL. Ensure it's correct.")
            
        # 会话历史管理
        self.chat_histories = {}  # 存储不同会话的历史记录 {session_id: [{"message": message, "timestamp": timestamp}, ...]}
        self.max_history = 20  # 默认每个会话保留20条历史记录
        self.memory_expire_hours = 24  # 默认记忆保留24小时
        
    def set_max_history(self, max_history: int):
        """设置最大保留的历史记录条数
        
        Args:
            max_history: 最大历史记录条数
        """
        if max_history < 1:
            raise ValueError("最大历史记录条数必须大于0")
        self.max_history = max_history
        logger.info(f"已设置最大历史记录条数为: {max_history}")
    
    def set_memory_expire_hours(self, hours: int):
        """设置记忆保留时间
        
        Args:
            hours: 记忆保留时间(小时)
        """
        if hours < 1:
            raise ValueError("记忆保留时间必须大于0小时")
        self.memory_expire_hours = hours
        logger.info(f"已设置记忆保留时间为: {hours}小时")
        
    def get_chat_history(self, session_id: str) -> List[Dict[str, Any]]:
        """获取指定会话的历史记录
        
        Args:
            session_id: 会话ID
            
        Returns:
            List[Dict]: 历史记录列表，符合Gemini API要求的格式
        """
        # 先清理过期的历史记录
        self._clean_expired_history(session_id)
        
        # 获取会话历史并转换为API所需格式
        history_with_timestamps = self.chat_histories.get(session_id, [])
        history = []
        
        for i, item in enumerate(history_with_timestamps):
            msg = item["message"]
            role = msg["role"]
            content = msg["content"]
            
            if role == "user":
                history.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant":
                history.append({"role": "model", "parts": [{"text": content}]})
        
        # 确保历史记录符合用户/助手交替模式
        valid_history = []
        last_role = None
        
        for item in history:
            current_role = item["role"]
            
            # 跳过连续的相同角色消息
            if current_role == last_role:
                continue
                
            valid_history.append(item)
            last_role = current_role
        
        return valid_history
        
    def add_to_chat_history(self, session_id: str, role: str, content: str):
        """添加消息到会话历史记录
        
        Args:
            session_id: 会话ID
            role: 角色，'user' 或 'assistant'
            content: 消息内容
        """
        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = []
            
        # 添加消息和时间戳
        current_time = time.time()
        self.chat_histories[session_id].append({
            "message": {"role": role, "content": content},
            "timestamp": current_time
        })
        
        # 如果超过最大长度，移除最早的消息
        while len(self.chat_histories[session_id]) > self.max_history:
            self.chat_histories[session_id].pop(0)
            
        logger.debug(f"已添加消息到会话 {session_id} 的历史记录，当前历史长度: {len(self.chat_histories[session_id])}")
        
    def clear_chat_history(self, session_id: str):
        """清除指定会话的历史记录
        
        Args:
            session_id: 会话ID
        """
        if session_id in self.chat_histories:
            self.chat_histories[session_id] = []
            logger.info(f"已清除会话 {session_id} 的历史记录")
    
    def _clean_expired_history(self, session_id: str = None):
        """清理过期的历史记录
        
        Args:
            session_id: 会话ID，如果为None则清理所有会话
        """
        current_time = time.time()
        expire_seconds = self.memory_expire_hours * 3600  # 转换为秒
        
        if session_id:
            # 只清理指定会话
            if session_id in self.chat_histories:
                # 过滤掉过期的记录
                self.chat_histories[session_id] = [
                    item for item in self.chat_histories[session_id]
                    if (current_time - item["timestamp"]) < expire_seconds
                ]
        else:
            # 清理所有会话
            for sid in list(self.chat_histories.keys()):
                # 过滤掉过期的记录
                self.chat_histories[sid] = [
                    item for item in self.chat_histories[sid]
                    if (current_time - item["timestamp"]) < expire_seconds
                ]
                
                # 如果会话没有有效记录，删除该会话
                if not self.chat_histories[sid]:
                    del self.chat_histories[sid]
    
    def _get_request_url(self, model: str, stream: bool = False, task: str = "generateContent") -> str:
        """Constructs the appropriate Gemini API URL."""
        action = "streamGenerateContent" if stream else task
        # Assumes base_url like https://generativelanguage.googleapis.com/v1beta
        # Adjust if using Vertex AI (structure might differ, e.g., :predict)
        return f"{self.base_url}/models/{model}:{action}?key={self.api_key}"

    async def _make_request(self,
                            url: str,
                            payload: Dict,
                            stream: bool) -> AsyncGenerator[Dict, None]:
        """Makes the HTTP request and handles streaming/non-streaming responses."""
        headers = {'Content-Type': 'application/json'}
        # Increased timeout for potentially long generations or complex workflows
        timeout = aiohttp.ClientTimeout(total=300, connect=30)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                logger.debug(f"Sending Gemini Request: URL={url}, Payload={json.dumps(payload, ensure_ascii=False)[:500]}...")
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Gemini API Error: {response.status} - {error_text}")
                        yield {"error": {"code": response.status, "message": f"Gemini API Error: {error_text}"}}
                        return

                    if stream:
                        # Process Server-Sent Events stream for Gemini
                        # Gemini's stream often sends a list of chunks in each event data
                        async for line_bytes in response.content:
                            line = line_bytes.decode('utf-8').strip()
                            if line.startswith("data: "):
                                data_str = line[len("data: "):]
                                try:
                                    # Gemini stream data might be a full JSON object per line now
                                    chunk = json.loads(data_str)
                                    yield chunk # Yield the raw chunk structure
                                except json.JSONDecodeError:
                                    logger.warning(f"Could not decode Gemini stream chunk: {data_str}")
                            elif line: # Log other lines if needed
                                logger.trace(f"Received non-data line from Gemini stream: {line}")

                    else:
                        # Handle non-streaming response
                        try:
                            full_response = await response.json()
                            logger.debug(f"Received Gemini Response: {json.dumps(full_response, ensure_ascii=False)[:500]}...")
                            yield full_response
                        except json.JSONDecodeError:
                            resp_text = await response.text()
                            logger.error(f"Failed to decode non-streaming Gemini JSON response: {resp_text}")
                            yield {"error": {"message": "Failed to decode Gemini JSON response"}}
                        except Exception as e:
                             logger.error(f"Error reading non-streaming Gemini response: {e}")
                             yield {"error": {"message": f"Error reading Gemini response: {e}"}}


        except asyncio.TimeoutError:
             logger.error(f"Request to Gemini API timed out: {url}")
             yield {"error": {"code": 408, "message": "Request Timeout"}}
        except aiohttp.ClientError as e:
            logger.error(f"HTTP Client Error connecting to Gemini API: {e}")
            yield {"error": {"code": 503, "message": f"HTTP Client Error: {e}"}}
        except Exception as e:
            logger.exception("Unexpected error during Gemini API request")
            yield {"error": {"code": 500, "message": f"Unexpected Error: {e}"}}

    async def chat_completion(self,
                              model: str,
                              messages: List[Dict[str, str]],
                              system_prompt: Optional[str] = None,
                              temperature: Optional[float] = None,
                              max_tokens: Optional[int] = None,
                              stream: bool = False) -> AsyncGenerator[Dict, None]:
        """Generates content using Gemini, mimicking chat completion. Handles streaming."""

        contents, system_instruction = convert_messages_to_gemini(messages)
        if not contents or contents[-1]['role'] != 'user':
             # Gemini API requires the last content item to be from the 'user' role
             logger.warning("Adding empty user turn to end of history for Gemini API.")
             contents.append({"role": "user", "parts": [{"text": "(continue)"}]})


        payload = {"contents": contents}

        # Add generation config
        generation_config = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens
        # Add other configs like topP, topK if needed
        if generation_config:
            payload["generationConfig"] = generation_config

        # Add system instruction if provided
        if system_prompt:
            # Gemini's v1beta supports systemInstruction object
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        request_url = self._get_request_url(model, stream=stream, task="generateContent")

        async for response_chunk in self._make_request(url=request_url, payload=payload, stream=stream):
            yield response_chunk # Yield the raw Gemini chunk/response

    async def function_calling(self,
                             model: str,
                             messages: List[Dict[str, str]],
                             tools: List[Dict],
                             system_prompt: Optional[str] = None,
                             temperature: Optional[float] = None # Gemini supports temp for function calling too
                            ) -> Dict:
        """Performs function calling using Gemini API."""

        contents, system_instruction = convert_messages_to_gemini(messages)
        if not contents or contents[-1]['role'] != 'user':
            # Ensure last message is user for the request
             logger.warning("Adding empty user turn for Gemini function calling request.")
             contents.append({"role": "user", "parts": [{"text": "(requesting tool use)"}]})

        gemini_tools = convert_tools_to_gemini(tools)

        payload = {
            "contents": contents,
            "tools": gemini_tools,
            # Optional: Tool Config to force function call 'mode': 'FUNCTION' or 'ANY'
            # "toolConfig": {"functionCallingConfig": {"mode": "ANY"}}
        }

        # Add generation config (optional but can influence function choice/args)
        generation_config = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if generation_config:
             payload["generationConfig"] = generation_config

        # Add system instruction if provided
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        # Function calling is typically non-streaming
        request_url = self._get_request_url(model, stream=False, task="generateContent")

        # Expecting a single response for function calling
        final_result = {}
        response_data = None
        async for result in self._make_request(url=request_url, payload=payload, stream=False):
            response_data = result # Get the first (and only) result
            break # Exit after first result for non-streaming

        if not response_data:
            logger.error("No response received from Gemini for function calling.")
            return {"error": "No response from Gemini"}

        if "error" in response_data:
            logger.error(f"Gemini function calling failed: {response_data['error']}")
            return {"error": response_data['error'].get('message', 'Unknown Gemini Error')}

        # Process the Gemini response to extract function calls or text content
        try:
            # Gemini response structure: response['candidates'][0]['content']['parts'][...]
            candidates = response_data.get("candidates", [])
            if not candidates:
                logger.warning("Gemini response missing 'candidates'.")
                return {"message": "", "tool_calls": []} # Or return an error

            # Usually interested in the first candidate
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if not parts:
                 logger.warning("Gemini response candidate missing 'parts'.")
                 return {"message": "", "tool_calls": []} # Or return an error

            tool_calls = []
            message_content = ""

            for part in parts:
                if "functionCall" in part:
                    fc = part["functionCall"]
                    # 确保生成一个唯一的ID，不依赖于参数内容（可能过长的代码会导致哈希不稳定）
                    tool_id = f"call_{len(tool_calls)}_{int(time.time()) % 10000}"
                    
                    # 正确处理各种参数类型，包括嵌套结构
                    arguments = fc.get("args", {})
                    
                    tool_calls.append({
                        "id": tool_id,
                        "name": fc.get("name"),
                        "arguments": arguments
                    })
                elif "text" in part:
                    message_content += part["text"] + "\n"

            # Return in the format expected by MCPAgent
            return {
                "message": message_content.strip(),
                "tool_calls": tool_calls
            }

        except (KeyError, IndexError, TypeError) as e:
            logger.exception(f"Error parsing Gemini function calling response: {response_data}")
            return {"error": f"Failed to parse Gemini response: {e}"}
        except Exception as e:
             logger.exception("Unexpected error parsing Gemini function calling response.")
             return {"error": f"Unexpected error parsing Gemini response: {e}"}

# --- TTS API Client Class (Rewritten for Fish Audio SDK) ---
class TTSClient:
    """使用 Fish Audio SDK 与 TTS API 交互的客户端"""

    def __init__(self, 
                 api_key: str, 
                 default_reference_id: Optional[str] = None):
        """初始化 Fish Audio TTS 客户端

        Args:
            api_key: Fish Audio API 密钥
            default_reference_id: 默认使用的自定义模型 ID (从 config 读取)
        """
        if not api_key:
            raise ValueError("Fish Audio API Key is required.")
        if not default_reference_id:
            logger.warning("Fish Audio Reference ID 未在配置中提供，TTS 将无法使用自定义模型。")
            # Depending on your needs, you might raise an error or allow fallback to general models
            # For now, we'll store it but TTS calls might fail if ref_id is required by API
        
        self.api_key = api_key
        self.default_reference_id = default_reference_id
        self.reference_id = default_reference_id  # 添加reference_id属性
        # SDK handles session internally, but we might create one instance 
        # Be mindful of potential issues if not used in async context correctly
        # For async usage, creating session per request might be safer
        # self.session = FishSession(api_key=self.api_key)
        logger.info(f"Fish Audio TTS客户端初始化完成 (API Key: ...{api_key[-4:]})")

    def clean_text_for_tts(self, text):
        """清理文本，移除不适合TTS的符号和格式"""
        if not text:
            return ""
            
        # 去除零宽字符
        text = text.replace('\u200b', '').replace('\u200c', '')
        
        # 移除markdown格式符号
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # 移除粗体 **text**
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # 移除斜体 *text*
        text = re.sub(r'__(.*?)__', r'\1', text)      # 移除粗体 __text__
        text = re.sub(r'_(.*?)_', r'\1', text)        # 移除斜体 _text_
        text = re.sub(r'`(.*?)`', r'\1', text)        # 移除代码 `code`
        
        # 移除项目符号，保留内容
        text = re.sub(r'^\s*[*\-+]\s+', '', text, flags=re.MULTILINE)
        
        # 替换多个连续空行为一个单换行
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # 替换制表符为空格
        text = text.replace('\t', ' ')
        
        # 统一换行符
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # 将多个连续空格替换为一个
        text = re.sub(r' {2,}', ' ', text)
        
        # 清理行首行尾空白
        text = '\n'.join([line.strip() for line in text.split('\n')])
        
        return text

    # --- Helper function to run sync generator in thread ---
    def _collect_audio_sync(self, session: FishSession, request: TTSRequest) -> bytes:
         """Synchronously iterates over the TTS generator and collects audio chunks."""
         audio_buffer = io.BytesIO()
         chunk_count = 0
         total_bytes = 0
         
         try:
             # 使用带有超时的方式收集音频数据
             for chunk in session.tts(request):
                  if chunk:
                       audio_buffer.write(chunk)
                       chunk_count += 1
                       total_bytes += len(chunk)
             logger.debug(f"Fish Audio TTS: 收集了 {chunk_count} 个数据块, 总计 {total_bytes} 字节")
         except Exception as e:
             logger.error(f"Fish Audio TTS: 收集音频块时出错: {e}")
             
         # 确保获取完整数据
         if chunk_count == 0 or total_bytes == 0:
             logger.warning("Fish Audio TTS: 未收集到任何音频数据")
         
         return audio_buffer.getvalue()

    async def text_to_speech(self, text: str, format: str = "mp3") -> Optional[bytes]:
        """将文本转换为语音

        Args:
            text: 要转换的文本
            format: 音频格式 (mp3, wav, pcm)

        Returns:
            bytes: 音频数据
        """
        if not text:
            logger.warning("TTS请求文本为空")
            return None
            
        # 确保使用SDK支持的格式
        supported_formats = ["mp3", "wav", "pcm"]
        if format.lower() not in supported_formats:
            logger.warning(f"不支持的格式: {format}，将使用默认mp3格式")
            format = "mp3"
            
        # 文本截断和处理
        max_length = 1500  # 安全的最大文本长度
        cleaned_text = self.clean_text_for_tts(text)  # 先清理文本
        
        if len(cleaned_text) > max_length:
            logger.warning(f"TTS请求文本过长: {len(cleaned_text)}字符，已截断到{max_length}字符")
            cleaned_text = cleaned_text[:max_length] + "..."
        
        logger.debug(f"Fish TTS请求文本(已清理): '{cleaned_text}' (长度:{len(cleaned_text)}字符)")
        
        try:
            # 创建新会话处理每次请求，避免复用问题
            session = FishSession(self.api_key)
            
            # 构建请求对象
            request = TTSRequest(
                reference_id=self.default_reference_id,
                text=cleaned_text,
                format=format.lower()
            )
            
            # 记录开始请求TTS的时间
            start_time = time.time()
            
            # 在线程中同步收集音频数据
            try:
                audio_bytes = await asyncio.wait_for(
                    asyncio.to_thread(self._collect_audio_sync, session, request),
                    timeout=30.0  # 30秒超时保护
                )
            except asyncio.TimeoutError:
                logger.error("Fish Audio TTS 请求超时(30秒)")
                return None
                
            # 记录请求完成的时间和音频大小
            end_time = time.time()
            request_time = end_time - start_time
                
            if audio_bytes and len(audio_bytes) > 100:
                logger.info(f"TTS请求成功，获取到{len(audio_bytes)}字节的音频数据，请求耗时{request_time:.2f}秒")
                return audio_bytes
            else:
                logger.error(f"TTS请求成功但返回的音频数据无效或过小: {len(audio_bytes) if audio_bytes else 0}字节")
                return None
        except Exception as e:
            logger.exception(f"TTS请求失败: {str(e)}")
            return None

# --- MiniMax T2A v2 客户端实现 ---
class MinimaxTTSClient:
    """使用 MiniMax T2A v2 API 进行文本到语音转换的客户端"""
    
    def __init__(self,
                api_key: str,
                group_id: str,
                base_url: str = "https://api.minimax.chat/v1/t2a_v2"):
        """初始化 MiniMax TTS 客户端

        Args:
            api_key: MiniMax API 密钥
            group_id: 用户所属的组ID
            base_url: T2A v2 API 的基础 URL
        """
        if not api_key:
            raise ValueError("MiniMax API Key is required.")
        if not group_id:
            raise ValueError("MiniMax Group ID is required.")
            
        self.api_key = api_key
        self.group_id = group_id
        self.base_url = base_url
        logger.info(f"MiniMax T2A v2 客户端初始化完成 (API Key: ...{api_key[-4:]})")
    
    def clean_text_for_tts(self, text):
        """清理文本，移除不适合TTS的符号和格式"""
        if not text:
            return ""
            
        # 去除零宽字符
        text = text.replace('\u200b', '').replace('\u200c', '')
        
        # 移除markdown格式符号
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # 移除粗体 **text**
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # 移除斜体 *text*
        text = re.sub(r'__(.*?)__', r'\1', text)      # 移除粗体 __text__
        text = re.sub(r'_(.*?)_', r'\1', text)        # 移除斜体 _text_
        text = re.sub(r'`(.*?)`', r'\1', text)        # 移除代码 `code`
        
        # 移除项目符号，保留内容
        text = re.sub(r'^\s*[*\-+]\s+', '', text, flags=re.MULTILINE)
        
        # 替换多个连续空行为一个单换行
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # 替换制表符为空格
        text = text.replace('\t', ' ')
        
        # 统一换行符
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # 将多个连续空格替换为一个
        text = re.sub(r' {2,}', ' ', text)
        
        # 清理行首行尾空白
        text = '\n'.join([line.strip() for line in text.split('\n')])
        
        return text

    async def text_to_speech(self,
                            text: str,
                            voice_id: str = "male-qn-qingse",
                            model: str = "speech-02-hd",
                            format: str = "mp3",
                            stream: bool = False,
                            speed: float = 1.0,
                            vol: float = 1.0,
                            pitch: float = 0.0,
                            emotion: Optional[str] = None,
                            sample_rate: int = 32000,
                            bitrate: int = 128000,
                            language_boost: Optional[str] = "auto",
                            pronunciation_dict: Optional[Dict] = None) -> Optional[bytes]:
        """使用 MiniMax T2A v2 API 将文本转换为语音
        
        Args:
            text: 要转换的文本内容
            voice_id: 声音ID，如 "male-qn-qingse"
            model: 模型版本，如 "speech-02-hd"、"speech-02-turbo"
            format: 音频格式，支持 "mp3"、"pcm"、"flac"、"wav"(非流式)
            stream: 是否使用流式输出
            speed: 语速，范围一般为0.5-2.0
            vol: 音量，范围一般为0.5-2.0
            pitch: 音调，范围一般为-1.0-1.0
            emotion: 情感，如 "happy"
            sample_rate: 采样率，如 32000, 16000
            bitrate: 比特率，如 128000
            language_boost: 增强特定语言的识别能力
            pronunciation_dict: 发音词典，用于自定义发音
            
        Returns:
            音频数据字节或None（如果发生错误）
        """
        try:
            # 构建请求URL（带有GroupId参数）
            url = f"{self.base_url}?GroupId={self.group_id}"
            
            # 构建请求头
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # 确保所有数值参数是整数
            sample_rate_int = int(sample_rate)
            bitrate_int = int(bitrate)
            
            # 将浮点数参数转换为整数，MiniMax API期望整数类型
            # 根据错误信息，它期望int64类型而不是浮点数
            speed_int = int(speed * 100)  # 转换为整数百分比
            vol_int = int(vol * 100)      # 转换为整数百分比
            pitch_int = int(pitch * 100)  # 转换为整数百分比
            
            # 清理文本，去除潜在的特殊字符和多余的空白
            text = self.clean_text_for_tts(text)
            
            # 文本长度检查和截断
            if len(text) > 2000:
                logger.warning(f"文本过长({len(text)}字符)，将被截断至2000字符")
                text = text[:2000]
                
            # 记录完整的TTS文本内容用于调试
            logger.debug(f"TTS完整文本内容: '{text}' (长度:{len(text)}字符)")
            
            # 构建请求体
            payload = {
                "model": model,
                "text": text,
                "stream": stream,
                "output_format": "hex",  # 默认使用hex编码
                "voice_setting": {
                    "voice_id": voice_id,
                    "speed": speed_int,
                    "vol": vol_int,
                    "pitch": pitch_int
                },
                "audio_setting": {
                    "sample_rate": sample_rate_int,
                    "bitrate": bitrate_int,
                    "format": format,
                    "channel": 1
                }
            }
            
            # 添加可选参数
            if emotion:
                payload["voice_setting"]["emotion"] = emotion
            
            if language_boost:
                payload["language_boost"] = language_boost
                
            if pronunciation_dict:
                payload["pronunciation_dict"] = pronunciation_dict
                
            # 记录最终请求内容检查
            logger.debug(f"发送 MiniMax T2A v2 请求: VoiceID={voice_id}, Model={model}, Format={format}, SampleRate={sample_rate_int}, Bitrate={bitrate_int}, Speed={speed_int}, Vol={vol_int}, Pitch={pitch_int}")
            
            # 检查请求体中的文本是否完整
            payload_text = payload["text"]
            if len(payload_text) != len(text):
                logger.error(f"请求体中的文本长度({len(payload_text)})与原始文本长度({len(text)})不一致，可能发生截断")
                
            # 检查最终请求体的文本内容，看是否与原始文本一致
            logger.debug(f"请求体中的文本内容: '{payload_text[:50]}...{payload_text[-50:] if len(payload_text) > 50 else ''}' (长度:{len(payload_text)}字符)")
            
            # 记录完整的请求体，便于调试
            request_json = json.dumps(payload, ensure_ascii=False)
            logger.debug(f"完整请求体内容长度: {len(request_json)}字节")
            
            # 如果是流式请求，使用不同的处理方式
            if stream:
                logger.warning("流式请求尚未实现，切换为非流式模式")
                payload["stream"] = False
            
            # 发送非流式请求
            response = await asyncio.to_thread(
                lambda: requests.post(url, headers=headers, json=payload)
            )
            
            if response.status_code != 200:
                logger.error(f"MiniMax T2A v2 API错误: {response.status_code} - {response.text}")
                return None
                
            # 解析响应
            result = response.json()
            
            # 检查响应中是否有错误
            if "base_resp" in result and result["base_resp"]["status_code"] != 0:
                error_msg = result["base_resp"]["status_msg"]
                logger.error(f"MiniMax T2A v2 API返回错误: {error_msg}")
                return None
                
            # 从响应中提取音频数据
            if "data" in result and "audio" in result["data"]:
                # 音频数据以十六进制字符串的形式返回
                audio_hex = result["data"]["audio"]
                audio_data = bytes.fromhex(audio_hex)
                
                audio_length = result.get("extra_info", {}).get("audio_length", 0)
                audio_size = result.get("extra_info", {}).get("audio_size", 0)
                text_length = result.get("extra_info", {}).get("text_length", 0)  # 检查服务端实际处理的文本长度
                
                # 检查服务端实际处理的文本长度与我们发送的长度是否一致
                if text_length > 0 and text_length != len(text):
                    logger.warning(f"服务端处理的文本长度({text_length})与发送的文本长度({len(text)})不一致，可能存在截断")
                
                logger.info(f"MiniMax T2A v2 请求成功，音频长度: {audio_length}ms, 大小: {audio_size}字节, 处理文本长度: {text_length}字符")
                
                return audio_data
            else:
                logger.warning("MiniMax T2A v2 API未返回有效的音频数据")
                return None
        
        except Exception as e:
            logger.exception(f"调用 MiniMax T2A v2 API 时发生错误: {e}")
            return None
            
    async def text_to_speech_streaming(self,
                                     text: str,
                                     voice_id: str = "male-qn-qingse",
                                     model: str = "speech-02-turbo",
                                     format: str = "mp3",
                                     **kwargs) -> Optional[bytes]:
        """使用 MiniMax T2A v2 API 的流式模式将文本转换为语音（实验性功能）
        
        返回完整的合并后的音频数据（与非流式相同）
        """
        logger.warning("流式TTS功能尚在实验阶段")
        
        try:
            # 构建请求URL（带有GroupId参数）
            url = f"{self.base_url}?GroupId={self.group_id}"
            
            # 构建请求头
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # 获取并确保所有数值参数都是整数类型
            sample_rate = int(kwargs.get("sample_rate", 32000))
            bitrate = int(kwargs.get("bitrate", 128000))
            
            # 将浮点数参数转换为整数，与非流式方法保持一致
            speed_raw = float(kwargs.get("speed", 1.0))
            vol_raw = float(kwargs.get("vol", 1.0))
            pitch_raw = float(kwargs.get("pitch", 0.0))
            
            # 转换为整数百分比
            speed = int(speed_raw * 100)
            vol = int(vol_raw * 100)
            pitch = int(pitch_raw * 100)
            
            # 清理文本，去除特殊格式符号
            text = self.clean_text_for_tts(text)
            
            # 文本长度检查
            if len(text) > 2000:
                logger.warning(f"流式TTS: 文本过长({len(text)}字符)，将被截断至2000字符")
                text = text[:2000]
                
            # 记录完整的TTS文本内容用于调试
            logger.debug(f"流式TTS完整文本内容: '{text}' (长度:{len(text)}字符)")
            
            # 构建请求体 - 其他参数通过kwargs传入
            payload = {
                "model": model,
                "text": text,
                "stream": True,  # 启用流式
                "voice_setting": {
                    "voice_id": voice_id,
                    "speed": speed,
                    "vol": vol,
                    "pitch": pitch
                },
                "audio_setting": {
                    "sample_rate": sample_rate,
                    "bitrate": bitrate,
                    "format": format,
                    "channel": 1
                }
            }
            
            # 添加可选参数
            if "emotion" in kwargs:
                payload["voice_setting"]["emotion"] = kwargs["emotion"]
            
            if "language_boost" in kwargs:
                payload["language_boost"] = kwargs["language_boost"]
                
            if "pronunciation_dict" in kwargs:
                payload["pronunciation_dict"] = kwargs["pronunciation_dict"]
                
            logger.debug(f"发送流式 MiniMax T2A v2 请求: VoiceID={voice_id}, Model={model}")
            
            # 以流式方式发送请求
            audio_buffer = io.BytesIO()
            
            # 使用requests的流式功能
            response = await asyncio.to_thread(
                lambda: requests.post(url, headers=headers, json=payload, stream=True)
            )
            
            if response.status_code != 200:
                logger.error(f"流式 MiniMax T2A v2 API错误: {response.status_code}")
                return None
                
            # 处理流式响应
            for line in response.iter_lines():
                if line:
                    if line.startswith(b'data:'):
                        try:
                            data = json.loads(line[5:])
                            if "data" in data and "extra_info" not in data:
                                if "audio" in data["data"]:
                                    audio_hex = data["data"]["audio"]
                                    audio_chunk = bytes.fromhex(audio_hex)
                                    audio_buffer.write(audio_chunk)
                        except json.JSONDecodeError as e:
                            logger.warning(f"解析流式响应JSON失败: {e}")
                        except Exception as e:
                            logger.warning(f"处理流式响应时发生错误: {e}")
            
            audio_data = audio_buffer.getvalue()
            
            if not audio_data:
                logger.warning("流式 MiniMax T2A v2 API未返回有效的音频数据")
                return None
                
            logger.info(f"流式 MiniMax T2A v2 请求成功，收到 {len(audio_data)} 字节音频数据")
            return audio_data
            
        except Exception as e:
            logger.exception(f"调用流式 MiniMax T2A v2 API 时发生错误: {e}")
            return None

# --- Keep Placeholder for ChargptLLMClient if needed for compatibility ---
# class ChargptLLMClient:
#     pass 