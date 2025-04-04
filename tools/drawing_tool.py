import json
import aiohttp
import asyncio
import base64
import time
import os
import re
import uuid
import tempfile
from typing import Dict, Any, Optional, List
from loguru import logger

from ..agent.mcp import Tool

class ModelScopeDrawingTool(Tool):
    """使用ModelScope模型生成图像的工具"""
    
    def __init__(self, api_base: str = "https://www.modelscope.cn/api/v1/muse/predict",
                 cookies: str = None, csrf_token: str = None, max_wait_time: int = 60):
        """初始化ModelScope绘画工具
        
        Args:
            api_base: ModelScope API 基础URL
            cookies: ModelScope网站Cookie字符串
            csrf_token: ModelScope网站CSRF Token
            max_wait_time: 最大等待时间(秒)
        """
        super().__init__(
            name="generate_image",
            description="根据文本描述生成图像",
            parameters={
                "prompt": {
                    "type": "string",
                    "description": "详细的图像描述，用英文描述效果更好"
                },
                "model": {
                    "type": "string",
                    "description": "使用的模型，可选: 'default', 'anime', 'realistic'",
                    "default": "default"
                },
                "ratio": {
                    "type": "string",
                    "description": "图像比例，可选: '1:1', '4:3', '3:4', '16:9', '9:16'",
                    "default": "1:1"
                }
            }
        )
        self.api_base = api_base.rstrip('/')
        self.submit_url = f"{self.api_base}/task/submit"
        self.status_url = f"{self.api_base}/task/status"
        self.cookies = cookies
        self.csrf_token = csrf_token
        
        # 尝试从环境变量获取Cookie和CSRF Token（如果未直接提供）
        if not self.cookies:
            self.cookies = os.environ.get("MODELSCOPE_COOKIES", "")
        if not self.csrf_token:
            self.csrf_token = os.environ.get("MODELSCOPE_CSRF_TOKEN", "")
        
        # 解析配置文件
        try:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.toml")
            if os.path.exists(config_path):
                # 避免依赖toml库，使用简单的解析方式获取值
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_content = f.read()
                    
                # 解析cookies和csrf_token
                if not self.cookies:
                    cookies_match = re.search(r'modelscope_cookies\s*=\s*"([^"]*)"', config_content)
                    if cookies_match:
                        self.cookies = cookies_match.group(1)
                
                if not self.csrf_token:
                    token_match = re.search(r'modelscope_csrf_token\s*=\s*"([^"]*)"', config_content)
                    if token_match:
                        self.csrf_token = token_match.group(1)
                
                # 解析max_wait_time
                wait_time_match = re.search(r'max_wait_time\s*=\s*(\d+)', config_content)
                if wait_time_match:
                    self.max_wait_time = int(wait_time_match.group(1))
                else:
                    self.max_wait_time = max_wait_time
        except Exception as e:
            logger.warning(f"无法加载配置: {e}")
            self.max_wait_time = max_wait_time
        
        # 模型映射配置
        self.model_config = {
            "default": {
                "checkpointModelVersionId": 80,  # FLUX.1 模型
                "loraArgs": []
            },
            "anime": {
                "checkpointModelVersionId": 80,
                "loraArgs": [{"modelVersionId": 48603, "scale": 1}]  # anime风格LoRA
            },
            "realistic": {
                "checkpointModelVersionId": 80,
                "loraArgs": [{"modelVersionId": 47474, "scale": 1}]  # 写实风格LoRA
            }
        }
        
        # 图像比例配置
        self.ratio_config = {
            "1:1": {"width": 1024, "height": 1024},
            "4:3": {"width": 1152, "height": 864},
            "3:4": {"width": 864, "height": 1152},
            "16:9": {"width": 1280, "height": 720},
            "9:16": {"width": 720, "height": 1280}
        }
        
        # 默认配置
        self.default_model = "default"
        self.default_ratio = "1:1"
        self.temp_dir = "temp_images"
        os.makedirs(self.temp_dir, exist_ok=True)
        
    async def execute(self, prompt: str, model: str = None, ratio: str = None) -> Dict[str, Any]:
        """执行图像生成
        
        Args:
            prompt: 详细的图像描述
            model: 使用的模型
            ratio: 图像比例
            
        Returns:
            Dict: 包含生成图像URL的结果
        """
        try:
            # 使用默认值
            model = model or self.default_model
            ratio = ratio or self.default_ratio
            
            # 验证参数
            if model not in self.model_config:
                return {"success": False, "error": f"不支持的模型: {model}，支持的模型: {', '.join(self.model_config.keys())}"}
            
            if ratio not in self.ratio_config:
                return {"success": False, "error": f"不支持的图像比例: {ratio}，支持的比例: {', '.join(self.ratio_config.keys())}"}
            
            # 获取实际的模型ID和尺寸
            model_args = self.model_config[model]
            dimensions = self.ratio_config[ratio]
            
            logger.info(f"使用{model}模型生成图像，比例: {ratio}，提示词: {prompt}")
            
            # 提交任务
            task_id = await self._submit_task(model_args, prompt, dimensions["width"], dimensions["height"])
            if not task_id:
                return {"success": False, "error": "提交任务失败"}
            
            logger.info(f"任务提交成功，任务ID: {task_id}")
            
            # 等待结果
            result = await self._wait_for_result(task_id)
            if not result:
                return {"success": False, "error": "生成图像超时或失败"}
            
            # 返回结果
            return {
                "success": True,
                "image_url": result.get("image_url", ""),
                "task_id": task_id,
                "prompt": prompt,
                "model": model,
                "width": dimensions["width"],
                "height": dimensions["height"]
            }
            
        except Exception as e:
            logger.exception(f"图像生成失败: {e}")
            return {"success": False, "error": f"图像生成失败: {str(e)}"}
    
    async def _submit_task(self, model_args: Dict, prompt: str, width: int, height: int) -> Optional[str]:
        """提交图像生成任务
        
        Args:
            model_args: 模型参数
            prompt: 提示词
            width: 图像宽度
            height: 图像高度
            
        Returns:
            Optional[str]: 任务ID，如果失败则返回None
        """
        try:
            async with aiohttp.ClientSession() as session:
                # 构建完整的请求头，包含认证信息
                headers = {
                    "Content-Type": "application/json",
                    "x-modelscope-accept-language": "zh_CN",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
                    "Origin": "https://www.modelscope.cn",
                    "Referer": "https://www.modelscope.cn/aigc/imageGeneration",
                    "Sec-Fetch-Dest": "empty",
                    "Sec-Fetch-Mode": "cors",
                    "Sec-Fetch-Site": "same-origin"
                }
                
                # 添加CSRF令牌和Cookie（如果提供）
                if self.csrf_token:
                    headers["x-csrf-token"] = self.csrf_token
                
                cookies = {}
                if self.cookies:
                    # 简单解析cookie字符串转成字典
                    cookie_parts = self.cookies.split(';')
                    for part in cookie_parts:
                        if '=' in part:
                            name, value = part.strip().split('=', 1)
                            cookies[name] = value
                
                # 准备请求数据
                data = {
                    "modelArgs": model_args,
                    "basicDiffusionArgs": {
                        "sampler": "DPM++ 2M Karras",
                        "guidanceScale": 3.5,
                        "seed": -1,
                        "numInferenceSteps": 30,
                        "height": height,
                        "width": width,
                        "numImagesPerPrompt": 1
                    },
                    "controlNetFullArgs": [],
                    "hiresFixFrontArgs": None,
                    "predictType": "TXT_2_IMG",
                    "promptArgs": {
                        "prompt": prompt,
                        "negativePrompt": ""
                    }
                }
                
                async with session.post(self.submit_url, json=data, headers=headers, cookies=cookies) as response:
                    if response.status != 200:
                        logger.error(f"任务提交失败，状态码: {response.status}")
                        return None
                        
                    response_data = await response.json()
                    if not response_data.get("Success"):
                        logger.error(f"任务提交响应错误: {response_data}")
                        return None
                        
                    task_id = response_data.get("Data", {}).get("data", {}).get("taskId")
                    return task_id
        except Exception as e:
            logger.exception(f"提交任务异常: {e}")
            return None
    
    async def _wait_for_result(self, task_id: str) -> Optional[Dict]:
        """等待任务完成并获取结果
        
        Args:
            task_id: 任务ID
            
        Returns:
            Optional[Dict]: 生成的图像URL，如果失败则返回None
        """
        start_time = time.time()
        max_wait_time = self.max_wait_time  # 使用实例变量
        
        try:
            async with aiohttp.ClientSession() as session:
                # 构建完整的请求头，包含认证信息
                headers = {
                    "Content-Type": "application/json",
                    "x-modelscope-accept-language": "zh_CN",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
                    "Origin": "https://www.modelscope.cn",
                    "Referer": "https://www.modelscope.cn/aigc/imageGeneration",
                    "Sec-Fetch-Dest": "empty",
                    "Sec-Fetch-Mode": "cors",
                    "Sec-Fetch-Site": "same-origin"
                }
                
                # 添加CSRF令牌和Cookie（如果提供）
                if self.csrf_token:
                    headers["x-csrf-token"] = self.csrf_token
                
                cookies = {}
                if self.cookies:
                    # 简单解析cookie字符串转成字典
                    cookie_parts = self.cookies.split(';')
                    for part in cookie_parts:
                        if '=' in part:
                            name, value = part.strip().split('=', 1)
                            cookies[name] = value
                
                while (time.time() - start_time) < max_wait_time:
                    # 查询任务状态
                    status_url = f"{self.status_url}?taskId={task_id}"
                    
                    async with session.get(status_url, headers=headers, cookies=cookies) as response:
                        if response.status != 200:
                            logger.error(f"获取任务状态失败，状态码: {response.status}")
                            await asyncio.sleep(2)
                            continue
                            
                        response_data = await response.json()
                        if not response_data.get("Success"):
                            logger.error(f"获取任务状态响应错误: {response_data}")
                            await asyncio.sleep(2)
                            continue
                        
                        # 获取任务状态
                        status = response_data.get("Data", {}).get("data", {}).get("status")
                        
                        if status == "SUCCEED":
                            # 任务成功，获取图像URL
                            image_data = response_data.get("Data", {}).get("data", {}).get("predictResult", {}).get("images", [])
                            if image_data and len(image_data) > 0:
                                return {"image_url": image_data[0].get("imageUrl")}
                            else:
                                logger.error("任务成功但未找到图像URL")
                                return None
                        elif status == "FAILED":
                            # 任务失败
                            error_msg = response_data.get("Data", {}).get("data", {}).get("errorMsg")
                            logger.error(f"任务失败: {error_msg}")
                            return None
                        else:
                            # 任务还在处理中，等待并继续查询
                            logger.info(f"任务正在处理中，状态: {status}")
                            await asyncio.sleep(2)
                
                # 超时
                logger.warning(f"等待任务超时，已等待 {max_wait_time} 秒")
                return None
        except Exception as e:
            logger.exception(f"等待任务结果异常: {e}")
            return None
    
    async def download_image(self, image_url: str) -> Optional[str]:
        """下载图片并保存到本地
        
        Args:
            image_url: 图片URL
            
        Returns:
            Optional[str]: 本地图片路径
        """
        try:
            # 创建唯一的文件名
            filename = f"{int(asyncio.get_event_loop().time())}_{uuid.uuid4().hex[:8]}.png"
            local_path = os.path.join(self.temp_dir, filename)
            
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    if response.status != 200:
                        logger.error(f"下载图片失败，状态码: {response.status}")
                        return None
                    
                    # 读取图片内容并保存
                    image_data = await response.read()
                    with open(local_path, "wb") as f:
                        f.write(image_data)
                    
                    logger.info(f"图片已下载到: {local_path}")
                    return local_path
        
        except Exception as e:
            logger.exception(f"下载图片异常: {e}")
            return None
            
    async def send_generated_image(self, bot, target_id: str, image_url: str, at_list: List[str] = None) -> bool:
        """下载并发送生成的图片
        
        Args:
            bot: WechatAPIClient 实例
            target_id: 目标ID (群ID或用户ID)
            image_url: 图片URL
            at_list: @用户列表 (可选)
            
        Returns:
            bool: 是否成功发送
        """
        try:
            # 下载图片
            local_path = await self.download_image(image_url)
            if not local_path:
                logger.error("下载图片失败，无法发送")
                await bot.send_at_message(target_id, "抱歉，下载生成的图片失败", at_list)
                return False
            
            # 发送图片
            try:
                # 检查是否有send_image_message方法
                if hasattr(bot, 'send_image_message'):
                    send_result = await bot.send_image_message(target_id, local_path)
                    success = True
                elif hasattr(bot, 'SendImageMessage'):
                    send_result = await bot.SendImageMessage(target_id, local_path)
                    success = True
                else:
                    logger.error("API不支持发送图片消息")
                    await bot.send_at_message(target_id, f"图片已生成，请访问链接查看: {image_url}", at_list)
                    success = False
                
                # 清理临时文件
                try:
                    os.remove(local_path)
                    logger.debug(f"临时图片文件已清理: {local_path}")
                except Exception as e:
                    logger.warning(f"清理临时图片文件失败: {e}")
                
                return success
                
            except Exception as e:
                logger.exception(f"发送图片异常: {e}")
                # 发送失败时，发送图片链接
                await bot.send_at_message(target_id, f"图片发送失败，请访问链接查看: {image_url}", at_list)
                return False
                
        except Exception as e:
            logger.exception(f"处理和发送图片异常: {e}")
            await bot.send_at_message(target_id, f"处理图片时出错，请访问链接查看: {image_url}", at_list)
            return False
    
    async def download_specific_image(self, image_url: str) -> Optional[str]:
        """下载特定URL的图片（用于已生成的图片链接）
        
        Args:
            image_url: 图片URL
            
        Returns:
            Optional[str]: 本地图片路径
        """
        try:
            logger.info(f"【绘图工具】开始下载图片: {image_url}")
            
            # 创建唯一的文件名
            filename = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.png"
            local_path = os.path.join(self.temp_dir, filename)
            
            # 确保目录存在
            os.makedirs(self.temp_dir, exist_ok=True)
            logger.debug(f"【绘图工具】临时目录已确认: {self.temp_dir}")
            
            # 下载图片
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
            }
            
            logger.debug(f"【绘图工具】准备发送HTTP请求下载图片")
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url, headers=headers) as response:
                    status = response.status
                    logger.debug(f"【绘图工具】收到图片下载响应，状态码: {status}")
                    
                    if status != 200:
                        logger.error(f"【绘图工具】下载图片失败，状态码: {status}")
                        return None
                    
                    # 读取图片内容并保存
                    image_data = await response.read()
                    data_length = len(image_data)
                    logger.debug(f"【绘图工具】成功读取图片数据，大小: {data_length} 字节")
                    
                    if data_length < 100:
                        logger.error(f"【绘图工具】图片数据异常，太小: {data_length} 字节")
                        return None
                    
                    try:
                        with open(local_path, "wb") as f:
                            f.write(image_data)
                        logger.info(f"【绘图工具】图片已成功保存到: {local_path}")
                    except Exception as write_err:
                        logger.exception(f"【绘图工具】保存图片到本地时出错: {write_err}")
                        return None
                    
                    # 验证文件是否已保存
                    if os.path.exists(local_path) and os.path.getsize(local_path) > 100:
                        logger.info(f"【绘图工具】图片文件已验证: {local_path}, 大小: {os.path.getsize(local_path)} 字节")
                        return local_path
                    else:
                        logger.error(f"【绘图工具】图片文件验证失败，可能未保存成功或文件过小")
                        return None
        
        except Exception as e:
            logger.exception(f"【绘图工具】下载特定图片异常: {e}")
            return None 