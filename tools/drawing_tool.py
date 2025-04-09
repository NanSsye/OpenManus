import json
import aiohttp
import asyncio
import time
import os
import re
import uuid
from typing import Dict, Any, Optional
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
        # 先初始化基本属性，稍后再更新工具定义
        self.name = "generate_image"
        self.description = "根据文本描述生成图像"
        self.parameters = {
            "prompt": {
                "type": "string",
                "description": "详细的图像描述，用英文描述效果更好"
            },
            "model": {
                "type": "string",
                "description": "使用的模型，可选: 'default', 'anime', 'realistic', 'rioko', 'custom'",
                "default": "default"
            },
            "ratio": {
                "type": "string",
                "description": "图像比例，可选: '1:1', '4:3', '3:4', '16:9', '9:16'",
                "default": "1:1"
            },
            "lora_model_id": {
                "type": "string",
                "description": "自定义LoRA模型ID，仅在model='custom'时使用",
                "default": ""
            },
            "lora_scale": {
                "type": "number",
                "description": "LoRA模型权重，通常在0.5-1.5之间",
                "default": 1.0
            }
        }

        # 暂时使用基本定义初始化父类
        super().__init__(
            name=self.name,
            description=self.description,
            parameters=self.parameters
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
        self.custom_lora_models = {}  # 自定义LoRA模型字典
        self.default_lora_id = ""     # 默认LoRA模型ID
        self.default_lora_scale = 1.0 # 默认LoRA权重
        self.default_model = "default" # 默认模型
        self.default_ratio = "1:1"    # 默认比例

        # 预先加载模型配置
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
            },
            "rioko": {
                "checkpointModelVersionId": 80,
                "loraArgs": [{"modelVersionId": 48603, "scale": 1}],  # Rioko LoRA模型
                "modelPath": "modelscope://sd1995/lora_rioko?revision=ckpt-24"
            },
            # custom模型会在运行时根据参数动态生成
        }

        # 图像比例配置
        self.ratio_config = {
            "1:1": {"width": 1024, "height": 1024},
            "4:3": {"width": 1152, "height": 864},
            "3:4": {"width": 864, "height": 1152},
            "16:9": {"width": 1280, "height": 720},
            "9:16": {"width": 720, "height": 1280}
        }

        try:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.toml")
            if os.path.exists(config_path):
                # 尝试使用tomllib或者toml库解析
                try:
                    import tomllib
                    with open(config_path, "rb") as f:
                        config = tomllib.load(f)
                except (ImportError, ModuleNotFoundError):
                    try:
                        import toml
                        with open(config_path, "r", encoding="utf-8") as f:
                            config = toml.load(f)
                    except (ImportError, ModuleNotFoundError):
                        # 如果没有toml库，使用简单的解析方式
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

                        # 解析默认LoRA配置
                        lora_id_match = re.search(r'default_lora_id\s*=\s*"([^"]*)"', config_content)
                        if lora_id_match:
                            self.default_lora_id = lora_id_match.group(1)

                        lora_scale_match = re.search(r'default_lora_scale\s*=\s*(\d+\.?\d*)', config_content)
                        if lora_scale_match:
                            self.default_lora_scale = float(lora_scale_match.group(1))

                        # 解析max_wait_time
                        wait_time_match = re.search(r'max_wait_time\s*=\s*(\d+)', config_content)
                        if wait_time_match:
                            self.max_wait_time = int(wait_time_match.group(1))
                        else:
                            self.max_wait_time = max_wait_time

                        # 解析默认模型和比例
                        model_match = re.search(r'default_model\s*=\s*"([^"]*)"', config_content)
                        if model_match:
                            self.default_model = model_match.group(1)

                        ratio_match = re.search(r'default_ratio\s*=\s*"([^"]*)"', config_content)
                        if ratio_match:
                            self.default_ratio = ratio_match.group(1)

                        # 无法解析自定义LoRA模型列表，会使用默认模型配置
                        logger.info(f"使用正则表达式解析配置文件，已加载默认配置：模型={self.default_model}，比例={self.default_ratio}，LoRA ID={self.default_lora_id}，LoRA权重={self.default_lora_scale}")
                        return

                # 如果成功使用tomllib或toml库解析
                drawing_config = config.get("drawing", {})

                # 提取默认设置
                if not self.cookies:
                    self.cookies = drawing_config.get("modelscope_cookies", "")
                if not self.csrf_token:
                    self.csrf_token = drawing_config.get("modelscope_csrf_token", "")

                self.max_wait_time = drawing_config.get("max_wait_time", max_wait_time)
                self.default_lora_id = drawing_config.get("default_lora_id", "48603")
                self.default_lora_scale = drawing_config.get("default_lora_scale", 0.7)
                self.default_model = drawing_config.get("default_model", "rioko")
                self.default_ratio = drawing_config.get("default_ratio", "1:1")

                logger.info(f"从配置文件加载默认设置：模型={self.default_model}，比例={self.default_ratio}，LoRA ID={self.default_lora_id}，LoRA权重={self.default_lora_scale}")

                # 加载自定义LoRA模型列表
                lora_models = drawing_config.get("lora_models", [])
                for model in lora_models:
                    name = model.get("name", "")
                    if name:
                        self.custom_lora_models[name] = {
                            "model_id": model.get("model_id", ""),
                            "model_path": model.get("model_path", ""),
                            "scale": model.get("scale", 0.7)
                        }

                logger.info(f"已加载{len(self.custom_lora_models)}个自定义LoRA模型配置")

                # 使用配置文件中的LoRA默认权重更新预设模型
                default_scale = self.default_lora_scale
                for model_name, model_config in self.model_config.items():
                    if "loraArgs" in model_config and model_config["loraArgs"]:
                        for lora_arg in model_config["loraArgs"]:
                            lora_arg["scale"] = float(default_scale)
                        logger.info(f"已将预设模型 {model_name} 的LoRA权重更新为配置值: {default_scale}")

                # 更新工具定义中的模型描述，包含自定义LoRA模型
                self._update_tool_definition()

        except Exception as e:
            logger.warning(f"无法加载配置: {e}")
            self.max_wait_time = max_wait_time

        # 添加配置文件中的自定义LoRA模型
        for name, model_info in self.custom_lora_models.items():
            if model_info.get("model_id"):
                model_id = model_info["model_id"]
                scale = model_info.get("scale", 0.7)
                model_path = model_info.get("model_path", "")

                # 创建或覆盖模型配置
                self.model_config[name] = {
                    "checkpointModelVersionId": 80,
                    "loraArgs": [{"modelVersionId": int(model_id), "scale": float(scale)}]
                }

                # 如果有模型路径，添加到配置中
                if model_path:
                    if model_path.startswith("modelUrl="):
                        # 直接使用提供的完整modelUrl
                        self.model_config[name]["modelPath"] = model_path
                    else:
                        # 使用标准格式的modelPath
                        self.model_config[name]["modelPath"] = model_path

                logger.info(f"已{'更新' if name in self.model_config else '添加'}自定义LoRA模型: {name}, ID: {model_id}, 权重: {scale}, 路径: {model_path}")

        # 更新工具定义
        self._update_tool_definition()

        # 默认配置
        self.temp_dir = "temp_images"
        os.makedirs(self.temp_dir, exist_ok=True)

    def _update_tool_definition(self):
        """更新工具定义，包含自定义LoRA模型"""
        # 获取所有可用的模型名称，包括预设模型和自定义LoRA模型
        available_models = list(self.model_config.keys())

        # 构建模型描述字符串
        model_description = "'default', 'anime', 'realistic', 'rioko'"

        # 添加自定义LoRA模型
        custom_models = [f"'{name}'" for name in available_models
                         if name not in ['default', 'anime', 'realistic', 'rioko']]
        if custom_models:
            model_description += ", " + ", ".join(custom_models)

        # 始终保留custom选项
        model_description += ", 'custom'"

        # 更新模型参数的描述
        self.parameters["model"]["description"] = f"使用的模型，可选: {model_description}"

        # 重新初始化Tool父类以更新工具定义
        self.__class__.__bases__[0].__init__(
            self,
            name=self.name,
            description=self.description,
            parameters=self.parameters
        )

        logger.info(f"已更新工具定义，模型参数描述: {self.parameters['model']['description']}")

    async def execute(self, prompt: str, model: str = None, ratio: str = None,
                     lora_model_id: str = "", lora_scale: float = 1.0) -> Dict[str, Any]:
        """执行图像生成

        Args:
            prompt: 详细的图像描述
            model: 使用的模型
            ratio: 图像比例
            lora_model_id: 自定义LoRA模型ID
            lora_scale: LoRA模型权重

        Returns:
            Dict: 包含生成图像URL的结果
        """
        try:
            # 使用默认值
            model = model or self.default_model
            ratio = ratio or self.default_ratio

            # 验证参数
            if model not in self.model_config and model != "custom":
                available_models = list(self.model_config.keys()) + ["custom"]
                return {"success": False, "error": f"不支持的模型: {model}，支持的模型: {', '.join(available_models)}"}

            # 如果指定了自定义LoRA模型名称，但该模型存在于配置中，直接使用该模型
            if model in self.model_config and model not in ['default', 'anime', 'realistic', 'rioko', 'custom']:
                logger.info(f"使用自定义LoRA模型: {model}")

            if ratio not in self.ratio_config:
                return {"success": False, "error": f"不支持的图像比例: {ratio}，支持的比例: {', '.join(self.ratio_config.keys())}"}

            # 获取实际的模型参数
            if model == "custom":
                # 未指定LoRA模型ID时，使用默认LoRA
                if not lora_model_id and self.default_lora_id:
                    lora_model_id = self.default_lora_id
                    lora_scale = self.default_lora_scale
                    logger.info(f"使用默认LoRA模型ID: {lora_model_id}, 权重: {lora_scale}")

                if not lora_model_id:
                    return {"success": False, "error": "使用custom模型时必须指定lora_model_id"}

                # 如果权重过高，适当降低以平衡LoRA影响和提示词影响
                if lora_scale > 0.8:
                    logger.info(f"自动调整LoRA权重从 {lora_scale} 到 0.8 以平衡效果")
                    lora_scale = 0.8

                # 尝试提取数值ID
                model_version_id = None
                try:
                    # 如果传入的是完整的模型路径，尝试从中提取ID
                    if "/" in lora_model_id and ":" in lora_model_id:
                        # 可能是类似 sd1995/lora_rioko:ckpt-24 格式
                        model_name_parts = lora_model_id.split(":")
                        if len(model_name_parts) > 1:
                            model_path = model_name_parts[0]  # 使用model_path避免未使用变量的警告
                            logger.debug(f"从模型路径提取的基本路径: {model_path}")
                            # 尝试在参数中查找versionId
                            match = re.search(r'versionId=(\d+)', lora_model_id)
                            if match:
                                model_version_id = int(match.group(1))
                    # 如果直接传入数字ID
                    elif lora_model_id.isdigit():
                        model_version_id = int(lora_model_id)
                except Exception as e:
                    logger.warning(f"解析模型ID失败: {e}, 将使用原始ID: {lora_model_id}")

                # 如果无法解析，使用原始ID
                if not model_version_id:
                    return {"success": False, "error": f"无法解析LoRA模型ID: {lora_model_id}, 请提供有效的模型版本ID"}

                # 创建自定义模型配置
                model_args = {
                    "checkpointModelVersionId": 80,  # FLUX.1 模型
                    "loraArgs": [{"modelVersionId": model_version_id, "scale": float(lora_scale)}]
                }
            else:
                model_args = self.model_config[model]

                # 如果是预设模型，但用户指定了LoRA权重，应用用户的设置
                if lora_scale != 1.0 and "loraArgs" in model_args and model_args["loraArgs"]:
                    logger.info(f"为预设模型 {model} 应用自定义LoRA权重: {lora_scale}")
                    for lora_arg in model_args["loraArgs"]:
                        lora_arg["scale"] = float(lora_scale)

            dimensions = self.ratio_config[ratio]

            model_description = f"custom LoRA({lora_model_id})" if model == "custom" else model
            logger.info(f"使用{model_description}模型生成图像，比例: {ratio}，提示词: {prompt}")

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
                "model": model_description,
                "lora_info": f"ID: {lora_model_id}, Scale: {lora_scale}" if model == "custom" else "",
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
                # 添加一个通用的负面提示词，避免模型过度依赖LoRA
                negative_prompt = "bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, text, watermark, signature, artist name, logo"

                data = {
                    "modelArgs": {
                        "checkpointModelVersionId": model_args.get("checkpointModelVersionId", 80),
                        "loraArgs": model_args.get("loraArgs", [])
                    },
                    "basicDiffusionArgs": {
                        "sampler": "DPM++ 2M Karras",
                        "guidanceScale": 7.0,  # 提高引导比例以更好地遵循提示词
                        "seed": -1,  # 随机种子
                        "numInferenceSteps": 30,
                        "height": height,
                        "width": width,
                        "numImagesPerPrompt": 1
                    },
                    "controlNetFullArgs": [],
                    "hiresFixFrontArgs": None,
                    "predictType": "TXT_2_IMG",
                    "promptArgs": {
                        "prompt": prompt + ", high quality, detailed, best quality",  # 增强提示词
                        "negativePrompt": negative_prompt
                    }
                }

                # 添加modelPath信息（如果有）
                if "modelPath" in model_args:
                    data["modelPath"] = model_args["modelPath"]
                    logger.info(f"使用完整模型路径: {model_args['modelPath']}")

                # 记录完整请求，方便调试
                logger.debug(f"提交任务请求: {json.dumps(data, ensure_ascii=False)}")

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

    async def send_generated_image(self, bot, target_id: str, image_url: str, at_list = None) -> bool:
        """下载并发送生成的图片

        Args:
            bot: WechatAPIClient实例
            target_id: 目标ID (群ID或用户ID)
            image_url: 图片URL
            at_list: @用户列表 (可选)

        Returns:
            bool: 是否成功发送
        """
        # 如果提供了at_list，记录一下
        if at_list:
            logger.debug(f"将在发送图片时@用户: {at_list}")
        try:
            # 下载图片
            local_path = await self.download_image(image_url)
            if not local_path:
                logger.error("下载图片失败，无法发送")
                await bot.send_text_message(target_id, "抱歉，下载生成的图片失败")
                return False

            # 发送图片
            try:
                # 先读取图片数据到内存
                with open(local_path, "rb") as f:
                    image_data = f.read()

                logger.info(f"准备发送图片，大小: {len(image_data)} 字节")
                success = False
                used_method = "unknown"  # 使用新变量避免警告

                # 检查是否有send_image_message方法
                if hasattr(bot, 'send_image_message'):
                    used_method = 'send_image_message'
                    # 直接传递图片数据而非路径
                    await bot.send_image_message(target_id, image=image_data)
                    success = True
                elif hasattr(bot, 'SendImageMessage'):
                    used_method = 'SendImageMessage'
                    # 直接传递图片数据而非路径
                    await bot.SendImageMessage(target_id, image=image_data)
                    success = True

                # 记录使用的方法
                if success:
                    logger.debug(f"使用方法 {used_method} 发送图片成功")
                else:
                    logger.error("API不支持发送图片消息")
                    await bot.send_text_message(target_id, f"图片已生成，请访问链接查看: {image_url}")

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
                await bot.send_text_message(target_id, f"图片发送失败，请访问链接查看: {image_url}")
                return False

        except Exception as e:
            logger.exception(f"处理和发送图片异常: {e}")
            await bot.send_text_message(target_id, f"处理图片时出错，请访问链接查看: {image_url}")
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