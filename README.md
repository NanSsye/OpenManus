# OpenManus 智能代理插件

OpenManus 是一个增强型智能代理插件，基于 Gemini 大型语言模型实现多步骤思考和工具使用，为微信机器人提供强大的智能助手功能，支持上下文对话和语音回复。

**作者：老夏的金库**

## 功能特点

- **多步骤思考**: 使用 MCP (多步认知过程) 方法进行复杂问题分析
- **工具集成**: 支持多种工具调用，包括计算器、日期时间、搜索等
- **语音合成**: 支持 Fish Audio 和 MiniMax T2A v2 双引擎语音合成
- **丰富交互**: 支持私聊和群聊，可通过触发词或@方式激活
- **灵活配置**: 支持多种配置选项，包括模型选择、语音参数、工具启用等

## 安装方法

1. 确保已安装必要的依赖:

   ```
   pip install -r requirements.txt
   ```

2. 将 OpenManus 插件目录放置在机器人的 plugins 目录下
3. 编辑配置文件 `config.toml`，设置你的 API 密钥和其他选项
4. 重启机器人服务

## 配置说明

配置文件位于 `plugins/OpenManus/config.toml`，主要配置项如下:

```toml
[basic]
# 插件基本配置
enable = true                 # 是否启用插件
trigger_keyword = "agent"     # 触发词，用于命令识别
allow_private_chat = true     # 是否允许私聊使用
respond_to_at = true          # 是否允许群里@使用

[gemini]
# Gemini API配置
api_key = ""                  # Gemini API密钥
base_url = "https://generativelanguage.googleapis.com/v1beta"  # API基础URL
use_sdk = true                # 是否使用官方SDK（必须为true才能启用上下文对话）

[agent]
# 代理配置
default_model = "gemini-2.0-flash"  # 默认使用的模型
max_tokens = 8192             # 最大生成token数
temperature = 0.7             # 温度参数(0-2)
max_steps = 10                # 最大执行步骤数

[mcp]
# MCP配置
enable_mcp = true             # 是否启用MCP思考
thinking_steps = 3            # 思考步骤数

[memory]
# 对话记忆配置
enable_memory = true          # 是否启用记忆功能
max_history = 10              # 最大历史记录条数
separate_context = true       # 是否为不同会话维护单独的上下文

[tools]
# 工具配置
enable_search = true          # 是否启用搜索工具
enable_calculator = true      # 是否启用计算器工具
enable_datetime = true        # 是否启用日期时间工具
enable_weather = true         # 是否启用天气工具
bing_api_key = ""             # Bing搜索API密钥
serper_api_key = ""           # Serper搜索API密钥
search_engine = "serper"      # 搜索引擎选择(bing或serper)

[tts]
# Fish Audio TTS配置
enable = false                # 是否启用Fish Audio TTS
api_key = ""                  # Fish Audio API密钥
reference_id = ""             # 自定义模型ID
format = "mp3"                # 输出格式
mp3_bitrate = 128             # MP3比特率

[minimax_tts]
# MiniMax TTS配置
enable = false                # 是否启用MiniMax TTS
api_key = ""                  # MiniMax API密钥
group_id = ""                 # MiniMax Group ID
model = "speech-02-hd"        # 模型版本
voice_id = "male-qn-qingse"   # 声音ID
format = "mp3"                # 输出格式
sample_rate = 32000           # 采样率
bitrate = 128000              # 比特率
speed = 1.0                   # 语速(0.5-2.0)
vol = 1.0                     # 音量(0.5-2.0)
pitch = 0.0                   # 音调(-1.0-1.0)
emotion = "neutral"           # 情感("happy", "sad", "angry", 等)
language_boost = "auto"       # 语言增强
```

## 使用方法

### 私聊模式

直接向机器人发送以触发词开头的消息:

```
agent 计算 (5+3)*2
```

### 群聊模式

两种方式:

1. @机器人 并输入问题
2. 以触发词开头:
   ```
   agent 查询最新比特币价格
   ```

### 上下文对话

系统会自动记住对话历史，要清除历史可发送:

```
agent 清除对话
```

或

```
agent 清除记忆
```

## 支持的工具

1. **计算器**: 执行数学计算

   - 用法: `agent 计算 <表达式>`
   - 示例: `agent 计算 sin(0.5)*5+sqrt(16)`

2. **日期时间**: 获取日期时间信息

   - 用法: `agent 日期 [操作]`
   - 示例: `agent 查询两周后是什么日期`

3. **搜索**: 执行网络搜索(需配置 API 密钥)

   - 用法: `agent 搜索 <关键词>`
   - 示例: `agent 搜索 2023年经济增长率`

4. **天气**: 获取天气信息(需配置 API 密钥)
   - 用法: `agent 天气 <城市>`
   - 示例: `agent 查询北京天气`
   - 高级示例: `agent 查询江西南昌的天气状况`
   - API 来源: ALAPI 天气接口，提供全面的天气数据，包括天气状况、温度、湿度、风力、空气质量和生活指数等

## 语音合成功能

OpenManus 支持两种语音合成引擎：

1. **Fish Audio TTS**:

   - 基于自定义音色的高质量语音合成
   - 需要配置 API 密钥和 Reference ID
   - 支持 mp3/wav/pcm 格式输出

2. **MiniMax T2A v2**:
   - 支持 100+系统音色和复刻音色
   - 提供情感控制、语速、音量等调整
   - 支持 mp3/pcm/flac/wav 格式输出
   - 可调整采样率、比特率等音频参数

当同时启用两种 TTS 引擎时，系统会优先使用 MiniMax TTS，如果失败则回退到 Fish Audio TTS。

## 开发文档

### 项目结构

```
plugins/OpenManus/
├── config.toml         # 配置文件
├── main.py             # 插件入口
├── api_client.py       # API客户端
├── README.md           # 说明文档
├── requirements.txt    # 依赖列表
├── agent/
│   └── mcp.py          # MCP代理实现
└── tools/
    └── basic_tools.py  # 基础工具实现
```

### 扩展开发

#### 添加新工具

要添加新工具，可以在`tools`目录下创建新的工具类并继承`Tool`基类:

```python
from ..agent.mcp import Tool

class MyNewTool(Tool):
    def __init__(self):
        super().__init__(
            name="my_tool",
            description="我的新工具",
            parameters={
                "param1": {
                    "type": "string",
                    "description": "参数1"
                }
            }
        )

    async def execute(self, param1: str) -> Dict[str, Any]:
        # 实现工具功能
        return {"result": f"处理结果: {param1}"}
```

然后在`main.py`的`_create_and_register_agent`方法中注册该工具。

#### 自定义工具开发指南

OpenManus 支持通过工具系统扩展 AI 助手的能力。以下是开发自定义工具的详细流程：

##### 1. 工具基类说明

所有工具必须继承自 `Tool` 基类：

```python
from plugins.OpenManus.agent.mcp import Tool

class MyCustomTool(Tool):
    def __init__(self):
        super().__init__(
            name="my_custom_tool",      # 工具名称，必须唯一
            description="这个工具用于实现某项特定功能",  # 工具说明，AI将根据此决定何时使用工具
            parameters={                # 参数定义，遵循JSON Schema规范
                "param1": {
                    "type": "string",
                    "description": "参数1的说明"
                },
                "param2": {
                    "type": "integer",
                    "description": "参数2的说明",
                    "default": 10       # 包含default表示此参数可选
                }
            },
            required=["param1"]         # 必填参数列表（可选，若不提供则自动根据是否有default判断）
        )

    async def execute(self, **kwargs):
        """实现工具逻辑，必须是异步方法"""
        # 获取参数
        param1 = kwargs.get("param1")
        param2 = kwargs.get("param2", 10)

        # 执行操作
        result = await self._do_something(param1, param2)

        # 返回结果（必须是字典格式）
        return {
            "result": result,
            "status": "success"
        }

    async def _do_something(self, param1, param2):
        """实现具体功能的辅助方法"""
        # 具体实现...
        return f"处理结果: {param1}, {param2}"
```

##### 2. 参数类型支持

参数定义支持以下类型：

- `string`：字符串
- `integer`：整数
- `number`：浮点数
- `boolean`：布尔值
- `array`：数组，可通过 `items` 定义元素类型
- `object`：对象，可通过 `properties` 定义属性

示例：

```python
parameters={
    "text": {
        "type": "string",
        "description": "要处理的文本"
    },
    "options": {
        "type": "array",
        "items": {"type": "string"},
        "description": "处理选项列表"
    },
    "config": {
        "type": "object",
        "properties": {
            "mode": {"type": "string"},
            "level": {"type": "integer"}
        },
        "description": "配置对象"
    }
}
```

##### 3. 配置集成

为使工具可配置，建议遵循以下步骤：

1. 在 `config.toml` 中添加工具配置：

```toml
[tools]
# 现有配置...
enable_my_custom_tool = true  # 是否启用自定义工具

[my_custom_tool]
# 工具特定配置
api_key = ""  # 如果需要API密钥
base_url = ""  # 如果需要API地址
timeout = 30   # 其他配置参数
```

2. 在 `main.py` 的 `__init__` 方法中读取配置：

```python
def __init__(self, config_path: str = "plugins/OpenManus/config.toml"):
    # ... 现有代码 ...

    # 读取自定义工具配置
    custom_tool_config = self.config.get("my_custom_tool", {})
    self.enable_my_custom_tool = tools_config.get("enable_my_custom_tool", False)
    self.custom_tool_api_key = custom_tool_config.get("api_key", "")
    self.custom_tool_base_url = custom_tool_config.get("base_url", "")
    self.custom_tool_timeout = custom_tool_config.get("timeout", 30)
```

3. 在 `_create_and_register_agent` 方法中注册工具：

```python
def _create_and_register_agent(self):
    # ... 现有代码 ...

    # 添加自定义工具
    if self.enable_my_custom_tool and self.custom_tool_api_key:
        tools.append(MyCustomTool(
            api_key=self.custom_tool_api_key,
            base_url=self.custom_tool_base_url,
            timeout=self.custom_tool_timeout
        ))
```

##### 4. 实现工具功能

在 `execute` 方法中实现工具核心功能：

1. **API 调用工具**：

```python
async def execute(self, **kwargs):
    """执行API调用"""
    query = kwargs.get("query")

    # 构建API请求
    headers = {"Authorization": f"Bearer {self.api_key}"}
    params = {"q": query}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/search",
                headers=headers,
                params=params,
                timeout=self.timeout
            ) as response:

                if response.status != 200:
                    return {"error": f"API返回错误: {response.status}"}

                data = await response.json()
                return {"results": data.get("results", [])}

    except Exception as e:
        return {"error": f"API调用失败: {str(e)}"}
```

2. **本地处理工具**：

```python
async def execute(self, **kwargs):
    """本地数据处理"""
    text = kwargs.get("text")
    mode = kwargs.get("mode", "default")

    # 根据模式执行不同处理
    if mode == "analyze":
        result = self._analyze_text(text)
    elif mode == "transform":
        result = self._transform_text(text)
    else:
        result = text

    return {"processed_text": result}
```

##### 5. 错误处理

工具应当妥善处理异常，避免整个系统因工具错误而崩溃：

```python
async def execute(self, **kwargs):
    try:
        # 工具逻辑...
        return {"result": result}
    except ValueError as ve:
        # 参数错误
        return {"error": f"参数错误: {str(ve)}"}
    except aiohttp.ClientError as ce:
        # 网络错误
        return {"error": f"API连接错误: {str(ce)}"}
    except Exception as e:
        # 其他错误
        return {"error": f"工具执行异常: {str(e)}"}
```

##### 6. 工具开发最佳实践

1. **清晰的描述**：提供准确的工具描述和参数说明，帮助 AI 正确选择和使用工具
2. **必要的验证**：在工具执行前验证参数的有效性
3. **合理的超时**：为网络请求设置适当的超时时间
4. **详细的日志**：记录工具执行的关键步骤和结果
5. **优雅的失败**：即使出错也返回有用的错误信息，而非抛出异常
6. **可配置性**：通过配置文件使工具行为可调整
7. **测试**：编写单元测试确保工具在各种情况下正常工作

##### 7. 工具类型示例

常见工具类型及示例：

1. **搜索工具**：从网络搜索信息
2. **翻译工具**：调用翻译 API 进行语言转换
3. **数据库工具**：查询本地数据库
4. **文档处理工具**：分析、生成或修改文档
5. **媒体处理工具**：处理图像、音频或视频
6. **IoT 控制工具**：与智能家居或其他设备交互
7. **AI 服务集成**：调用其他 AI 服务如图像识别

通过开发自定义工具，您可以显著扩展 OpenManus 的能力，使其适应各种特定场景和需求。

## 依赖要求

所需的依赖项已在项目根目录的`requirements.txt`文件中列出：

```
# 核心依赖
aiohttp>=3.8.4
tomli>=2.0.1
loguru>=0.6.0
pydub>=0.25.1
aiofiles>=23.1.0


# 音频处理依赖
fish-audio-sdk>=1.0.0  # Fish Audio TTS（可选）
requests>=2.28.0       # MiniMax TTS使用

# 工具依赖
python-dateutil>=2.8.2  # 日期时间工具
```

## 常见问题

### API 密钥问题

如果遇到 API 连接问题，请检查:

1. API 密钥是否正确设置
2. 网络连接是否正常
3. API 基础 URL 是否需要修改

对于 ALAPI 天气接口:

1. 需要在[ALAPI 官网](https://alapi.cn)注册账号
2. 创建 API Token 并获取密钥
3. 将密钥填入`config.toml`的`weather_api_key`字段



### 语音合成问题

如果语音合成不工作，请检查:

1. 相应 TTS 引擎的`enable`是否设置为`true`
2. API 密钥和其他必要参数是否已正确配置
3. 音频参数是否设置为合理的值

### 性能优化

如果遇到响应慢的问题:

1. 减少`thinking_steps`的值
2. 使用更快的模型（如`gemini-2.0-flash`）
3. 减少`max_tokens`值

## 版权和许可

OpenManus 插件使用 MIT 许可证。更多信息请参见 LICENSE 文件。

**作者：老夏的金库**
