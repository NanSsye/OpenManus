# OpenManus 智能代理插件

OpenManus 是一个增强型智能代理插件，基于大型语言模型实现多步骤思考和工具使用，为微信机器人提供强大的智能助手功能。

## 功能特点

- **多步骤思考**: 使用 MCP (多步认知过程) 方法进行复杂问题分析
- **工具集成**: 支持多种工具调用，包括计算器、日期时间、搜索等
- **丰富交互**: 支持私聊和群聊，可通过触发词或@方式激活
- **灵活配置**: 支持多种配置选项，包括模型选择、工具启用等

## 安装方法

1. 确保已安装必要的依赖:

   ```
   pip install aiohttp tomli loguru
   ```

2. 将 OpenManus 插件目录放置在机器人的 plugins 目录下
3. 编辑配置文件 `config.toml`，设置你的 API 密钥和其他选项
4. 重启机器人服务

## 配置说明

配置文件位于 `plugins/OpenManus/config.toml`，主要配置项如下:

```toml
[basic]
# 插件基本配置
enabled = true                # 是否启用插件
trigger_word = "agent"        # 触发词，用于命令识别
private_chat_enabled = true   # 是否允许私聊使用
group_at_enabled = true       # 是否允许群里@使用

[api]
# API配置
openai_api_key = ""           # OpenAI API密钥
openai_api_base = "https://api.openai.com/v1"  # API基础URL

[agent]
# 代理配置
default_model = "gpt-4o"      # 默认使用的模型
max_tokens = 4096             # 最大生成token数
temperature = 0.0             # 温度参数(0-1)
max_steps = 5                 # 最大执行步骤数

[mcp]
# MCP配置
enable = true                 # 是否启用MCP思考
thinking_steps = 3            # 思考步骤数

[tools]
# 工具配置
enable_search = true          # 是否启用搜索工具
enable_calculator = true      # 是否启用计算器工具
enable_datetime = true        # 是否启用日期时间工具
enable_weather = false        # 是否启用天气工具
bing_api_key = ""             # Bing搜索API密钥(可选)
weather_api_key = ""          # ALAPI天气API的token密钥，需要在ALAPI官网注册获取
alapi_base_url = "https://v3.alapi.cn/api"  # ALAPI基础URL
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

## 开发文档

### 项目结构

```
plugins/OpenManus/
├── config.toml         # 配置文件
├── main.py             # 插件入口
├── api_client.py       # API客户端
├── README.md           # 说明文档
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

然后在`main.py`的`_init_agent`方法中注册该工具。

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

### 性能优化

如果遇到响应慢的问题:

1. 减少`thinking_steps`的值
2. 使用更快的模型
3. 减少`max_tokens`值

## 版权和许可

OpenManus 插件使用 MIT 许可证。更多信息请参见 LICENSE 文件。
