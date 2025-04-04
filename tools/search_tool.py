import aiohttp
import json
from typing import Dict, Any, Optional
from loguru import logger

from ..agent.mcp import Tool

class SearchTool(Tool):
    """搜索工具，用于执行网络搜索"""
    
    def __init__(self, api_key: Optional[str] = None, 
                search_url: str = "https://api.bing.microsoft.com/v7.0/search",
                search_engine: str = "bing"):
        """初始化搜索工具
        
        Args:
            api_key: 搜索API密钥 (Bing 或 Serper.dev)
            search_url: 搜索API地址 (Bing 或 Serper.dev)
            search_engine: 搜索引擎类型 (bing 或 serper)
        """
        super().__init__(
            name="search",
            description="执行网络搜索，查询特定信息",
            parameters={
                "query": {
                    "type": "string",
                    "description": "搜索查询关键词"
                },
                "count": {
                    "type": "integer",
                    "description": "返回结果数量",
                    "default": 5
                }
            }
        )
        self.api_key = api_key
        self.search_url = search_url
        self.search_engine = search_engine.lower()
        
        # 根据引擎类型调整默认URL (如果未提供)
        if not search_url:
            if self.search_engine == "serper":
                self.search_url = "https://google.serper.dev/search"
            else:
                self.search_url = "https://api.bing.microsoft.com/v7.0/search"
        
    async def execute(self, query: str, count: int = 5) -> Dict[str, Any]:
        """执行网络搜索
        
        Args:
            query: 搜索查询关键词
            count: 返回结果数量
            
        Returns:
            Dict: 搜索结果
        """
        logger.info(f"执行搜索: query={query}, count={count}, 引擎={self.search_engine}")
        
        if not self.api_key:
            # 模拟搜索结果
            logger.warning("未提供搜索API密钥，返回模拟结果")
            return {
                "results": [
                    {
                        "title": f"模拟搜索结果 1: {query}",
                        "snippet": f"这是关于 {query} 的模拟搜索结果。由于未提供API密钥，无法执行真实搜索。",
                        "url": f"https://example.com/result1-{query}"
                    },
                    {
                        "title": f"模拟搜索结果 2: {query}",
                        "snippet": f"更多关于 {query} 的模拟信息。这是第二个模拟结果。",
                        "url": f"https://example.com/result2-{query}"
                    }
                ],
                "warning": "使用模拟搜索结果。要获取真实搜索结果，请提供有效的API密钥。"
            }
            
        # 根据搜索引擎选择不同的实现
        if self.search_engine == "serper":
            return await self._search_with_serper(query, count)
        else:  # 默认使用 Bing
            return await self._search_with_bing(query, count)
    
    async def _search_with_bing(self, query: str, count: int) -> Dict[str, Any]:
        """使用Bing搜索
        
        Args:
            query: 搜索查询关键词
            count: 返回结果数量
            
        Returns:
            Dict: 搜索结果
        """
        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key
        }
        
        params = {
            "q": query,
            "count": count,
            "responseFilter": "webpages",
            "textFormat": "raw",
            "mkt": "zh-CN" # 设定市场为中国
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.search_url, headers=headers, params=params) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Bing搜索API错误: {response.status} - {error_text}")
                        return {"error": f"Bing搜索失败: HTTP {response.status}"}
                        
                    json_data = await response.json()
                    
                    # 提取结果
                    web_pages = json_data.get("webPages", {}).get("value", [])
                    results = []
                    for page in web_pages:
                        results.append({
                            "title": page.get("name", ""),
                            "snippet": page.get("snippet", ""),
                            "url": page.get("url", "")
                        })
                        
                    return {"results": results}
                    
        except Exception as e:
            logger.error(f"Bing搜索错误: {str(e)}")
            return {"error": f"Bing搜索失败: {str(e)}"}
    
    async def _search_with_serper(self, query: str, count: int) -> Dict[str, Any]:
        """使用Serper.dev搜索
        
        Args:
            query: 搜索查询关键词
            count: 返回结果数量 (Serper可能不直接支持精确数量，但会影响返回)
            
        Returns:
            Dict: 搜索结果
        """
        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
        
        payload = json.dumps({
            "q": query,
            "num": count, # Serper使用num参数
            "gl": "cn", # 设定国家为中国
            "hl": "zh-cn" # 设定语言为简体中文
        })
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.search_url, headers=headers, data=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Serper搜索API错误: {response.status} - {error_text}")
                        return {"error": f"Serper搜索失败: HTTP {response.status}"}
                        
                    json_data = await response.json()
                    
                    # 提取结果
                    organic_results = json_data.get("organic", [])
                    results = []
                    for item in organic_results:
                        results.append({
                            "title": item.get("title", ""),
                            "snippet": item.get("snippet", ""),
                            "url": item.get("link", "")
                        })
                        
                    return {"results": results}
                    
        except Exception as e:
            logger.error(f"Serper搜索错误: {str(e)}")
            return {"error": f"Serper搜索失败: {str(e)}"} 