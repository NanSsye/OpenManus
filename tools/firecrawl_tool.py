import json
import aiohttp
from typing import Dict, Any, Optional, List, Union
from loguru import logger
from firecrawl.firecrawl import FirecrawlApp

from ..agent.mcp import Tool

class FirecrawlTool(Tool):
    """Firecrawl 工具，用于网站爬取、搜索和数据提取"""
    
    def __init__(self, api_key: Optional[str] = None):
        """初始化 Firecrawl 工具
        
        Args:
            api_key: Firecrawl API 密钥
        """
        super().__init__(
            name="firecrawl",
            description="使用 Firecrawl 爬取网站、搜索内容并提取数据",
            parameters={
                "action": {
                    "type": "string",
                    "description": "要执行的操作：'scrape'(单页爬取), 'crawl'(多页爬取), 'search'(搜索)",
                    "enum": ["scrape", "crawl", "search"]
                },
                "url": {
                    "type": "string",
                    "description": "要爬取的网站 URL (当 action 为 'scrape' 或 'crawl' 时必填)"
                },
                "query": {
                    "type": "string",
                    "description": "搜索查询关键词 (当 action 为 'search' 时必填)"
                },
                "format": {
                    "type": "string",
                    "description": "返回内容的格式",
                    "enum": ["markdown", "html", "json"],
                    "default": "markdown"
                },
                "limit": {
                    "type": "integer",
                    "description": "爬取的最大页面数量 (仅适用于 'crawl' 操作)",
                    "default": 10
                },
                "extract_schema": {
                    "type": "string",
                    "description": "用于结构化数据提取的 JSON 模式 (当 format 为 'json' 时可选)"
                }
            }
        )
        self.api_key = api_key
        self._client = None
        
    @property
    def client(self) -> FirecrawlApp:
        """获取或初始化 Firecrawl 客户端"""
        if self._client is None:
            if not self.api_key:
                logger.warning("未提供 Firecrawl API 密钥，某些功能可能无法使用")
            self._client = FirecrawlApp(api_key=self.api_key)
        return self._client
    
    async def execute(self, 
                     action: str, 
                     url: Optional[str] = None,
                     query: Optional[str] = None,
                     format: str = "markdown",
                     limit: int = 10,
                     extract_schema: Optional[str] = None) -> Dict[str, Any]:
        """执行 Firecrawl 操作
        
        Args:
            action: 操作类型 ("scrape", "crawl", "search")
            url: 要爬取的 URL (scrape/crawl)
            query: 搜索查询 (search)
            format: 返回格式 (markdown/html/json)
            limit: 爬取页面上限 (crawl)
            extract_schema: JSON 提取模式 (可选)
            
        Returns:
            Dict: 操作结果
        """
        logger.info(f"执行 Firecrawl 操作: {action}, URL={url}, query={query}, format={format}")
        
        if not self.api_key:
            logger.warning("未提供 Firecrawl API 密钥，返回模拟结果")
            return self._mock_results(action, url, query)
        
        try:
            formats = [format]
            
            if action == "scrape":
                if not url:
                    return {"error": "爬取操作需要提供 URL 参数"}
                
                params = {"formats": formats}
                if extract_schema and format == "json":
                    try:
                        schema_obj = json.loads(extract_schema)
                        params["jsonOptions"] = {"schema": schema_obj}
                    except json.JSONDecodeError:
                        logger.error(f"JSON 模式解析失败: {extract_schema}")
                        return {"error": "JSON 模式格式不正确"}
                
                result = self.client.scrape_url(url, params=params)
                return {"result": result, "url": url}
            
            elif action == "crawl":
                if not url:
                    return {"error": "爬取操作需要提供 URL 参数"}
                
                # 构建爬取参数
                params = {
                    "limit": limit,
                    "scrapeOptions": {"formats": formats}
                }
                
                if extract_schema and format == "json":
                    try:
                        schema_obj = json.loads(extract_schema)
                        params["scrapeOptions"]["jsonOptions"] = {"schema": schema_obj}
                    except json.JSONDecodeError:
                        logger.error(f"JSON 模式解析失败: {extract_schema}")
                        return {"error": "JSON 模式格式不正确"}
                
                # 执行爬取
                result = self.client.crawl_url(url, params=params, poll_interval=10)
                return {"result": result, "url": url}
            
            elif action == "search":
                if not query:
                    return {"error": "搜索操作需要提供 query 参数"}
                
                # 构建搜索参数，包括爬取选项
                params = {
                    "query": query,
                    "scrapeOptions": {"formats": formats}
                }
                
                if extract_schema and format == "json":
                    try:
                        schema_obj = json.loads(extract_schema)
                        params["scrapeOptions"]["jsonOptions"] = {"schema": schema_obj}
                    except json.JSONDecodeError:
                        logger.error(f"JSON 模式解析失败: {extract_schema}")
                        return {"error": "JSON 模式格式不正确"}
                
                # 执行搜索
                result = self.client.search(params)
                return {"result": result, "query": query}
            
            else:
                return {"error": f"不支持的操作类型: {action}"}
            
        except Exception as e:
            logger.exception(f"Firecrawl 操作失败: {str(e)}")
            return {"error": f"Firecrawl 操作失败: {str(e)}"}
    
    def _mock_results(self, action: str, url: Optional[str] = None, query: Optional[str] = None) -> Dict[str, Any]:
        """提供模拟结果，用于未配置 API 密钥的情况
        
        Args:
            action: 操作类型
            url: URL
            query: 查询
            
        Returns:
            Dict: 模拟结果
        """
        if action == "scrape" or action == "crawl":
            return {
                "warning": "使用模拟数据。要获取真实数据，请提供有效的 Firecrawl API 密钥。",
                "result": {
                    "markdown": f"# 模拟网页内容: {url}\n\n这是从 {url} 抓取的模拟内容。由于未提供 API 密钥，无法执行真实爬取。\n\n## 内容概述\n\n* 这是一个模拟的网页\n* 包含一些示例文本\n* 没有真实数据",
                    "url": url
                }
            }
        elif action == "search":
            return {
                "warning": "使用模拟搜索结果。要获取真实结果，请提供有效的 Firecrawl API 密钥。",
                "result": {
                    "data": [
                        {
                            "url": f"https://example.com/result1-{query}",
                            "title": f"模拟搜索结果 1: {query}",
                            "description": f"这是关于 {query} 的模拟搜索结果。"
                        },
                        {
                            "url": f"https://example.com/result2-{query}",
                            "title": f"模拟搜索结果 2: {query}",
                            "description": f"更多关于 {query} 的模拟信息。"
                        }
                    ],
                    "query": query
                }
            }
        else:
            return {"error": f"不支持的操作类型: {action}"} 