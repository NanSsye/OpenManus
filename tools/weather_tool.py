import json
import aiohttp
from typing import Dict, Any, Optional
from loguru import logger

from ..agent.mcp import Tool

class WeatherTool(Tool):
    """天气工具，用于获取天气信息"""
    
    def __init__(self, api_key: Optional[str] = None, 
                weather_url: str = "https://v3.alapi.cn/api/tianqi",
                forecast_url: str = "https://v3.alapi.cn/api/tianqi/seven",
                index_url: str = "https://v3.alapi.cn/api/tianqi/index"):
        """初始化天气工具
        
        Args:
            api_key: ALAPI的token密钥
            weather_url: 实时天气API地址 (当前未使用，保留以备后用)
            forecast_url: 7天天气预报API地址
            index_url: 天气指数API地址
        """
        super().__init__(
            name="weather",
            description="获取特定城市或地区未来7天的天气预报信息。",
            parameters={
                "city": {
                    "type": "string",
                    "description": "目标城市名称，例如 '北京' 或 '上海'",
                },
                "province": {
                    "type": "string",
                    "description": "省份名称，例如 '江西'，与city配合使用可提高准确性",
                    "default": ""
                },
                "city_id": {
                    "type": "string",
                    "description": "城市ID，可以精确定位城市，获取更详细的天气指数",
                    "default": ""
                },
                "use_ip": {
                    "type": "boolean",
                    "description": "是否使用IP定位（设为true会忽略其他位置参数）",
                    "default": False
                },
                "longitude": {
                    "type": "string",
                    "description": "经度，与纬度配合使用可精确定位",
                    "default": ""
                },
                "latitude": {
                    "type": "string",
                    "description": "纬度，与经度配合使用可精确定位",
                    "default": ""
                }
            }
        )
        self.api_key = api_key
        self.weather_url = weather_url
        self.forecast_url = forecast_url
        self.index_url = index_url
        
    async def _get_weather_indexes(self, city_id: str) -> Dict[str, Any]:
        """获取天气指数信息
        
        Args:
            city_id: 城市ID
            
        Returns:
            Dict: 天气指数信息或错误信息
        """
        if not city_id:
            return {"error": "获取天气指数需要提供城市ID"}
        if not self.api_key:
             return {"error": "获取天气指数需要有效的API密钥"}
            
        params = {
            "token": self.api_key,
            "city_id": city_id
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.index_url, params=params) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"天气指数API错误: {response.status} - {error_text}")
                        return {"error": f"天气指数API请求失败: HTTP {response.status}"}
                        
                    json_data = await response.json()
                    
                    if not json_data.get("success", False):
                        error_msg = json_data.get("message", "未知错误")
                        logger.error(f"天气指数API返回错误: {error_msg}")
                        return {"error": f"天气指数API返回错误: {error_msg}"}
                        
                    index_data = json_data.get("data", [])
                    if not index_data:
                        return {"warning": "天气指数API返回数据为空"}
                        
                    # 处理天气指数数据
                    indexes = {}
                    for index in index_data:
                        name = index.get("name", "未知指数")
                        indexes[name] = {
                            "content": index.get("content", ""),
                            "level": index.get("level", ""),
                            "type": index.get("type", ""),
                            "date": index.get("date", "")
                        }
                        
                    return {"indexes": indexes}
                    
        except Exception as e:
            logger.error(f"获取天气指数错误: {str(e)}")
            return {"error": f"获取天气指数失败: {str(e)}"}
        
    async def execute(self, city: str = "", province: str = "", city_id: str = "", 
                     use_ip: bool = False, longitude: str = "", latitude: str = "") -> Dict[str, Any]:
        """获取天气信息 (未来7天预报)
        
        Args:
            city: 城市名称
            province: 省份名称
            city_id: 城市ID
            use_ip: 是否使用IP定位
            longitude: 经度
            latitude: 纬度
            
        Returns:
            Dict: 天气信息 (包含今日实况和未来预报)
        """
        logger.info(f"获取天气信息: city={city}, province={province}, city_id={city_id}, use_ip={use_ip}, lon={longitude}, lat={latitude}")
        
        if not self.api_key:
            logger.warning("未提供ALAPI的token，无法获取天气数据")
            return {"error": "未提供有效的API密钥，无法获取天气数据，请联系管理员配置天气API密钥"}
        
        # 构建API请求参数
        params = {
            "token": self.api_key
        }
        
        query_city_id = city_id
        if use_ip:
            params["ip"] = "auto"
            query_city_id = None
        elif city_id:
            params["city_id"] = city_id
        elif longitude and latitude:
            params["lon"] = longitude
            params["lat"] = latitude
            query_city_id = None
        elif city:
            params["city"] = city
            if province:
                params["province"] = province
            query_city_id = None
        else:
            return {"error": "未提供有效的位置信息，请提供城市名称、城市ID、IP或经纬度"}
        
        target_url = self.forecast_url
        
        try:
            result = {}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(target_url, params=params) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"天气API错误 ({target_url}): {response.status} - {error_text}")
                        return {"error": f"天气API请求失败: HTTP {response.status} - 无法获取天气数据"}
                    
                    json_data = await response.json()
                    
                    if not json_data.get("success", False):
                        error_msg = json_data.get("message", "未知错误")
                        logger.error(f"天气API返回错误 ({target_url}): {error_msg}")
                        return {"error": f"天气API返回错误: {error_msg} - 无法获取天气数据"}
                    
                    forecast_data = json_data.get("data", [])
                    if not forecast_data:
                        return {"error": "天气API返回数据为空，无法获取天气信息"}
                    
                    # Find city_id if not known
                    if not query_city_id and "city_id" in forecast_data[0]:
                         query_city_id = forecast_data[0]["city_id"]
                    
                    # Process the first day (today's weather)
                    today_data = forecast_data[0]
                    result = {
                        "city": today_data.get("city", city or "未知城市"),
                        "province": today_data.get("province", province or "未知省份"),
                        "city_id": query_city_id or today_data.get("city_id", None),
                        "date": today_data.get("date", "未知"), # Today's date
                        "day_weather": today_data.get("wea_day", "未知"),
                        "night_weather": today_data.get("wea_night", "未知"),
                        "temperature": f"{today_data.get('temp_night', '?')}°C-{today_data.get('temp_day', '?')}°C",
                        "day_wind": f"{today_data.get('wind_day', '?')} {today_data.get('wind_day_level', '?')}",
                        "night_wind": f"{today_data.get('wind_night', '?')} {today_data.get('wind_night_level', '?')}",
                        "air_quality": {
                            "aqi": today_data.get("air", "未知"),
                            "level": today_data.get("air_level", "未知")
                        },
                        "precipitation": today_data.get("precipitation", "未知"),
                        "sunrise": today_data.get("sunrise", "未知"),
                        "sunset": today_data.get("sunset", "未知")
                    }
                    
                    # Get indexes if we have city_id
                    if query_city_id:
                        index_result = await self._get_weather_indexes(query_city_id)
                        if "indexes" in index_result:
                            result["weather_indexes"] = index_result["indexes"]
                        elif "warning" in index_result:
                            logger.warning(index_result["warning"])
                        elif "error" in index_result:
                            logger.error(f"获取天气指数失败: {index_result['error']}")
                            result["index_error"] = index_result['error']
                    
                    # Process and add the future forecast (next 6 days)
                    if len(forecast_data) > 1:
                        future_forecast = []
                        for day in forecast_data[1:]: # Skip today
                            forecast = {
                                "date": day.get("date", "未知"),
                                "day_weather": day.get("wea_day", "未知"),
                                "night_weather": day.get("wea_night", "未知"),
                                "temperature": f"{day.get('temp_night', '?')}°C-{day.get('temp_day', '?')}°C",
                                "day_wind": f"{day.get('wind_day', '?')} {day.get('wind_day_level', '?')}",
                                "night_wind": f"{day.get('wind_night', '?')} {day.get('wind_night_level', '?')}",
                                "air": day.get("air", "未知"),
                                "air_level": day.get("air_level", "未知"),
                                "precipitation": day.get("precipitation", "未知")
                            }
                            future_forecast.append(forecast)
                        
                        if future_forecast:
                            result["future_forecast"] = future_forecast
            
            return result
                    
        except Exception as e:
            logger.exception(f"天气查询主流程错误: {str(e)}")
            return {"error": f"天气查询失败: {str(e)} - 无法获取天气数据"} 