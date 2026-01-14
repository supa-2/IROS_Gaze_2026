from langchain.tools import tool
@tool
def search(query:str) -> str:
    """在地图数据集中搜索信息"""
    return f"当前位置周边物品:{query}"