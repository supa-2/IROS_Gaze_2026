import base64
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

class VisualMapper:
    def __init__(self):
        # 假设我们用 Qwen-VL 或 GPT-4o
        self.vlm = ChatOpenAI(model="gpt-4o", max_tokens=300)

    def image_to_topology(self, image_path):
        """
        输入：环境照片
        输出：局部拓扑 JSON (Nodes & Edges)
        """
        # 1. 读取图片并编码
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        # 2. 构造 Prompt，要求 VLM 输出结构化数据
        prompt = """
        分析这张博物馆视角的图片。
        请提取画面中的关键物体（作为节点）及其空间关系（作为边）。
        
        重点关注：
        1. 展品 (Exhibits)
        2. 说明牌 (Labels)
        3. 出口/通道 (Passages)
        
        请严格以 JSON 格式输出，格式如下：
        {
            "nodes": [{"id": "obj1", "label": "青铜鼎", "type": "Exhibit"}],
            "edges": [{"source": "obj1", "target": "obj2", "relation": "next_to"}]
        }
        不要输出任何其他废话。
        """

        # 3. 调用模型
        msg = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        )
        
        response = self.vlm.invoke([msg])
        
        # 4. 解析 JSON (要做一些容错处理)
        try:
            local_graph = json.loads(response.content.replace("```json", "").replace("```", ""))
            return local_graph
        except:
            return {"error": "Failed to parse VLM output"}

# 测试接口
if __name__ == "__main__":
    # 模拟一张图片路径
    mapper = VisualMapper()
    print(mapper.image_to_topology("test_scene.jpg"))