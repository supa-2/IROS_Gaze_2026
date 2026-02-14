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

        # 2. 构造 Prompt，要求 VLM 输出完整展品信息
        prompt = """
        分析这张博物馆视角的图片，提取所有展品及其详细信息。

        对每个识别到的展品/物品，请提供：
        1. id: 唯一标识符 (如 "exhibit_1")
        2. label: 简短名称 (如 "青铜鼎")
        3. type: 类型 (Exhibit/Label/Passage)
        4. description: 详细描述 (材质、年代、风格、内容等，至少20字)
        5. bbox: 边界框 [x1, y1, x2, y2] (以左上角为原点)
        6. center: 中心点 [cx, cy]
        7. confidence: 置信度 0-1
        8. visual_features: 视觉特征 (颜色、形状、大小等)

        空间关系请用 edges 表示：
        - source: 起点ID
        - target: 终点ID
        - relation: 关系类型 (next_to/above/below/left_of/right_of)

        请严格以 JSON 格式输出，不要任何解释文字。
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