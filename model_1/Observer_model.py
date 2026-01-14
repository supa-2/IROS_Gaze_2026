from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
import config
from model_2.State_model import RobotState

# 1.获取模型
model = ChatOpenAI(
    model="Deepseek-V3.2",
    openai_key = ["model1_api_key"],
    openai_api_base = "https://llmapi.paratera.com/v1/",
    temperature=0.7,
    max_tokens = 4096,
    timeout=30
)
# 2. 创建agent
agent_1 = create_agent(model)
def module_perception(state:RobotState):
    print("---gaze 正在观察环境...---")

    detected_items = "xxx"

    return {"semantic_context":detected_items}