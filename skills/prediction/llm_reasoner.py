"""
LLM Reasoner - LLM推理器

This module implements Chain-of-Thought reasoning using GPT-4o to predict
the next gaze target based on context, history, and spatial constraints.
"""

import json
from typing import Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import ModelConfig, AttentionConfig


class LLMReasoner:
    """
    LLM推理器 - Chain-of-Thought推理

    This class uses GPT-4o to perform expert-level spatial behavior analysis
    and predict the next gaze target with reasoning and confidence scoring.
    """

    def __init__(self, config: ModelConfig):
        """
        初始化LLM推理器

        Args:
            config: ModelConfig对象，包含LLM配置
        """
        self.llm = ChatOpenAI(
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
            timeout=120,  # 120秒超时（适应 Qwen API）
            request_timeout=120
        )
        self.output_parser = StrOutputParser()

    def predict_next(
        self,
        context: Dict,
        attention_config: AttentionConfig
    ) -> Dict:
        """
        预测下一个凝视目标

        Args:
            context: ContextBuilder构建的上下文字典
            attention_config: AttentionConfig对象，包含注意力等级配置

        Returns:
            预测结果字典，包含:
            - prediction_id: 预测的展品ID
            - prediction_name: 预测的展品名称
            - attention_level: 预测的注意力等级 (A/B/C/D/E)
            - estimated_duration: 预计停留时间(秒)
            - confidence: 置信度 (0.0-1.0)
            - reasoning: 推理过程说明
            - error: 错误信息 (如果失败)
        """
        if "error" in context:
            return {
                "error": context["error"],
                "prediction_id": None,
                "reasoning": f"Context error: {context['error']}"
            }

        # 构建专家级prompt
        system_prompt = self._get_system_prompt(attention_config)
        user_prompt = self._format_user_prompt(context)

        # 执行推理
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt)
        ])

        chain = prompt | self.llm | self.output_parser

        try:
            result = chain.invoke({})
            return self._parse_llm_output(result)
        except Exception as e:
            return {
                "error": str(e),
                "prediction_id": None,
                "reasoning": f"LLM invocation failed: {str(e)}"
            }

    def _get_system_prompt(self, attention_config: AttentionConfig) -> str:
        """构建系统提示词"""
        duration_info = "\n".join([
            f"  - Level {level}: {attention_config.ATTENTION_DURATION[level]}s - {attention_config.ATTENTION_DESCRIPTION[level]}"
            for level in ['A', 'B', 'C', 'D', 'E']
        ])

        return f"""你是一个博物馆空间行为分析专家 (Space Syntax Expert)。

基于用户的历史行为、当前空间环境和展品特征，预测下一个关注目标。

**分析维度：**
1. **语义连贯性 (Semantic Continuity)**: 用户是否在阅读相关内容序列？(如：说明牌 -> 展品 -> 补充信息)
2. **空间流线 (Spatial Flow)**: 用户是否顺应展厅设计的推荐动线 (Next关系)？
3. **视觉显著性 (Visual Saliency)**: 是否有视觉上突出的展品？(通过Visual关系连接)
4. **个人兴趣 (Personal Interest)**: 基于历史行为，用户偏好什么类型？
5. **访问历史 (Visit History)**: 用户是否已经访问过某些展品？避免重复预测已访问的展品

**注意力等级标准：**
{duration_info}

**输出格式（严格按照JSON）：**
```json
{{{{
  "prediction_id": "展品ID",
  "prediction_name": "展品名称",
  "attention_level": "A/B/C/D/E",
  "estimated_duration": 秒数,
  "confidence": 0.0-1.0,
  "reasoning": "详细的推理过程，说明为什么选择这个展品..."
}}}}
```

**重要提示：**
- 只输出JSON，不要有其他文字
- prediction_id必须是可达选项中的一个
- confidence应该基于多个维度的综合判断
- reasoning应该清晰展示分析过程
"""

    def _format_user_prompt(self, context: Dict) -> str:
        """格式化用户提示词"""
        current = context['current']
        history = context['history']
        spatial = context['spatial']
        stats = context['statistics']

        prompt = f"""**当前状态**
- 位置: {current['name']} ({current['id']})
- 特征: {current['features']}
- 当前注意力: {current['attention_level']} (已停留{current['estimated_duration']}秒)

**历史行为** (最近{len(history)}个)
"""
        for i, h in enumerate(history, 1):
            prompt += f"{i}. [{h['level']}] {h['name']} ({h['id']}) - {h['duration']}s\n"

        prompt += f"""
**统计信息**
- 总凝视次数: {stats['total_gazes']}
- 已访问展品数: {stats['unique_exhibits']}
- 已访问展品IDs: {', '.join(stats['visited_exhibits'])}

**空间环境**
"""
        # 前序路径
        prev_path = spatial.get('previous_path', [])
        if prev_path:
            prev_str = " -> ".join([f"{n['name']}" for n in prev_path])
            prompt += f"- 来自动线: {prev_str}\n"

        # 后续路径
        next_path = spatial.get('next_path', [])
        if next_path:
            next_str = " -> ".join([f"{n['name']}" for n in next_path])
            prompt += f"- 去往动线: {next_str}\n"

        # 可达选项
        prompt += "- 可达选项:\n"
        reachable = spatial.get('reachable_options', [])
        if not reachable:
            prompt += "  无可达选项\n"
        else:
            for opt in reachable:
                # 获取邻居节点的详细信息
                neighbor_name = opt.get('name', 'Unknown')
                prompt += f"  • {opt['id']} ({neighbor_name}) - 关系: {opt['relation']}\n"

        prompt += "\n请基于以上信息预测下一个ID。"

        return prompt

    def _parse_llm_output(self, output: str) -> Dict:
        """
        解析LLM输出

        Args:
            output: LLM返回的原始字符串

        Returns:
            解析后的字典，如果解析失败则返回error字段
        """
        # 尝试提取JSON
        try:
            # 尝试直接解析
            return json.loads(output)
        except json.JSONDecodeError:
            # 尝试提取JSON块
            try:
                # 查找 ```json ... ``` 块
                start = output.find("```json")
                if start != -1:
                    start += 7  # 跳过 "```json"
                    end = output.find("```", start)
                    if end != -1:
                        json_str = output[start:end].strip()
                        return json.loads(json_str)

                # 查找 { ... } 块
                start = output.find("{")
                if start != -1:
                    end = output.rfind("}") + 1
                    if end > start:
                        json_str = output[start:end]
                        return json.loads(json_str)

                # 如果都失败了
                return {
                    "error": "Failed to parse JSON from LLM output",
                    "raw_output": output,
                    "prediction_id": None
                }
            except Exception as e:
                return {
                    "error": f"JSON parsing error: {str(e)}",
                    "raw_output": output,
                    "prediction_id": None
                }

    def predict_with_explanation(
        self,
        context: Dict,
        attention_config: AttentionConfig,
        verbose: bool = True
    ) -> Dict:
        """
        带详细解释的预测方法

        Args:
            context: 上下文字典
            attention_config: 注意力配置
            verbose: 是否打印详细信息

        Returns:
            预测结果字典
        """
        result = self.predict_next(context, attention_config)

        if verbose and "error" not in result:
            print("\n" + "="*50)
            print("🔮 LLM Prediction Result")
            print("="*50)
            print(f"Target: {result['prediction_name']} ({result['prediction_id']})")
            print(f"Attention Level: {result['attention_level']}")
            print(f"Estimated Duration: {result['estimated_duration']}s")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"\nReasoning:")
            print(result['reasoning'])
            print("="*50 + "\n")

        return result

    def batch_predict(
        self,
        contexts: List[Dict],
        attention_config: AttentionConfig
    ) -> List[Dict]:
        """
        批量预测

        Args:
            contexts: 上下文字典列表
            attention_config: 注意力配置

        Returns:
            预测结果列表
        """
        results = []
        for context in contexts:
            result = self.predict_next(context, attention_config)
            results.append(result)
        return results
