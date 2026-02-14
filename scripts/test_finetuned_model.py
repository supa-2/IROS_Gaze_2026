"""
微调后模型测试脚本

用于测试 base 模型微调后的效果
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def create_prompt(instruction: str, input_text: str = "") -> str:
    """构造测试 prompt"""
    if input_text and input_text.strip():
        return f"<instruction>{instruction}</instruction>\n<input>{input_text}</input>\n<output>"
    else:
        return f"<instruction>{instruction}</instruction>\n<output>"


def test_model(model_path: str, device: str = "auto"):
    """
    测试微调后的模型

    Args:
        model_path: 微调后的模型路径
        device: 设备 (cuda/cpu/auto)
    """
    print(f"[INFO] Loading model from: {model_path}")

    # 加载模型和 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device
    )

    print(f"[INFO] Model loaded successfully")

    # 测试用例
    test_cases = [
        {
            "name": "路径规划测试",
            "instruction": "规划一条包含注意力等级(A-E)的视觉扫描路径，用于构建注意力热图。",
            "input": """当前场景可见展品(Visual Context):
- 【丁香花】: 一幅画着由白色圆盆栽开满白色小花，并且绿叶繁盛的画，直立挂起来，背景为黑色
- 【金鱼兰】: 一幅画着土红色盆子载种着一支叶片细长，花朵呈金鱼状的画
- 【说明文字】: 墙面展签，印有展品名称"极乐鸟"，无长文本""",
            "expected_prefix": "热度路径规划:"
        },
        {
            "name": "下一步预测测试",
            "instruction": "基于当前视觉场景和历史注视行为，预测用户下一个关注目标的注意力等级 (A/B/C/D/E)。",
            "input": """当前场景可见展品(Visual Context):
- 【水果与船】: 一副画着水果香蕉、蛋黄果、船的画
- 【森林之歌】: 一幅画着高耸树木，树枝交织在一起的画
- 【鸟类】: 另外一副画着鸟，鹦鹉的画

历史行为:
1. [B] 水果与船""",
            "expected_prefix": "下一步预测:"
        },
        {
            "name": "特征归因测试",
            "instruction": "识别并提取出与目标展品【二十四节气圆盘】的高强度关注（B级）相对应的视觉显著性特征。",
            "input": """当前场景可见展品(Visual Context):
- 【耕织图】: 通过数字活化展示中国农耕文化
- 【二十四节气圆盘】: 融合虚拟现实技术的动态影像装置""",
            "expected_prefix": "特征归因:"
        },
        {
            "name": "零样本泛化测试（新展品）",
            "instruction": "规划一条包含注意力等级(A-E)的视觉扫描路径，用于构建注意力热图。",
            "input": """当前场景可见展品(Visual Context):
- 【星空展品】: 一个展示宇宙星空的沉浸式投影装置
- 【互动屏幕】: 触摸式信息查询屏幕，显示展品详情
- 【模型展示】: 精细的航天器模型""",
            "expected_prefix": "热度路径规划:"
        }
    ]

    print("\n" + "="*80)
    print("Running Test Cases")
    print("="*80)

    results = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}: {test_case['name']}")
        print(f"{'='*80}")

        # 构造 prompt
        prompt = create_prompt(test_case['instruction'], test_case['input'])

        print(f"\n[Input]")
        print(f"Instruction: {test_case['instruction'][:80]}...")
        print(f"Input: {test_case['input'][:100]}...")

        # 生成
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1
            )

        # 解码
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取生成的部分（移除 prompt）
        generated_only = generated_text[len(prompt):].strip()

        print(f"\n[Generated Output]")
        print(generated_only)

        # 检查格式
        has_expected_prefix = generated_only.startswith(test_case['expected_prefix'])
        has_brackets = any(f"[{level}]" in generated_only for level in ['A', 'B', 'C', 'D', 'E'])

        print(f"\n[Validation]")
        print(f"  Expected prefix '{test_case['expected_prefix']}': {'✓' if has_expected_prefix else '✗'}")
        print(f"  Contains attention level [A-E]: {'✓' if has_brackets else '✗'}")

        results.append({
            'name': test_case['name'],
            'format_ok': has_expected_prefix,
            'has_levels': has_brackets,
            'output': generated_only
        })

    # 总结
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)

    format_ok_count = sum(1 for r in results if r['format_ok'])
    has_levels_count = sum(1 for r in results if r['has_levels'])

    print(f"\nTotal tests: {len(results)}")
    print(f"Format correct: {format_ok_count}/{len(results)}")
    print(f"Has attention levels: {has_levels_count}/{len(results)}")

    print("\nDetailed results:")
    for r in results:
        status = "✓ PASS" if r['format_ok'] and r['has_levels'] else "✗ FAIL"
        print(f"  {status} - {r['name']}")

    success_rate = format_ok_count / len(results) * 100
    print(f"\nSuccess rate: {success_rate:.1f}%")

    if success_rate >= 75:
        print("[INFO] Model performance is GOOD! ✓")
    elif success_rate >= 50:
        print("[INFO] Model performance is FAIR. Consider more training.")
    else:
        print("[INFO] Model performance is POOR. Check data and hyperparameters.")

    return results


def interactive_test(model_path: str, device: str = "auto"):
    """
    交互式测试模式

    Args:
        model_path: 微调后的模型路径
        device: 设备
    """
    print(f"[INFO] Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device
    )

    print("\n" + "="*80)
    print("Interactive Test Mode")
    print("="*80)
    print("Enter your test cases (type 'quit' to exit)\n")

    while True:
        print("\n" + "-"*80)
        instruction = input("Instruction (or 'quit'): ").strip()

        if instruction.lower() == 'quit':
            print("[INFO] Exiting...")
            break

        if not instruction:
            print("[WARN] Instruction cannot be empty")
            continue

        input_text = input("Input (press Enter to skip): ").strip()

        # 构造 prompt
        prompt = create_prompt(instruction, input_text)

        print("\n[Generating...]")

        # 生成
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )

        # 解码
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_only = generated_text[len(prompt):].strip()

        print(f"\n[Output]")
        print(generated_only)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test fine-tuned base model')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to fine-tuned model')
    parser.add_argument('--mode', type=str, default='auto',
                       choices=['auto', 'test', 'interactive'],
                       help='Test mode: auto (run test cases), interactive (manual input)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: cuda, cpu, or auto')

    args = parser.parse_args()

    if args.mode == 'interactive':
        interactive_test(args.model_path, args.device)
    else:
        test_model(args.model_path, args.device)
