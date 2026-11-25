from ceval_eval.prompts import (
    build_baseline_prompt,
    build_fewshot_prompt,
    build_selfcheck_prompt,
    build_strict_prompt,
)


def sample_record():
    return {
        "subject": "college_economics",
        "subject_cn": "大学经济学",
        "question": "在完全竞争市场中，长期均衡时厂商的利润为？",
        "options": {"A": "正利润", "B": "零利润", "C": "负利润", "D": "不确定"},
        "answer": "B",
    }


def sample_example():
    return {
        "subject": "college_economics",
        "question": "需求曲线向下倾斜说明什么？",
        "options": {"A": "价格上升需求量下降", "B": "价格上升需求量上升", "C": "无关", "D": "不确定"},
        "answer": "A",
    }


def test_baseline_prompt_contains_question_and_options():
    prompt = build_baseline_prompt(sample_record())
    assert "大学经济学" in prompt
    assert "完全竞争市场" in prompt
    assert "A. 正利润" in prompt
    assert prompt.rstrip().endswith("答案：")


def test_strict_prompt_requests_only_letter():
    prompt = build_strict_prompt(sample_record())
    assert "只输出一个选项字母" in prompt
    assert "不要输出解释" in prompt


def test_fewshot_prompt_contains_examples_then_target():
    prompt = build_fewshot_prompt(sample_record(), [sample_example()])
    assert "示例 1" in prompt
    assert "需求曲线向下倾斜" in prompt
    assert prompt.index("示例 1") < prompt.index("在完全竞争市场")


def test_selfcheck_prompt_contains_first_answer():
    prompt = build_selfcheck_prompt(sample_record(), "答案：A")
    assert "第一轮回答：答案：A" in prompt
    assert "复核结果" in prompt
