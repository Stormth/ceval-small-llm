from __future__ import annotations

from typing import Any


def format_options(options: dict[str, str]) -> str:
    return "\n".join(f"{letter}. {options[letter]}" for letter in ("A", "B", "C", "D"))


def question_block(record: dict[str, Any], *, include_answer: bool = False) -> str:
    block = (
        f"题目：{record['question']}\n"
        f"{format_options(record['options'])}"
    )
    if include_answer:
        block += f"\n答案：{record['answer']}"
    return block


def build_baseline_prompt(record: dict[str, Any]) -> str:
    return (
        f"以下是中国关于{record['subject_cn']}考试的单项选择题，请选出其中的正确答案。\n\n"
        f"{record['question']}\n"
        f"{format_options(record['options'])}\n\n"
        "答案："
    )


def build_strict_prompt(record: dict[str, Any]) -> str:
    return (
        f"以下是中国关于{record['subject_cn']}考试的单项选择题。\n"
        "请阅读题目和四个选项，选出唯一正确的答案。\n"
        "只输出一个选项字母（A、B、C 或 D），不要输出解释，不要输出其他文字。\n\n"
        f"{record['question']}\n"
        f"{format_options(record['options'])}\n\n"
        "答案："
    )


def build_cot_prompt(record: dict[str, Any]) -> str:
    return (
        f"以下是中国关于{record['subject_cn']}考试的单项选择题。\n"
        "请先简要分析题目，再给出最终选项。\n"
        "最后一行必须严格写成“答案：X”，其中 X 是 A、B、C、D 中的一个。\n\n"
        f"{record['question']}\n"
        f"{format_options(record['options'])}\n\n"
        "请开始作答。"
    )


def build_fewshot_prompt(record: dict[str, Any], examples: list[dict[str, Any]]) -> str:
    if not examples:
        raise ValueError(f"No few-shot examples for subject={record['subject']}")
    example_blocks = []
    for index, example in enumerate(examples, start=1):
        example_blocks.append(
            f"示例 {index}：\n"
            f"{example['question']}\n"
            f"{format_options(example['options'])}\n"
            f"答案：{example['answer']}"
        )
    return (
        f"以下是中国关于{record['subject_cn']}考试的单项选择题。\n"
        "请参考示例的答题格式，选出唯一正确的答案。\n\n"
        + "\n\n".join(example_blocks)
        + "\n\n现在回答下面的题目：\n"
        + f"{record['question']}\n"
        f"{format_options(record['options'])}\n\n"
        "答案："
    )


def build_selfcheck_prompt(record: dict[str, Any], first_output: str) -> str:
    return (
        "你在回答一道中文单项选择题。请检查题目、选项和第一轮回答，"
        "判断第一轮答案是否正确；如果错误，请改正。\n"
        "最后一行必须严格写成“答案：X”，其中 X 是 A、B、C、D 中的一个。\n\n"
        f"{record['question']}\n"
        f"{format_options(record['options'])}\n\n"
        f"第一轮回答：{first_output}\n\n"
        "复核结果："
    )


def build_prompt(
    record: dict[str, Any],
    template: str,
    examples_by_subject: dict[str, list[dict[str, Any]]] | None = None,
) -> str:
    if template == "baseline":
        return build_baseline_prompt(record)
    if template == "strict":
        return build_strict_prompt(record)
    if template == "cot":
        return build_cot_prompt(record)
    if template == "fewshot":
        examples = (examples_by_subject or {}).get(record["subject"], [])
        return build_fewshot_prompt(record, examples)
    raise ValueError(f"Unknown prompt template: {template}")
