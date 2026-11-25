from ceval_eval.parsing import last_nonempty_line, majority_vote, parse_answer


def test_parse_answer_prefers_last_line():
    assert parse_answer("A 看起来不对\n答案：D") == "D"
    assert parse_answer("推理中提到 B，但最后没有给答案") is None


def test_parse_answer_common_formats():
    assert parse_answer("答案：C") == "C"
    assert parse_answer("最终答案是 B") == "B"
    assert parse_answer("所以选 D。") == "D"
    assert parse_answer("C") == "C"
    assert parse_answer("答案：Ｃ") == "C"


def test_parse_answer_ignores_letters_inside_reasoning():
    assert parse_answer("A 是干扰项，B 也不对，但我还没决定") is None


def test_last_nonempty_line():
    assert last_nonempty_line("第一行\n\n  \n最后一行  \n") == "最后一行"


def test_majority_vote_tie_break():
    assert majority_vote(["A", "B", "A", "B"]) == "A"
    assert majority_vote([None, "B", "B"]) == "B"
    assert majority_vote([None, None]) is None
