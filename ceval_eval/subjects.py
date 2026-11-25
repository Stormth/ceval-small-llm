from __future__ import annotations

# Chinese names and categories follow the official C-Eval subject_mapping.json.
# Only the subjects used by this project's fixed 8 x 25 subset are listed here.
SUBJECT_META: dict[str, dict[str, str]] = {
    "college_economics": {
        "name_en": "College Economics",
        "name_cn": "大学经济学",
        "category": "Social Science",
    },
    "accountant": {
        "name_en": "Accountant",
        "name_cn": "注册会计师",
        "category": "Other",
    },
    "college_programming": {
        "name_en": "College Programming",
        "name_cn": "大学编程",
        "category": "STEM",
    },
    "electrical_engineer": {
        "name_en": "Electrical Engineer",
        "name_cn": "注册电气工程师",
        "category": "STEM",
    },
    "art_studies": {
        "name_en": "Art Studies",
        "name_cn": "艺术学",
        "category": "Humanities",
    },
    "teacher_qualification": {
        "name_en": "Teacher Qualification",
        "name_cn": "教师资格",
        "category": "Social Science",
    },
    "environmental_impact_assessment_engineer": {
        "name_en": "Environmental Impact Assessment Engineer",
        "name_cn": "环境影响评价工程师",
        "category": "Other",
    },
    "civil_servant": {
        "name_en": "Civil Servant",
        "name_cn": "公务员",
        "category": "Other",
    },
}


def subject_name_cn(subject: str) -> str:
    return SUBJECT_META.get(subject, {}).get("name_cn", subject)


def subject_category(subject: str) -> str:
    return SUBJECT_META.get(subject, {}).get("category", "Other")
