"""한국 아동 말소리(자음·모음) 발달 규준 — LLM 임상 코멘트의 연령 기반 근거.

생활연령 대비 '연령에 적절한 오류'인지 '주목해야 할 오류'인지 판단할 근거를 제공한다.
규준은 참고용이며, 공식 판정은 표준화 검사(U-TAP2/APAC)의 백분위/표준점수로 해야 한다.

[참고 문헌]
- 김영태 (1996). 그림자음검사를 이용한 취학전 아동의 자음정확도 연구.
  말-언어장애연구, 1, 7-33. (자음 발달 4단계: 출현·관습적·숙달·완전습득)
- 김영태·신문자 (2004). 우리말 조음·음운평가(U-TAP). 학지사.
- 김민정·배소영·박창일 (2007). 아동용 발음평가(APAC).
- 김민정 (2006). '아동용 조음검사'에 나타난 취학 전 아동의 음운 오류패턴.
  Communication Sciences & Disorders, 11(2).
- 김수진·신지영 (2020). 말소리장애(2판). 시그마프레스.
- 송인미·성철재 (2018). 만 2-4세 한국 아동의 단모음과 이중모음 산출 특징.
  말소리와 음성과학, 10(1).
- 하승희 외, U-TAP2 (2020) 표준화: 만 2세 후반~7세, 연령 증가에 따른 자음정확도 발달.
"""

from __future__ import annotations

# 자음 완전습득(95~100% 정확) 기대 연령(세) — 김영태(1996) 4단계 중 완전습득 단계 기준.
# 같은 연령대의 자음은 그 연령 무렵 대부분 아동이 정확히 산출한다.
# (ㅎ은 자료에 따라 차이가 있어 근사치로 3세에 배치 — 임상가 검수.)
CONSONANT_MASTERY_YEAR: dict[str, int] = {
    "ㅁ": 2, "ㅍ": 2, "ㅇ": 2,
    "ㅂ": 3, "ㅃ": 3, "ㄸ": 3, "ㅌ": 3, "ㅎ": 3,
    "ㄴ": 4, "ㄷ": 4, "ㄲ": 4,
    "ㄱ": 5, "ㅋ": 5, "ㅈ": 5, "ㅉ": 5, "ㅊ": 5, "ㅆ": 5,
    "ㅅ": 6, "ㄹ": 6,
}

# 김영태(1996) 완전습득 연령대별 자음(표시용).
CONSONANT_STAGE_TABLE: list[tuple[str, str]] = [
    ("2;0–2;11", "ㅁ, ㅍ, ㅇ"),
    ("3;0–3;11", "ㅂ, ㅃ, ㄸ, ㅌ, ㅎ"),
    ("4;0–4;11", "ㄴ, ㄷ, ㄲ"),
    ("5;0–5;11", "ㄱ, ㅋ, ㅈ, ㅉ, ㅊ, ㅆ"),
    ("6;0–6;11", "ㅅ, ㄹ"),
]

# 모음(중성) 발달: 단모음은 자음보다 일찍, 이중모음은 더 늦게.
VOWEL_NOTE = (
    "단모음(ㅏㅓㅗㅜㅡㅣㅐ/ㅔ)은 대체로 만 3;0 이전에 90% 이상 정확히 산출되어 "
    "자음보다 일찍 안정된다. 이중모음(ㅑㅕㅛㅠㅘㅝ 등)은 만 3~4세에 발달하며, "
    "ㅢ·ㅟ·ㅚ 등 일부는 더 늦게까지 오류가 보일 수 있다. (송인미·성철재, 2018)"
)

# 발달적 음운변동의 대략적 소멸(억제) 연령(세). 이 연령 이후에도 빈번하면 주목.
PROCESS_RESOLUTION_YEAR: dict[str, int] = {
    "파열음화": 4, "긴장음화": 4, "이완음화": 4, "기식음화": 4,
    "연구개음의전방화": 4, "전방화": 4,
    "종성생략": 4, "어말종성생략": 4, "어중종성생략": 4,
    "마찰음화": 5, "파찰음화": 5,
    "유음의단순화": 6, "유음화": 6,
}

# 비전형(비발달적) 패턴 — 연령과 무관하게 정상발달에서 드물어 장애를 시사.
ATYPICAL_NOTE = (
    "후방화·어두초성생략·도치·탈비음화·성문음화·음절생략 등 비전형(비발달적) 패턴은 "
    "연령과 무관하게 정상발달에서 드물며 말소리장애를 시사할 수 있다."
)


def _pcc_expectation(age_years: float) -> str:
    """연령별 대략적 자음정확도(PCC) 기대 범위(참고)."""
    if age_years < 3:
        return "약 70~85% (개인차 큼)"
    if age_years < 4:
        return "약 85~92%"
    if age_years < 5:
        return "약 90~96%"
    if age_years < 6:
        return "약 95% 이상"
    return "거의 100%에 근접"


def _intelligibility_expectation(age_years: float) -> str:
    """연령별 대략적 말명료도 기대(친숙하지 않은 청자 기준, 참고)."""
    if age_years < 3:
        return "약 50~75%"
    if age_years < 4:
        return "약 75~90%"
    if age_years < 5:
        return "약 90~100%"
    return "거의 100% (이 연령 이후 낮으면 주목)"


def developmental_reference(age_months: int | None) -> dict | None:
    """생활연령(개월) → 발달 규준 비교 정보. 연령 미상이면 None."""
    if not age_months or age_months <= 0:
        return None
    age_years = age_months / 12.0
    mastered = sorted(
        [c for c, y in CONSONANT_MASTERY_YEAR.items() if y <= age_years],
        key=lambda c: CONSONANT_MASTERY_YEAR[c])
    developing = sorted(
        [c for c, y in CONSONANT_MASTERY_YEAR.items() if y > age_years],
        key=lambda c: CONSONANT_MASTERY_YEAR[c])
    resolved = sorted({p for p, y in PROCESS_RESOLUTION_YEAR.items() if y <= age_years})
    return {
        "age_months": age_months,
        "age_years": round(age_years, 1),
        "mastered_expected": mastered,      # 이 연령엔 정확해야 할 자음(오류 시 주목)
        "developing": developing,           # 아직 발달 중(오류가 연령상 정상일 수 있음)
        "expected_pcc": _pcc_expectation(age_years),
        "expected_intelligibility": _intelligibility_expectation(age_years),
        "processes_should_resolve": resolved,  # 이 연령엔 거의 사라졌어야 할 발달적 변동
    }


def reference_text(age_months: int | None) -> str:
    """LLM 프롬프트용 연령 기반 발달 규준 텍스트."""
    stage = "\n".join(f"  · {band}: {cs}" for band, cs in CONSONANT_STAGE_TABLE)
    head = (
        "[자음 완전습득(95~100%) 기대 연령 — 김영태(1996)]\n"
        f"{stage}\n"
        f"[모음] {VOWEL_NOTE}\n"
        f"[비전형 패턴] {ATYPICAL_NOTE}"
    )
    ref = developmental_reference(age_months)
    if ref is None:
        return (head + "\n[생활연령] 미입력 — 연령 대비 적절성 판단은 제한적입니다. "
                "사이드바에 생년월일/검사일을 입력하면 연령 기반 비교가 가능합니다.")
    y, m = divmod(ref["age_months"], 12)
    return (
        f"{head}\n\n"
        f"[이 대상자: 생활연령 {y}세 {m}개월 ({ref['age_months']}개월) 기준 비교]\n"
        f"  · 이 연령에 완전습득이 기대되는 자음(오류 시 임상적으로 주목): "
        f"{', '.join(ref['mastered_expected']) or '없음'}\n"
        f"  · 아직 발달 중이라 오류가 연령상 정상일 수 있는 자음: "
        f"{', '.join(ref['developing']) or '없음'}\n"
        f"  · 이 연령에 대부분 사라졌어야 할 발달적 음운변동: "
        f"{', '.join(ref['processes_should_resolve']) or '해당 없음'}\n"
        f"  · 연령 기대 자음정확도(PCC): {ref['expected_pcc']} / "
        f"말명료도: {ref['expected_intelligibility']} (참고)\n"
        "→ 관찰된 오류가 '연령상 정상 범위'인지 '연령 대비 지연/주목 대상'인지 위 기준으로 판단하세요. "
        "단, 최종 판정은 표준화 검사(U-TAP2/APAC)의 백분위로 확인해야 합니다."
    )


REFERENCES = [
    "김영태 (1996). 그림자음검사를 이용한 취학전 아동의 자음정확도 연구. 말-언어장애연구, 1, 7-33.",
    "김영태·신문자 (2004). 우리말 조음·음운평가(U-TAP). 학지사.",
    "김민정·배소영·박창일 (2007). 아동용 발음평가(APAC).",
    "김민정 (2006). '아동용 조음검사'에 나타난 취학 전 아동의 음운 오류패턴. CSD, 11(2).",
    "송인미·성철재 (2018). 만 2-4세 한국 아동의 단모음과 이중모음 산출 특징. 말소리와 음성과학, 10(1).",
    "U-TAP2 (2020) 표준화 연구: 만 2;6~7세 자음정확도 발달.",
    "김수진·신지영 (2020). 말소리장애(2판). 시그마프레스.",
]
