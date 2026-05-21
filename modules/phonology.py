"""한국어 자음 자질표 + 오류 음운변동(상대분석) 분류.

참고: 김민정(2006) '아동용 조음검사에 나타난 취학 전 아동의 음운 오류패턴',
배소영·김민정 APAC, U-TAP2 등 한국 임상에서 통용되는 음운변동 범주.

설계 원칙:
- 의무적(자연스러운) 음운변동(경음화·비음화·유음화·격음화·구개음화·종성중화 등)은
  목표어 g2p 발음형에 이미 반영되므로 '오류'로 잡히지 않는다.
- 여기서는 목표 발음형 대비 산출형의 차이(=오류)만 음운변동으로 분류한다.
- 모음(중성)은 음절핵이라 생략되지 않는다 → 모음 생략은 음절 생략으로 해석.
- 하나의 대치는 여러 변동에 동시에 해당할 수 있다(예: ㅋ→ㄷ = 전방화 + 이완음화).
"""

from __future__ import annotations

# 자음 자질: 자모 -> (조음위치, 조음방법, 발성유형)
CONSONANT_FEATURES: dict[str, tuple[str, str, str]] = {
    "ㄱ": ("연구개", "파열", "평음"), "ㄲ": ("연구개", "파열", "경음"), "ㅋ": ("연구개", "파열", "격음"),
    "ㄷ": ("치조", "파열", "평음"), "ㄸ": ("치조", "파열", "경음"), "ㅌ": ("치조", "파열", "격음"),
    "ㅂ": ("양순", "파열", "평음"), "ㅃ": ("양순", "파열", "경음"), "ㅍ": ("양순", "파열", "격음"),
    "ㅅ": ("치조", "마찰", "평음"), "ㅆ": ("치조", "마찰", "경음"), "ㅎ": ("성문", "마찰", "평음"),
    "ㅈ": ("경구개", "파찰", "평음"), "ㅉ": ("경구개", "파찰", "경음"), "ㅊ": ("경구개", "파찰", "격음"),
    "ㄴ": ("치조", "비음", "공명"), "ㅁ": ("양순", "비음", "공명"), "ㅇ": ("연구개", "비음", "공명"),
    "ㄹ": ("치조", "유음", "공명"),
}

# 조음위치 전후 순서(앞 ← → 뒤). 산출 위치가 더 앞이면 전방화, 뒤면 후방화.
_PLACE_ORDER = {"양순": 1, "치조": 2, "경구개": 3, "연구개": 4, "성문": 5}

_MANNER_PROCESS = {
    "파열": "파열음화", "마찰": "마찰음화", "파찰": "파찰음화", "비음": "비음화", "유음": "유음화",
}

# 비전형(비발달적) 패턴 — 정상발달에서 드물어 장애를 시사(임상가 검수 필요).
ATYPICAL_PROCESSES = {
    "어두초성생략", "후방화", "첨가", "도치", "탈비음화", "성문음화", "음절생략",
}


def classify_substitution(target: str, produced: str) -> list[str]:
    """자음 대치(target→produced)를 오류 음운변동으로 분류(복수 가능)."""
    tf = CONSONANT_FEATURES.get(target)
    pf = CONSONANT_FEATURES.get(produced)
    if not tf or not pf:
        return ["대치"]
    t_place, t_manner, t_phon = tf
    p_place, p_manner, p_phon = pf
    procs: list[str] = []

    # 조음방법 변동
    if t_manner != p_manner:
        if t_manner == "유음":
            procs.append("유음의단순화")
        elif t_manner == "비음" and p_manner != "비음":
            procs.append("탈비음화")
            if p_manner in _MANNER_PROCESS:
                procs.append(_MANNER_PROCESS[p_manner])
        else:
            procs.append(_MANNER_PROCESS.get(p_manner, "조음방법변동"))

    # 조음위치 변동
    if t_place != p_place:
        if p_place == "성문":
            procs.append("성문음화")
        elif _PLACE_ORDER[p_place] < _PLACE_ORDER[t_place]:
            procs.append("전방화")
            if t_place == "연구개":
                procs.append("연구개음의전방화")
        else:
            procs.append("후방화")

    # 발성유형 변동(장애음 간)
    if t_phon in ("평음", "경음", "격음") and p_phon in ("평음", "경음", "격음") and t_phon != p_phon:
        if p_phon == "경음":
            procs.append("긴장음화")
        elif p_phon == "격음":
            procs.append("기식음화")
        else:  # 평음
            procs.append("이완음화")

    return procs or ["대치"]


def classify_omission(position: str) -> str:
    """자음 생략을 위치별 음운변동으로 분류."""
    return {
        "어두초성": "어두초성생략", "어중초성": "어중초성생략",
        "어말종성": "어말종성생략", "어중종성": "어중종성생략", "종성": "종성생략",
    }.get(position, "생략")


def is_atypical(process: str) -> bool:
    return process in ATYPICAL_PROCESSES
