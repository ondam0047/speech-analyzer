"""말명료도(이해가능도) — 전사에서 못 알아들은 부분(*)으로 계산.

표기 규칙: 대상자가 말했으나 알아듣지 못한 부분은 음절 수만큼 '*'로 표기한다.
예) 3음절을 못 알아들었으면 '***', 부분적으로는 '엄***'(첫 음절만 이해).

지표:
- 어절 명료도 = ('*'가 없는 어절 수) / (전체 어절 수) × 100
- 음절 명료도 = (전체 음절 - 불명료 음절) / 전체 음절 × 100
  · 전체 음절 = 한글 음절 수 + '*' 개수,  불명료 음절 = '*' 개수
"""

from __future__ import annotations


def _syllables(text: str) -> tuple[int, int]:
    """(전체 음절, 불명료 음절). 전체 = 한글 음절 + '*' 개수, 불명료 = '*' 개수."""
    hangul = sum(1 for ch in text if 0xAC00 <= ord(ch) <= 0xD7A3)
    stars = text.count("*")
    return hangul + stars, stars


def compute_intelligibility(utterances: list[str]) -> dict:
    """아동 발화 리스트 → 말명료도(어절·음절 기준) + 발화별 상세."""
    total_words = intel_words = 0
    total_syl = unintel_syl = 0
    per: list[dict] = []
    for u in utterances:
        s = str(u or "").strip()
        if not s:
            continue
        words = s.split()
        wt = len(words)
        wi = sum(1 for w in words if "*" not in w)
        syl_t, stars = _syllables(s)
        total_words += wt
        intel_words += wi
        total_syl += syl_t
        unintel_syl += stars
        per.append({
            "utterance": s, "words": wt, "intelligible_words": wi,
            "unintelligible_words": wt - wi,
            "syllables": syl_t, "unintelligible_syllables": stars,
        })
    word_pct = round(intel_words / total_words * 100, 1) if total_words else 0.0
    syl_pct = (round((total_syl - unintel_syl) / total_syl * 100, 1)
               if total_syl else 0.0)
    return {
        "total_words": total_words,
        "intelligible_words": intel_words,
        "unintelligible_words": total_words - intel_words,
        "word_intelligibility": word_pct,
        "total_syllables": total_syl,
        "unintelligible_syllables": unintel_syl,
        "syllable_intelligibility": syl_pct,
        "per_utterance": per,
    }
