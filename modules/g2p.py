"""g2p 래퍼 — 목표어(표준 철자) → 발음형 변환.

g2pkk(메캡 불필요) 우선, 없으면 g2pk 사용.

g2pkk는 init 시 nltk `cmudict` 데이터를 런타임 다운로드한다. 배포 환경에서
네트워크가 제한되면 다운로드가 실패해 변환이 원문 그대로 반환되므로(자연
음운변동이 오류로 오인됨), 레포에 동봉한 nltk_data를 우선 경로로 등록한다.
"""

from __future__ import annotations

import os

_BUNDLED_NLTK = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nltk_data")
try:  # 번들 cmudict를 우선 사용 → 런타임 다운로드 불필요
    import nltk

    if os.path.isdir(_BUNDLED_NLTK) and _BUNDLED_NLTK not in nltk.data.path:
        nltk.data.path.insert(0, _BUNDLED_NLTK)
except Exception:
    pass


def _load_g2p():
    try:
        from g2pkk import G2p
        return G2p()
    except Exception:
        from g2pk import G2p
        return G2p()


class G2PConverter:
    """목표어 → 발음형 변환기."""

    def __init__(self) -> None:
        self._g2p = _load_g2p()

    def to_pronunciation(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        try:
            return self._g2p(text)
        except Exception:
            return text  # 변환 실패 시 원문 유지

    def to_pronunciation_words(self, text: str) -> str:
        """어절 단위로 g2p 적용 후 공백 결합.

        조음 분석은 어절 단위로 정렬·비교하므로 발음형도 어절 단위로 만든다.
        문장 전체를 한 번에 변환하면 어절 경계를 넘는 연음이 잘못 적용된다
        (예: '지금 있는 이' → '지그 민느 니'). 어절 단위면 '지금 인는 이'.
        """
        text = (text or "").strip()
        if not text:
            return ""
        return " ".join(self.to_pronunciation(w) for w in text.split())
