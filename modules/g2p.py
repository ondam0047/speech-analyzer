"""g2p 래퍼 — 목표어(표준 철자) → 발음형.

순수 파이썬 규칙 기반(modules.g2p_rules)으로 동작한다. 네이티브 의존성
(mecab/g2pkk/nltk)과 런타임 다운로드가 없어 배포 환경에서 안정적으로 동작한다.
의무적 음운변동(연음·경음화·비음화·유음화·격음화·구개음화·종성중화 등)을
어절 단위로 적용해, 자연 변동이 조음 오류로 오인되지 않게 한다.
"""

from __future__ import annotations

from modules.g2p_rules import rule_g2p_word


class G2PConverter:
    """목표어 → 발음형 변환기(규칙 기반)."""

    def to_pronunciation(self, text: str) -> str:
        """어절(또는 단어) 단위 발음형."""
        return rule_g2p_word(text)

    def to_pronunciation_words(self, text: str) -> str:
        """문장을 어절 단위로 변환 후 공백 결합(어절 경계 연음 오류 방지)."""
        text = (text or "").strip()
        if not text:
            return ""
        return " ".join(rule_g2p_word(w) for w in text.split())
