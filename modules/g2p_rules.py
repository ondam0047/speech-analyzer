"""순수 파이썬 한국어 g2p (규칙 기반) — 네이티브 의존성 없음.

조음 분석은 어절 단위로 비교하므로 단어(어절) 단위 발음형을 생성한다.
의무적(자연스러운) 음운변동을 적용해, 자연 변동이 오류로 오인되지 않게 한다.

다루는 규칙:
- 연음(초성우선원리), ㅎ탈락, 격음화, 구개음화
- 종성 중화(평파열음화), 자음군 단순화
- 비음화, 유음화, 경음화

형태소 경계가 필요한 일부 규칙(사이시옷·ㄴ첨가·용언 어간 경음화 등)은
완전하지 않을 수 있어 임상가 검수가 필요하다(자동 분석의 한계).
"""

from __future__ import annotations

from modules.jamo_split import CHO, JONG, JUNG, decompose_char

_JONG_IDX = {j: i for i, j in enumerate(JONG)}
_BASE = 0xAC00


def _compose(cho: str, jung: str, jong: str = "") -> str:
    return chr(_BASE + CHO.index(cho) * 588 + JUNG.index(jung) * 28 + _JONG_IDX.get(jong, 0))


# 자음군: (앞에 남는 받침, 연음 시 뒤로 넘어가는 자음)
_CLUSTER_SPLIT = {
    "ㄳ": ("ㄱ", "ㅅ"), "ㄵ": ("ㄴ", "ㅈ"), "ㄶ": ("ㄴ", "ㅎ"),
    "ㄺ": ("ㄹ", "ㄱ"), "ㄻ": ("ㄹ", "ㅁ"), "ㄼ": ("ㄹ", "ㅂ"),
    "ㄽ": ("ㄹ", "ㅅ"), "ㄾ": ("ㄹ", "ㅌ"), "ㄿ": ("ㄹ", "ㅍ"), "ㅀ": ("ㄹ", "ㅎ"),
    "ㅄ": ("ㅂ", "ㅅ"),
}
# 자음군이 자음 앞/어말에서 단순화될 때 남는 받침
_CLUSTER_CODA = {
    "ㄳ": "ㄱ", "ㄵ": "ㄴ", "ㄶ": "ㄴ", "ㄺ": "ㄱ", "ㄻ": "ㅁ", "ㄼ": "ㄹ",
    "ㄽ": "ㄹ", "ㄾ": "ㄹ", "ㄿ": "ㅂ", "ㅀ": "ㄹ", "ㅄ": "ㅂ",
}
# 종성 중화(평파열음화)
_NEUTRAL = {
    "ㄲ": "ㄱ", "ㅋ": "ㄱ", "ㅅ": "ㄷ", "ㅆ": "ㄷ", "ㅈ": "ㄷ", "ㅊ": "ㄷ",
    "ㅌ": "ㄷ", "ㅎ": "ㄷ", "ㅍ": "ㅂ",
}
_TENSE = {"ㄱ": "ㄲ", "ㄷ": "ㄸ", "ㅂ": "ㅃ", "ㅅ": "ㅆ", "ㅈ": "ㅉ"}
_ASPIRATE_BY_H = {"ㄱ": "ㅋ", "ㄷ": "ㅌ", "ㅈ": "ㅊ", "ㅂ": "ㅍ", "ㅅ": "ㅆ"}
_H_ASPIRATE = {"ㄱ": "ㅋ", "ㄷ": "ㅌ", "ㅈ": "ㅊ", "ㅅ": "ㅆ"}  # ㅎ + C
_STOPS = {"ㄱ", "ㄷ", "ㅂ"}


def _neutralize(coda: str) -> str:
    if not coda:
        return ""
    if coda in _CLUSTER_CODA:
        return _CLUSTER_CODA[coda]
    return _NEUTRAL.get(coda, coda)


def _g2p_syllables(syls: list[list[str]]) -> None:
    """[ [초,중,종], ... ] in-place 변환(보통 자음 음운변동 적용)."""
    n = len(syls)
    for i in range(n - 1):
        cur, nxt = syls[i], syls[i + 1]
        coda, onset = cur[2], nxt[0]
        if not coda:
            continue
        is_h_cluster = coda in ("ㄶ", "ㅀ")
        base_coda = "ㄴ" if coda == "ㄶ" else ("ㄹ" if coda == "ㅀ" else coda)

        # 구개음화(ㅎ 결합): 닫히/굳히 → 다치/구치 (격음화보다 우선)
        if coda in ("ㄷ", "ㅌ") and onset == "ㅎ" and nxt[1] == "ㅣ":
            nxt[0] = "ㅊ"; cur[2] = ""; continue

        # 격음화
        if coda == "ㅎ" and onset in _H_ASPIRATE:
            nxt[0] = _H_ASPIRATE[onset]; cur[2] = ""; continue
        if is_h_cluster and onset in _H_ASPIRATE:
            nxt[0] = _H_ASPIRATE[onset]; cur[2] = base_coda; continue
        if coda in ("ㄺ", "ㄼ", "ㄵ") and onset == "ㅎ":
            keep, moved = _CLUSTER_SPLIT[coda]
            nxt[0] = _ASPIRATE_BY_H.get(moved, moved); cur[2] = keep; continue
        if _neutralize(coda) in _STOPS and onset == "ㅎ":
            nxt[0] = _ASPIRATE_BY_H[_neutralize(coda)]; cur[2] = ""; continue

        # ㅎ 탈락 (받침 ㅎ + 모음)
        if onset == "ㅇ":
            if coda == "ㅎ":
                cur[2] = ""; continue
            if is_h_cluster:
                nxt[0] = base_coda; cur[2] = ""; continue

        # 구개음화 (ㄷ/ㅌ + 이)
        if coda in ("ㄷ", "ㅌ") and nxt[1] == "ㅣ" and onset == "ㅇ":
            nxt[0] = "ㅈ" if coda == "ㄷ" else "ㅊ"; cur[2] = ""; continue

        # 연음 (받침 + 모음). ㅇ 받침은 이동하지 않음.
        if onset == "ㅇ":
            if coda in _CLUSTER_SPLIT:
                keep, moved = _CLUSTER_SPLIT[coda]
                cur[2] = keep
                nxt[0] = "ㅆ" if moved == "ㅅ" else moved  # ㅅ계 자음군은 경음
            elif coda == "ㅇ":
                pass
            else:
                nxt[0] = coda; cur[2] = ""
            continue

        # ㄺ + ㄱ → ㄹ + ㄲ (용언 어간 특수)
        if coda == "ㄺ" and onset == "ㄱ":
            cur[2] = "ㄹ"; nxt[0] = "ㄲ"; continue

        # 자음 앞: 중화 → 비음화/유음화 → 경음화
        cluster_tensing = coda in ("ㄵ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ")
        nc = _neutralize(coda)
        if coda == "ㄼ" and cur[0] == "ㅂ" and cur[1] == "ㅏ":  # 밟- 예외
            nc = "ㅂ"

        # 유음화
        if (nc == "ㄴ" and onset == "ㄹ") or (nc == "ㄹ" and onset == "ㄴ"):
            cur[2] = "ㄹ"; nxt[0] = "ㄹ"; continue

        # 비음화
        if nc in _STOPS and onset in ("ㄴ", "ㅁ"):
            nc = {"ㄱ": "ㅇ", "ㄷ": "ㄴ", "ㅂ": "ㅁ"}[nc]
            cur[2] = nc; continue
        if onset == "ㄹ" and nc in (_STOPS | {"ㅁ", "ㅇ"}):
            nxt[0] = "ㄴ"
            if nc in _STOPS:
                nc = {"ㄱ": "ㅇ", "ㄷ": "ㄴ", "ㅂ": "ㅁ"}[nc]
                cur[2] = nc
                # ㄴ 뒤 ㄹ→ㄴ 후 다시 비음화 종결
            else:
                cur[2] = nc
            continue

        # 경음화
        if (nc in _STOPS or cluster_tensing) and onset in _TENSE:
            nxt[0] = _TENSE[onset]
        cur[2] = nc

    # 어말(또는 변환 후 남은) 받침 중화
    if syls:
        syls[-1][2] = _neutralize(syls[-1][2])


def rule_g2p_word(word: str) -> str:
    """단어(어절) → 발음형. 한글만 변환, 그 외 문자는 그대로 둔다."""
    word = (word or "").strip()
    if not word:
        return ""
    out: list[str] = []
    run: list[list[str]] = []

    def _flush():
        if run:
            _g2p_syllables(run)
            out.extend(_compose(c, j, k) for c, j, k in run)
            run.clear()

    for ch in word:
        d = decompose_char(ch)
        if d is None:
            _flush()
            out.append(ch)
        else:
            run.append([d[0], d[1], d[2]])
    _flush()
    return "".join(out)
