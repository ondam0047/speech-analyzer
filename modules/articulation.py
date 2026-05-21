"""조음 분석 — 목표어/산출형 비교.

입력: [(목표어, 산출형), ...]
처리:
  1. 목표어 → 발음형(g2p)
  2. 어절 정렬 → 어절 내 음소(자모) 정렬(Needleman-Wunsch)
  3. 자음(초성·종성, 초성 ㅇ 제외) 기준으로 컨퓨전 매트릭스 / PCC / 위치별 오류 산출
  4. 모음(중성)은 PCC와 분리해 PVC(모음정확도)로 별도 집계(음절핵이라 생략 불가)
  5. 오류(대치/생략/첨가)를 한국어 음운변동(상대분석)으로 분류
산출: confusion_matrix, position_errors, pcc, phoneme_accuracy, errors,
      pvc, vowel_accuracy, vowel_confusion_matrix, vowel_errors, additions_detail,
      phonological_processes

* 의무적(자연스러운) 음운변동은 목표어 g2p 발음형에 이미 반영 → 오류로 잡히지 않음.
"""

from __future__ import annotations

import difflib
from collections import Counter, defaultdict

from modules.g2p import G2PConverter
from modules.jamo_split import word_phonemes
from modules.phonology import classify_omission, classify_substitution, is_atypical

OMISSION = "∅"  # 생략(대응 산출 음소 없음)


def _nw_align(a: list, b: list, score, gap: float = -1.0) -> list[tuple]:
    """Needleman-Wunsch 전역 정렬. (a_index|None, b_index|None) 리스트 반환."""
    n, m = len(a), len(b)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + gap
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = max(
                dp[i - 1][j - 1] + score(a[i - 1], b[j - 1]),
                dp[i - 1][j] + gap,
                dp[i][j - 1] + gap,
            )
    i, j, out = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + score(a[i - 1], b[j - 1]):
            out.append((i - 1, j - 1)); i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + gap:
            out.append((i - 1, None)); i -= 1
        else:
            out.append((None, j - 1)); j -= 1
    out.reverse()
    return out


def _align_words(tw: list[str], pw: list[str]) -> list[tuple]:
    """어절 정렬. 개수 같으면 인덱스 zip, 다르면 유사도 기반 정렬."""
    if len(tw) == len(pw):
        return [(i, i) for i in range(len(tw))]

    def sim(x, y):
        return 2 * difflib.SequenceMatcher(None, x, y).ratio() - 1
    return _nw_align(tw, pw, sim, gap=-0.5)


def _is_consonant(ph: dict) -> bool:
    """자음 여부. 초성 ㅇ(무음 초성)은 제외."""
    if ph["kind"] == "종성":
        return True
    return ph["kind"] == "초성" and ph["jamo"] != "ㅇ"


def _position(ph: dict) -> str:
    if ph["kind"] == "초성":
        return "어두초성" if ph["syllable"] == 0 else "어중초성"
    if ph["kind"] == "종성":
        return "종성"
    return "중성"


def _fine_position(ph: dict, last_syl: int) -> str:
    """음운변동 분류용 세부 위치(종성을 어말/어중으로 구분)."""
    if ph["kind"] == "초성":
        return "어두초성" if ph["syllable"] == 0 else "어중초성"
    if ph["kind"] == "종성":
        return "어말종성" if ph["syllable"] == last_syl else "어중종성"
    return "중성"


POSITION_ORDER = ["어두초성", "어중초성", "종성"]


def analyze_articulation(pairs: list[tuple[str, str]]) -> dict:
    g2p = G2PConverter()
    confusion: dict[str, Counter] = defaultdict(Counter)
    position_errors: Counter = Counter()
    position_total: Counter = Counter()
    phoneme_total: Counter = Counter()
    phoneme_correct: Counter = Counter()
    errors: list[dict] = []
    # 모음(중성) — 자음 PCC와 별도 집계(PVC). 음절핵이라 생략되지 않음.
    vowel_confusion: dict[str, Counter] = defaultdict(Counter)
    vowel_total: Counter = Counter()
    vowel_correct: Counter = Counter()
    vowel_errors: list[dict] = []
    additions = 0
    additions_detail: list[dict] = []
    # 오류 음운변동(상대분석) 집계
    process_counter: Counter = Counter()
    syllable_omissions = 0

    for target_text, produced_text in pairs:
        tw = (target_text or "").split()
        pw = (produced_text or "").split()
        for ti, pj in _align_words(tw, pw):
            t_word = tw[ti] if ti is not None else None
            p_word = pw[pj] if pj is not None else None
            if t_word is None:
                # 목표어 없는 산출 어절 전체가 첨가 — 자음만 카운트(상세 포함)
                for pph in word_phonemes(p_word or ""):
                    if _is_consonant(pph):
                        additions += 1
                        process_counter["첨가"] += 1
                        additions_detail.append({
                            "produced": pph["jamo"], "position": _position(pph),
                            "target_word": "", "produced_word": p_word or "",
                        })
                continue
            t_pron = g2p.to_pronunciation(t_word)
            t_phs = word_phonemes(t_pron)
            p_phs = word_phonemes(p_word or "")
            last_syl = max((x["syllable"] for x in t_phs), default=0)
            ops = _nw_align(
                [x["jamo"] for x in t_phs], [x["jamo"] for x in p_phs],
                lambda x, y: 1.0 if x == y else -1.0,
            )
            for ai, bj in ops:
                tph = t_phs[ai] if ai is not None else None
                pph = p_phs[bj] if bj is not None else None
                if tph is None:
                    if pph is not None and _is_consonant(pph):
                        additions += 1
                        process_counter["첨가"] += 1
                        additions_detail.append({
                            "produced": pph["jamo"], "position": _position(pph),
                            "target_word": t_word, "produced_word": p_word or "",
                        })
                    continue
                target_j = tph["jamo"]
                if _is_consonant(tph):
                    pos = _position(tph)
                    phoneme_total[target_j] += 1
                    position_total[pos] += 1
                    if pph is not None and pph["jamo"] == target_j:
                        phoneme_correct[target_j] += 1
                    else:
                        # 산출이 없거나 무음 초성(ㅇ)이면 자음 생략으로 처리
                        null_onset = (pph is not None and pph["kind"] == "초성"
                                      and pph["jamo"] == "ㅇ")
                        if pph is None or null_onset:
                            produced_j = OMISSION
                            procs = [classify_omission(_fine_position(tph, last_syl))]
                        elif _is_consonant(pph):
                            produced_j = pph["jamo"]
                            procs = classify_substitution(target_j, produced_j)
                        else:
                            produced_j = pph["jamo"]
                            procs = ["대치"]  # 자음→모음(정렬 잡음)
                        confusion[target_j][produced_j] += 1
                        position_errors[pos] += 1
                        for pr in procs:
                            process_counter[pr] += 1
                        errors.append({
                            "word": t_word, "target_pron": t_pron,
                            "produced_word": p_word or "",
                            "target": target_j, "produced": produced_j,
                            "position": pos, "process": ", ".join(procs),
                        })
                elif tph["kind"] == "중성":  # 모음: 대치만 집계(생략은 음절생략으로)
                    if pph is not None and pph["kind"] == "중성":
                        vowel_total[target_j] += 1
                        if pph["jamo"] == target_j:
                            vowel_correct[target_j] += 1
                        else:
                            vowel_confusion[target_j][pph["jamo"]] += 1
                            process_counter["모음대치"] += 1
                            vowel_errors.append({
                                "word": t_word, "target_pron": t_pron,
                                "produced_word": p_word or "",
                                "target": target_j, "produced": pph["jamo"],
                            })
                    elif pph is None:
                        # 음절핵 미산출 = 음절 생략(추정)
                        syllable_omissions += 1
                        process_counter["음절생략"] += 1

    total = sum(phoneme_total.values())
    correct = sum(phoneme_correct.values())
    pcc = round(correct / total * 100, 1) if total else 0.0
    phoneme_accuracy = {
        ph: round(phoneme_correct[ph] / phoneme_total[ph] * 100, 1)
        for ph in sorted(phoneme_total, key=lambda p: -phoneme_total[p])
    }
    vtotal = sum(vowel_total.values())
    vcorrect = sum(vowel_correct.values())
    pvc = round(vcorrect / vtotal * 100, 1) if vtotal else 0.0
    vowel_accuracy = {
        ph: round(vowel_correct[ph] / vowel_total[ph] * 100, 1)
        for ph in sorted(vowel_total, key=lambda p: -vowel_total[p])
    }
    phonological_processes = [
        {"process": p, "count": c, "type": "비전형" if is_atypical(p) else "발달적"}
        for p, c in process_counter.most_common()
    ]

    return {
        "confusion_matrix": {t: dict(c) for t, c in confusion.items()},
        "position_errors": {p: position_errors.get(p, 0) for p in POSITION_ORDER},
        "position_total": {p: position_total.get(p, 0) for p in POSITION_ORDER},
        "pcc": pcc,
        "phoneme_accuracy": phoneme_accuracy,
        "errors": errors,
        "pvc": pvc,
        "vowel_accuracy": vowel_accuracy,
        "vowel_confusion_matrix": {t: dict(c) for t, c in vowel_confusion.items()},
        "vowel_errors": vowel_errors,
        "additions_detail": additions_detail,
        "phonological_processes": phonological_processes,
        "summary": {
            "total_consonants": total,
            "correct_consonants": correct,
            "error_count": len(errors),
            "additions": additions,
            "total_vowels": vtotal,
            "correct_vowels": vcorrect,
            "vowel_error_count": len(vowel_errors),
            "syllable_omissions": syllable_omissions,
        },
    }
