"""인사이트 레이어 (M5) — 조음/언어 분석 → LLM 임상 코멘트.

APAC 음운변동 분류를 prompt로 제공해 패턴 해석을 요청한다.
룰 엔진으로 강제 분류하지 않으며, 사실 요약은 분류 없이 수치만 제시한다.
"""

from __future__ import annotations

import os

from modules.norms import (
    developmental_reference,
    language_reference,
    language_reference_text,
    reference_text,
)
from modules.transcription import TranscriptionError, _get_client

APAC_TAXONOMY = (
    "조음방법 변동: 파열음화, 마찰음화, 파찰음화, 비음화, 탈비음화, 유음화, 유음의 단순화\n"
    "조음위치 변동: 전방화(연구개음의 전방화 포함), 후방화, 성문음화\n"
    "발성유형 변동: 긴장음화, 기식음화, 이완음화\n"
    "음절구조 변동: 어두초성생략, 어중초성생략, 어말종성생략, 어중종성생략, 음절생략, 첨가, 도치\n"
    "발달적: 파열음화·마찰음화·연구개음의 전방화·긴장음화·유음의 단순화·종성생략 등\n"
    "비전형(비발달적): 어두초성생략·후방화·첨가·도치·탈비음화·성문음화·음절생략 등\n"
    "(경음화·비음화·유음화 등 의무적 음운변동은 정상 산출이므로 오류로 보지 않음)"
)


def summarize_articulation(articulation: dict) -> str:
    """분류 없는 사실 요약 텍스트 (프롬프트/오프라인 표시용)."""
    s = articulation["summary"]
    lines = [
        f"PCC(자음정확도): {articulation['pcc']}% "
        f"(정확 {s['correct_consonants']}/{s['total_consonants']}, 오류 {s['error_count']}, 첨가 {s['additions']})",
        f"위치별 오류: {articulation['position_errors']} / 전체 {articulation['position_total']}",
    ]
    if s.get("total_vowels"):
        lines.append(
            f"PVC(모음정확도): {articulation.get('pvc', 0.0)}% "
            f"(정확 {s.get('correct_vowels', 0)}/{s['total_vowels']}, 오류 {s.get('vowel_error_count', 0)})")
    pairs = []
    for tgt, row in articulation["confusion_matrix"].items():
        for prod, cnt in row.items():
            pairs.append((cnt, f"{tgt}→{prod}"))
    pairs.sort(reverse=True)
    if pairs:
        lines.append("주요 자음 오류 패턴(목표→산출, 빈도순): "
                     + ", ".join(f"{p} ({c})" for c, p in pairs[:12]))
    low = [(acc, ph) for ph, acc in articulation["phoneme_accuracy"].items() if acc < 100]
    low.sort()
    if low:
        lines.append("정확도 낮은 자음: " + ", ".join(f"{ph} {acc}%" for acc, ph in low[:10]))
    vpairs = []
    for tgt, row in (articulation.get("vowel_confusion_matrix") or {}).items():
        for prod, cnt in row.items():
            vpairs.append((cnt, f"{tgt}→{prod}"))
    vpairs.sort(reverse=True)
    if vpairs:
        lines.append("주요 모음 오류 패턴(목표→산출, 빈도순): "
                     + ", ".join(f"{p} ({c})" for c, p in vpairs[:8]))
    pp = articulation.get("phonological_processes") or []
    if pp:
        dev = [f"{x['process']}({x['count']})" for x in pp if x["type"] == "발달적"]
        atyp = [f"{x['process']}({x['count']})" for x in pp if x["type"] == "비전형"]
        lines.append("오류 음운변동(자동분류, 의무적 변동 제외): "
                     + ", ".join(f"{x['process']}({x['count']})" for x in pp[:14]))
        if atyp:
            lines.append("그중 비전형(비발달적) 패턴: " + ", ".join(atyp))
    if s.get("syllable_omissions"):
        lines.append(f"음절 생략(추정): {s['syllable_omissions']}회")
    return "\n".join(lines)


def summarize_language(language: dict) -> str:
    st = language["stats"]
    lines = [
        f"발화 수 {st['utterance_count']}, MLU-w {st['mlu_w']}, MLU-m {st['mlu_m']}, "
        f"TTR {st['ttr']}, NDW {st['ndw']}, TNW {st['tnw']}",
        f"품사: {st['semantic_counts']}",
        f"문법형태소(범주): {st.get('gram_categories', {})}",
    ]
    gmf = st.get("gram_morpheme_freq", [])
    by_cat: dict[str, list[str]] = {}
    for g in gmf:
        by_cat.setdefault(g["category"], []).append(f"{g['morpheme']}({g['count']})")
    for cat in ("연결어미", "어말어미", "전성어미", "선어말어미", "조사"):
        if by_cat.get(cat):
            lines.append(f"{cat} 사용형: " + ", ".join(by_cat[cat][:12]))
    lines.append(f"문장유형(추정): {st['sentence_types']}")
    return "\n".join(lines)


def norm_comparison(articulation: dict, age_months: int | None) -> str:
    """대상자의 실제 오류를 생활연령 규준과 직접 대조한 결과(프롬프트·표시용).

    어떤 자음 오류가 '연령상 완전습득 기대인데 오류(지연/주목)'인지,
    '아직 발달 중이라 정상 범위'인지, 어떤 음운변동이 '이 연령엔 사라졌어야 하는데
    관찰됨'인지를 명시해 LLM이 규준 대비 현행수준을 비교 기술하도록 한다.
    """
    ref = developmental_reference(age_months)
    if ref is None:
        return ""
    mastered = set(ref["mastered_expected"])
    developing = set(ref["developing"])
    should_resolve = set(ref["processes_should_resolve"])
    acc = articulation.get("phoneme_accuracy", {})
    err_phonemes = [ph for ph, a in acc.items() if a < 100]
    on_mastered = [ph for ph in err_phonemes if ph in mastered]
    on_developing = [ph for ph in err_phonemes if ph in developing]
    pp = articulation.get("phonological_processes") or []
    proc_late = [x["process"] for x in pp if x["process"] in should_resolve]
    atypical = [x["process"] for x in pp if x["type"] == "비전형"]
    vowel_err = bool(articulation.get("vowel_errors"))

    y, m = divmod(ref["age_months"], 12)
    lines = [
        f"[생활연령 규준 대비 현행수준 비교 — 이 비교를 코멘트에 반드시 반영]",
        f"  · 생활연령 {y}세 {m}개월 / 대상자 PCC {articulation.get('pcc')}% "
        f"(연령 기대 {ref['expected_pcc']})",
    ]
    if on_mastered:
        lines.append("  · ⚠ 이 연령엔 완전습득이 기대되나 오류가 관찰된 자음(연령 대비 지연/주목): "
                     + ", ".join(on_mastered))
    if on_developing:
        lines.append("  · 아직 발달 중이라 오류가 연령상 정상 범위일 수 있는 자음: "
                     + ", ".join(on_developing))
    if not err_phonemes:
        lines.append("  · 자음 오류 없음 — 연령 기대 수준 충족")
    if proc_late:
        lines.append("  · ⚠ 이 연령엔 대부분 사라졌어야 할 발달적 음운변동이 관찰됨: "
                     + ", ".join(sorted(set(proc_late))))
    if atypical:
        lines.append("  · ⚠ 비전형(비발달적) 패턴(연령 무관 주목): " + ", ".join(sorted(set(atypical))))
    if vowel_err:
        lines.append("  · 모음 오류 있음 — 단모음은 만 3세 이전 안정되므로 연령 대비 검토 필요")
    return "\n".join(lines)


def language_norm_comparison(language: dict, age_months: int | None) -> str:
    """대상자의 언어 수치(MLU 등)를 생활연령 규준과 대조한 결과(프롬프트·표시용)."""
    ref = language_reference(age_months)
    if ref is None:
        return ""
    s = language["stats"]
    lo_m, hi_m = ref["mlu_m_range"]
    lo_w, hi_w = ref["mlu_w_range"]

    def judge(v: float, lo: float, hi: float) -> str:
        if v < lo:
            return "연령 기대보다 낮음(주목)"
        if v > hi:
            return "연령 기대 이상"
        return "연령 기대 범위"

    y, m = divmod(ref["age_months"], 12)
    return "\n".join([
        "[생활연령 규준 대비 현행수준 비교 — 언어, 이 비교를 코멘트에 반드시 반영]",
        f"  · 생활연령 {y}세 {m}개월",
        f"  · MLU-m {s['mlu_m']} (연령 참고 {lo_m}~{hi_m}): {judge(s['mlu_m'], lo_m, hi_m)}",
        f"  · MLU-w {s['mlu_w']} (연령 참고 {lo_w}~{hi_w}): {judge(s['mlu_w'], lo_w, hi_w)}",
        f"  · NDW {s['ndw']}, TTR {s['ttr']} (TTR은 표본 크기에 민감 — 절대비교 주의)",
        f"  · 문법/구문 기대: {ref['grammar_note']}",
        "  ※ MLU 참고 범위는 근사치 — 최종 판정은 표준화 검사로 확인.",
    ])


def _build_prompt(articulation: dict | None, language: dict | None,
                  age_months: int | None = None) -> tuple[str, str]:
    norm_blocks = []
    if articulation is not None:
        norm_blocks.append(f"[APAC 음운변동 분류]\n{APAC_TAXONOMY}")
        norm_blocks.append(f"[조음 발달 규준 — 연령 기반 근거]\n{reference_text(age_months)}")
    if language is not None:
        norm_blocks.append(f"[언어 발달 규준 — 연령 기반 근거]\n{language_reference_text(age_months)}")
    system = (
        "당신은 아동 말·언어 평가를 돕는 언어재활(언어치료) 전문가입니다. "
        "주어진 자동 분석 수치를 임상적으로 해석하되, 단정하지 말고 경향으로 기술하고, "
        "자동 분석의 한계(전사 오류·정렬 잡음 가능)를 전제로 임상가 검수를 권고하세요. "
        "아래 발달 규준을 참고해, 관찰된 패턴이 대상자의 생활연령에 비추어 "
        "'연령상 정상 범위'인지 '연령 대비 지연/주목 대상'인지 반드시 근거와 함께 기술하세요.\n\n"
        + "\n\n".join(norm_blocks)
    )
    parts = []
    if articulation is not None:
        parts.append("[조음 분석 요약]\n" + summarize_articulation(articulation))
        cmp_text = norm_comparison(articulation, age_months)
        if cmp_text:
            parts.append(cmp_text)
    if language is not None:
        parts.append("[언어 분석 요약]\n" + summarize_language(language))
        lcmp = language_norm_comparison(language, age_months)
        if lcmp:
            parts.append(lcmp)
    user = (
        "다음 자동 분석 결과를 바탕으로 한국어 불릿으로 정리하세요.\n"
        "① 두드러진 특징(조음 분석이면 오류·음운변동, 언어 분석이면 어휘·구문) 해석\n"
        "② **생활연령 규준 대비 현행수준 비교** — 위 '규준 대비 현행수준 비교' 정보를 활용해, "
        "어떤 지표(자음/음운변동/MLU 등)가 연령 기대 대비 '지연/주목 대상'이고 어떤 것이 "
        "'연령상 정상 범위'인지 구체적 음소·지표명과 함께 반드시 기술(이 항목은 생략 금지)\n"
        "③ 임상적 제언\n"
        "확실하지 않은 부분은 검수 필요로 표시하고, 최종 판정은 표준화 검사 백분위로 "
        "확인하도록 권고하세요.\n\n"
        + "\n\n".join(parts)
    )
    return system, user


def generate_insight(
    articulation: dict | None = None,
    language: dict | None = None,
    model: str | None = None,
    api_key: str | None = None,
    age_months: int | None = None,
) -> str:
    """LLM 임상 코멘트 생성. API 키 미설정 시 TranscriptionError."""
    if articulation is None and language is None:
        raise TranscriptionError("분석 결과가 없습니다.")
    client = _get_client(api_key)
    model = model or os.getenv("INSIGHT_MODEL", "gpt-4o-mini")
    system, user = _build_prompt(articulation, language, age_months)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.3,
        )
    except Exception as e:
        raise TranscriptionError(f"인사이트 생성 실패: {e}") from e
    return (resp.choices[0].message.content or "").strip()
