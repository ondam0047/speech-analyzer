"""인사이트 레이어 (M5) — 조음/언어 분석 → LLM 임상 코멘트.

APAC 음운변동 분류를 prompt로 제공해 패턴 해석을 요청한다.
룰 엔진으로 강제 분류하지 않으며, 사실 요약은 분류 없이 수치만 제시한다.
"""

from __future__ import annotations

import os

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


def _build_prompt(articulation: dict | None, language: dict | None) -> tuple[str, str]:
    system = (
        "당신은 아동 말·언어 평가를 돕는 언어재활(언어치료) 전문가입니다. "
        "주어진 자동 분석 수치를 임상적으로 해석하되, 단정하지 말고 경향으로 기술하고, "
        "자동 분석의 한계(전사 오류·정렬 잡음 가능)를 전제로 임상가 검수를 권고하세요. "
        "아래 APAC 음운변동 분류를 참고해 관찰된 패턴이 어떤 변동에 해당할 수 있는지 제시하세요.\n\n"
        f"[APAC 음운변동 분류]\n{APAC_TAXONOMY}"
    )
    parts = []
    if articulation is not None:
        parts.append("[조음 분석 요약]\n" + summarize_articulation(articulation))
    if language is not None:
        parts.append("[언어 분석 요약]\n" + summarize_language(language))
    user = (
        "다음 자동 분석 결과를 바탕으로, ① 두드러진 조음·음운 오류 패턴과 가능한 음운변동 해석, "
        "② 어휘·구문(언어) 특징, ③ 임상적 제언을 간결한 한국어 불릿으로 정리하세요. "
        "확실하지 않은 부분은 검수 필요로 표시하세요.\n\n"
        + "\n\n".join(parts)
    )
    return system, user


def generate_insight(
    articulation: dict | None = None,
    language: dict | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> str:
    """LLM 임상 코멘트 생성. API 키 미설정 시 TranscriptionError."""
    if articulation is None and language is None:
        raise TranscriptionError("분석 결과가 없습니다.")
    client = _get_client(api_key)
    model = model or os.getenv("INSIGHT_MODEL", "gpt-4o-mini")
    system, user = _build_prompt(articulation, language)
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
