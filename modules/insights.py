"""분석 결과 → 사실 요약 텍스트(분류 없는 수치 요약).

LLM 임상 코멘트 기능은 비용 대비 무료 규준 비교 보고서와 중복되어 제거됨.
여기서는 화면/보고서용 텍스트 요약만 제공한다.
"""

from __future__ import annotations


def summarize_articulation(articulation: dict) -> str:
    """분류 없는 사실 요약 텍스트."""
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
