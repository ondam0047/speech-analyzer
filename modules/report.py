"""분석 결과 → 인쇄용 HTML 보고서.

브라우저에서 열어 인쇄(Ctrl+P) → 'PDF로 저장'하면 PDF가 된다. 한글 폰트 의존 없음.
"""

from __future__ import annotations

import html
from datetime import date

from modules.articulation import POSITION_ORDER
from modules.morpheme import BROAD_OF, GRAM_ORDER, SEMANTIC_ORDER, SENTENCE_TYPES
from modules.norms import (
    ATYPICAL_NOTE,
    CONSONANT_STAGE_TABLE,
    LANGUAGE_REFERENCES,
    REFERENCES,
    VOWEL_NOTE,
    developmental_reference,
    language_reference,
)

_CSS = """
body{font-family:'Malgun Gothic','맑은 고딕','Apple SD Gothic Neo',sans-serif;color:#1a1a1a;
 max-width:920px;margin:24px auto;padding:0 16px;line-height:1.55}
h1{font-size:22px;border-bottom:3px solid #4C78A8;padding-bottom:6px;margin-bottom:4px}
h2{font-size:17px;margin-top:26px;color:#2a4d69;border-left:5px solid #4C78A8;padding-left:8px}
h3{font-size:14px;margin:16px 0 4px;color:#34495e}
table{border-collapse:collapse;width:100%;margin:8px 0;font-size:13px}
th,td{border:1px solid #cfd8e3;padding:5px 8px;text-align:center}
th{background:#eef3f8}
td.l,th.l{text-align:left}
.metrics{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}
.metric{flex:1;min-width:110px;background:#f5f8fc;border:1px solid #dbe6f0;border-radius:8px;padding:8px 10px;text-align:center}
.metric .v{font-size:20px;font-weight:700;color:#2a4d69}
.metric .l{font-size:12px;color:#555}
.muted{color:#777;font-size:12px;margin:4px 0}
.atyp{color:#c0392b;font-weight:700}
.tag{display:inline-block;background:#eef3f8;border:1px solid #dbe6f0;border-radius:10px;padding:1px 8px;margin:2px;font-size:12px}
.insight{white-space:pre-wrap;background:#f7faf7;border:1px solid #d6e6d6;border-radius:8px;padding:10px 12px;font-size:13px;line-height:1.6}
@media print{body{margin:0}h2{break-after:avoid}}
"""


def _esc(x) -> str:
    return html.escape(str(x))


def _metric(label, value) -> str:
    return f'<div class="metric"><div class="v">{_esc(value)}</div><div class="l">{_esc(label)}</div></div>'


def _table(headers, rows, left_cols=()) -> str:
    th = "".join(
        f'<th class="l">{_esc(h)}</th>' if i in left_cols else f"<th>{_esc(h)}</th>"
        for i, h in enumerate(headers))
    body = ""
    for r in rows:
        tds = "".join(
            f'<td class="l">{_esc(c)}</td>' if i in left_cols else f"<td>{_esc(c)}</td>"
            for i, c in enumerate(r))
        body += f"<tr>{tds}</tr>"
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def _language_section(result: dict) -> str:
    s = result["stats"]
    out = ["<h2>📝 언어 분석</h2>"]
    out.append('<div class="metrics">' + "".join([
        _metric("발화 수", s["utterance_count"]),
        _metric("MLU-w", s["mlu_w"]),
        _metric("MLU-m", s["mlu_m"]),
        _metric("TTR", s["ttr"]),
        _metric("TNW(총 낱말)", s["tnw"]),
        _metric("NDW(다른 낱말)", s["ndw"]),
    ]) + "</div>")
    out.append('<p class="muted">낱말 = 체언+용언+수식언+독립언(용언은 기본형). '
               "MLU-w=총 낱말/발화 · MLU-m=총 형태소/발화 · TTR=NDW/TNW.</p>")

    out.append("<h3>의미 영역 (품사 세분화)</h3>")
    out.append(_table(
        ["품사", "대분류", "총 낱말", "서로 다른 낱말"],
        [[c, BROAD_OF[c], s["semantic_counts"][c], s["semantic_ndw"][c]] for c in SEMANTIC_ORDER],
        left_cols=(0, 1)))

    out.append("<h3>문법형태소 (범주)</h3>")
    out.append(_table(["범주", "빈도"], [[c, s["gram_categories"][c]] for c in GRAM_ORDER],
                      left_cols=(0,)))

    gmf = s.get("gram_morpheme_freq", [])
    if gmf:
        order = {c: i for i, c in enumerate(GRAM_ORDER)}
        gmf_sorted = sorted(gmf, key=lambda g: (order.get(g["category"], 99), -g["count"]))
        conn = [g for g in gmf_sorted if g["category"] == "연결어미"]
        if conn:
            out.append("<h3>연결어미 사용형</h3>")
            out.append("".join(f'<span class="tag">{_esc(g["morpheme"])} ({g["count"]})</span>'
                               for g in conn))
        out.append("<h3>조사·어미 사용 빈도 (형태소별)</h3>")
        out.append(_table(["범주", "형태소", "빈도"],
                          [[g["category"], g["morpheme"], g["count"]] for g in gmf_sorted],
                          left_cols=(0, 1)))

    out.append("<h3>문장유형 (자동 추정)</h3>")
    out.append(_table(["문장유형", "발화 수"], [[t, s["sentence_types"][t]] for t in SENTENCE_TYPES],
                      left_cols=(0,)))
    out.append('<p class="muted">연결어미·전성어미 기반 자동 추정 — 임상가 검수 필요.</p>')

    out.append("<h3>발화별 상세</h3>")
    rows = [[i + 1, u["text"], u["sentence_type"], u["words"], u["morphemes"]]
            for i, u in enumerate(result["utterances"])]
    out.append(_table(["#", "발화", "문장유형", "낱말", "형태소"], rows, left_cols=(1,)))
    return "\n".join(out)


def _articulation_section(result: dict) -> str:
    s = result["summary"]
    out = ["<h2>🔊 조음 분석</h2>"]
    out.append('<div class="metrics">' + "".join([
        _metric("PCC(자음)", f'{result["pcc"]}%'),
        _metric("PVC(모음)", f'{result.get("pvc", 0.0)}%'),
        _metric("목표 자음", s["total_consonants"]),
        _metric("자음 오류", s["error_count"]),
        _metric("첨가", s["additions"]),
        _metric("음절생략", s.get("syllable_omissions", 0)),
    ]) + "</div>")
    out.append('<p class="muted">PCC=정확 자음/목표 자음×100(초성 ㅇ 제외) · PVC=정확 모음/목표 모음×100. '
               "목표어는 g2p 발음형으로 변환 후 비교(연음·경음화 등 자연 변동은 오류 아님).</p>")

    pp = result.get("phonological_processes") or []
    if pp:
        out.append("<h3>오류 음운변동 패턴 (상대분석)</h3>")
        rows = []
        for x in pp:
            proc = x["process"]
            if x["type"] == "비전형":
                proc = f'<span class="atyp">{_esc(proc)} ⚠</span>'
            else:
                proc = _esc(proc)
            rows.append((proc, x["type"], x["count"]))
        # 직접 HTML(아이콘 포함) — _table은 escape하므로 수동 구성
        body = "".join(f"<tr><td class='l'>{p}</td><td>{_esc(t)}</td><td>{_esc(c)}</td></tr>"
                       for p, t, c in rows)
        out.append("<table><thead><tr><th class='l'>음운변동</th><th>유형</th><th>빈도</th>"
                   f"</tr></thead><tbody>{body}</tbody></table>")
        atyp = [x["process"] for x in pp if x["type"] == "비전형"]
        if atyp:
            out.append(f'<p class="atyp">⚠ 비전형(비발달적) 패턴: {_esc(", ".join(atyp))} — 임상가 검수 권장.</p>')

    pe, pt = result["position_errors"], result["position_total"]
    out.append("<h3>위치별 오류</h3>")
    out.append(_table(
        ["위치", "오류", "전체", "오류율(%)"],
        [[p, pe[p], pt[p], round(pe[p] / pt[p] * 100, 1) if pt[p] else 0.0] for p in POSITION_ORDER],
        left_cols=(0,)))

    pa = result["phoneme_accuracy"]
    if pa:
        out.append("<h3>자음 음소별 정확도</h3>")
        out.append(_table(["음소", "정확도(%)"], [[k, v] for k, v in pa.items()], left_cols=(0,)))
    va = result.get("vowel_accuracy") or {}
    if va:
        out.append("<h3>모음 정확도</h3>")
        out.append(_table(["모음", "정확도(%)"], [[k, v] for k, v in va.items()], left_cols=(0,)))

    if result["errors"]:
        out.append("<h3>자음 오류 상세</h3>")
        out.append(_table(
            ["목표어절", "목표발음", "산출어절", "목표(음소)", "산출(음소)", "위치", "음운변동"],
            [[e.get("word", ""), e.get("target_pron", ""), e.get("produced_word", ""),
              e["target"], e["produced"], e["position"], e.get("process", "")]
             for e in result["errors"]], left_cols=(0, 1, 2, 6)))
    return "\n".join(out)


def _intelligibility_section(intel: dict) -> str:
    out = ["<h2>🗣️ 말명료도 (이해가능도)</h2>"]
    out.append('<div class="metrics">' + "".join([
        _metric("어절 명료도", f'{intel["word_intelligibility"]}%'),
        _metric("음절 명료도", f'{intel["syllable_intelligibility"]}%'),
        _metric("전체 어절", intel["total_words"]),
        _metric("이해 어절", intel["intelligible_words"]),
        _metric("불명료 어절", intel["unintelligible_words"]),
    ]) + "</div>")
    out.append('<p class="muted">못 알아들은 부분은 음절 수만큼 ‘*’로 표기. '
               "어절 명료도 = ‘*’ 없는 어절/전체 어절×100 · "
               "음절 명료도 = (전체 음절−불명료 음절)/전체 음절×100.</p>")
    return "\n".join(out)


def _norms_section(age_months: int | None) -> str:
    out = ["<h2>📚 발달 규준 (해석 근거)</h2>"]
    out.append("<h3>자음 완전습득(95~100%) 기대 연령 — 김영태(1996)</h3>")
    out.append(_table(["연령", "완전습득 자음"], [[b, c] for b, c in CONSONANT_STAGE_TABLE],
                      left_cols=(0, 1)))
    ref = developmental_reference(age_months)
    if ref:
        y, m = divmod(ref["age_months"], 12)
        out.append(f"<h3>생활연령 {y}세 {m}개월 기준 비교</h3>")
        out.append(_table(
            ["항목", "내용"],
            [["완전습득 기대 자음 (오류 시 주목)", ", ".join(ref["mastered_expected"]) or "없음"],
             ["아직 발달 중 (연령상 정상 가능)", ", ".join(ref["developing"]) or "없음"],
             ["사라졌어야 할 발달적 음운변동",
              ", ".join(ref["processes_should_resolve"]) or "해당 없음"],
             ["연령 기대 자음정확도(PCC)", ref["expected_pcc"]],
             ["연령 기대 말명료도", ref["expected_intelligibility"]]],
            left_cols=(0, 1)))
        out.append('<p class="muted">관찰된 오류가 ‘연령상 정상 범위’인지 ‘연령 대비 지연/주목 '
                   "대상’인지의 해석 근거입니다. 최종 판정은 표준화 검사(U-TAP2/APAC) 백분위로 "
                   "확인해야 합니다.</p>")
    else:
        out.append('<p class="muted">생활연령 미입력 — 연령 대비 비교는 제한적입니다.</p>')
    out.append(f'<p class="muted"><b>모음</b> {_esc(VOWEL_NOTE)}</p>')
    out.append(f'<p class="muted"><b>비전형 패턴</b> {_esc(ATYPICAL_NOTE)}</p>')
    out.append("<h3>참고 문헌</h3>")
    out.append('<p class="muted">' + "<br>".join(_esc(r) for r in REFERENCES) + "</p>")
    return "\n".join(out)


def _language_norms_section(age_months: int | None) -> str:
    out = ["<h2>📚 언어 발달 규준 (해석 근거)</h2>"]
    ref = language_reference(age_months)
    if ref:
        y, m = divmod(ref["age_months"], 12)
        lo_m, hi_m = ref["mlu_m_range"]
        lo_w, hi_w = ref["mlu_w_range"]
        out.append(f"<h3>생활연령 {y}세 {m}개월 참고 범위(근사)</h3>")
        out.append(_table(
            ["항목", "내용"],
            [["MLU-m(형태소) 참고 범위", f"{lo_m}~{hi_m}"],
             ["MLU-w(낱말) 참고 범위", f"{lo_w}~{hi_w}"],
             ["문법/구문 기대", ref["grammar_note"]]],
            left_cols=(0, 1)))
        out.append('<p class="muted">MLU 참고 범위는 표본·전사 기준에 민감한 근사치입니다. '
                   "TTR은 표본 크기에 민감해 연령 절대비교에 부적합합니다. "
                   "최종 판정은 표준화 검사로 확인해야 합니다.</p>")
    else:
        out.append('<p class="muted">생활연령 미입력 — 연령 대비 비교는 제한적입니다.</p>')
    out.append("<h3>참고 문헌</h3>")
    out.append('<p class="muted">' + "<br>".join(_esc(r) for r in LANGUAGE_REFERENCES) + "</p>")
    return "\n".join(out)


def _insight_section(insight: str) -> str:
    body = _esc(insight)
    return ('<h2>🧠 LLM 임상 코멘트 (참고)</h2>'
            f'<div class="insight">{body}</div>'
            '<p class="muted">AI가 생성한 해석으로 임상가 검수가 필요합니다. '
            "최종 판정은 표준화 검사 결과로 확인하세요.</p>")


def _patient_section(patient: dict) -> str:
    items = []
    if patient.get("name"):
        items.append(("대상자", patient["name"]))
    if patient.get("birth"):
        items.append(("생년월일", patient["birth"]))
    if patient.get("test"):
        items.append(("검사일", patient["test"]))
    if patient.get("age"):
        items.append(("생활연령", patient["age"]))
    if not items:
        return ""
    return _table(["항목", "내용"], items, left_cols=(0, 1))


def build_report_html(language: dict | None = None, articulation: dict | None = None,
                      patient: dict | None = None, intelligibility: dict | None = None,
                      insight: str | None = None,
                      title: str = "자발화 분석 보고서") -> str:
    age_months = (patient or {}).get("age_months")
    sections = []
    if patient:
        ps = _patient_section(patient)
        if ps:
            sections.append("<h2>대상자 정보</h2>" + ps)
    if intelligibility and intelligibility.get("total_words"):
        sections.append(_intelligibility_section(intelligibility))
    if language is not None:
        sections.append(_language_section(language))
        sections.append(_language_norms_section(age_months))
    if articulation is not None:
        sections.append(_articulation_section(articulation))
        sections.append(_norms_section(age_months))
    if insight:
        sections.append(_insight_section(insight))
    if not sections:
        sections.append("<p>분석 결과가 없습니다.</p>")
    body = "\n".join(sections)
    return (
        "<!doctype html><html lang='ko'><head><meta charset='utf-8'>"
        f"<title>{_esc(title)}</title><style>{_CSS}</style></head><body>"
        f"<h1>{_esc(title)}</h1>"
        f"<p class='muted'>생성일: {date.today().isoformat()} · 자동 분석 결과이므로 임상가 검수가 필요합니다.</p>"
        f"{body}</body></html>"
    )
