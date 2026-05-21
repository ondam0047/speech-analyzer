"""📝 언어 분석 — MLU/TTR/NDW.

전사 결과 불러오기 / 텍스트 직접 입력 / 음성 업로드(자동 전사 + 화자 지정 → 아동 발화만).
낱말 = 체언+용언+수식언+독립언. 의미/문법 영역 분석.
"""

import os
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.insights import generate_insight, summarize_language  # noqa: E402
from modules.intelligibility import compute_intelligibility  # noqa: E402
from modules.norms import LANGUAGE_REFERENCES, language_reference_text  # noqa: E402
from modules.shared_ui import (  # noqa: E402
    SHARED_TRANSCRIPT,
    _read_manual_table,
    api_key_input,
    child_utterances_from,
    get_analyzer,
    render_language_results,
    report_download_button,
    require_password,
    voice_target_review,
)
from modules.transcription import TranscriptionError  # noqa: E402

st.set_page_config(page_title="언어 분석", page_icon="📝", layout="wide")
require_password()
api_key = api_key_input()

SAMPLE_UTTERANCES = """엄마랑 아빠랑 같이 큰집에 갔어요
토끼가 추운데 죽어서 너무 슬펐어요
시골이라서 연을 날릴 수 있어요
국이 너무 뜨거워서 못 먹었어요
그러면 우리 같이 영화 보러 가요"""

st.title("📝 언어 분석")
st.caption("MLU-w · MLU-m · TTR · NDW · TNW  ·  낱말 = 체언 + 용언 + 수식언 + 독립언")

input_mode = st.radio(
    "입력 방식", ["전사 결과 불러오기", "텍스트 직접 입력", "음성 업로드"], horizontal=True)

utterances: list[str] | None = None

if input_mode == "전사 결과 불러오기":
    shared = st.session_state.get(SHARED_TRANSCRIPT) or []
    if "lang_load_area" not in st.session_state:
        st.session_state["lang_load_area"] = "\n".join(shared)
    st.markdown("**전사 페이지에서 저장한 아동 발화를 불러오거나, 전사 엑셀/CSV를 업로드**하세요.")
    if st.button(f"📋 전사 페이지 결과 불러오기 ({len(shared)}개)", disabled=not shared):
        st.session_state["lang_load_area"] = "\n".join(shared)
        st.rerun()
    up = st.file_uploader("또는 전사 엑셀/CSV 업로드 (.xlsx, .csv)", type=["xlsx", "csv"], key="lang_imp")
    if up is not None:
        sig = (up.name, getattr(up, "size", None))
        if st.session_state.get("lang_imp_sig") != sig:
            st.session_state["lang_imp_sig"] = sig
            try:
                rows = _read_manual_table(up)
                loaded = [r["목표어"] for r in rows if r["목표어"]]
                if loaded:
                    st.session_state["lang_load_area"] = "\n".join(loaded)
                    st.rerun()
                else:
                    st.warning("불러올 목표어가 없습니다.")
            except Exception as e:
                st.error(f"파일을 읽지 못했습니다: {e}")
    text = st.text_area("아동 발화 (한 줄 = 한 발화, 수정 가능)", key="lang_load_area", height=240)
    if not shared and not st.session_state.get("lang_load_area"):
        st.info("전사 페이지에서 ‘분석에 사용’으로 저장하거나, 위에 엑셀/CSV를 업로드하세요.")
    if st.button("분석 실행", type="primary", key="lang_load_run"):
        utterances = [line for line in text.splitlines() if line.strip()] or None
        if utterances is None:
            st.warning("발화가 없습니다.")

elif input_mode == "텍스트 직접 입력":
    st.markdown("**발화를 한 줄에 하나씩 입력하세요.** (한 줄 = 한 발화)")
    st.caption("정확한 지표를 위해 표준어로 정규화하고 반복·수정·간투사(마디)는 제외한 전사를 권장합니다.")
    if st.button("예시 발화 불러오기"):
        st.session_state["lang_text"] = SAMPLE_UTTERANCES
    text = st.text_area(
        "발화 입력", key="lang_text", height=220,
        placeholder="예)\n엄마랑 아빠랑 같이 큰집에 갔어요\n그러면 우리 같이 영화 보러 가요")
    if st.button("분석 실행", type="primary"):
        utterances = [line for line in text.splitlines() if line.strip()]
        if not utterances:
            st.warning("발화를 한 줄 이상 입력하세요.")
            utterances = None

else:  # 음성 업로드 (언어 분석은 목표어만 필요)
    st.markdown("**음성을 업로드하면 Whisper로 전사합니다.** 화자(아동/치료사/제외)를 지정하면 **아동 발화만** 분석합니다.")
    st.caption("OpenAI API 키 필요 · Whisper 25MB 제한.")
    edited = voice_target_review("lang", api_key)
    if edited is not None and st.button("분석 실행", type="primary"):
        utterances = child_utterances_from(edited) or None
        if utterances is None:
            st.warning("아동 발화로 지정된 항목이 없습니다.")

if utterances:
    st.session_state["lang_result"] = get_analyzer().analyze(utterances)
    st.session_state["intelligibility"] = compute_intelligibility(utterances)
    st.session_state.pop("lang_insight", None)

result = st.session_state.get("lang_result")
if result:
    render_language_results(result)

    st.divider()
    st.subheader("🗣️ 말명료도 (이해가능도)")
    intel = st.session_state.get("intelligibility") or compute_intelligibility(
        [u["text"] for u in result["utterances"]])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("어절 명료도", f"{intel['word_intelligibility']}%")
    m2.metric("음절 명료도", f"{intel['syllable_intelligibility']}%")
    m3.metric("이해 어절 / 전체", f"{intel['intelligible_words']} / {intel['total_words']}")
    m4.metric("불명료 어절", intel["unintelligible_words"])
    st.caption("못 알아들은 부분을 음절 수만큼 ‘*’로 표기하면 명료도가 계산됩니다(예: 3음절 → ***). "
               "어절 명료도 = ‘*’ 없는 어절 / 전체 어절 × 100. 전사 페이지에서도 확인할 수 있습니다.")

    st.divider()
    report_download_button(language=result, key="lang")

    st.divider()
    st.subheader("임상 코멘트")
    st.text(summarize_language(result))
    patient = st.session_state.get("patient_info") or {}
    age_months = patient.get("age_months")
    if age_months:
        st.caption(f"LLM 코멘트는 생활연령 **{patient.get('age', '')}** 기준 언어 발달 규준으로 "
                   "연령 대비 적절성(MLU 등)을 함께 해석합니다.")
    else:
        st.warning("⚠️ 생활연령이 계산되지 않아 LLM 임상 코멘트를 생성할 수 없습니다. "
                   "사이드바에 **생년월일·검사일**을 입력하세요.")
    with st.expander("📚 연령 기반 언어 발달 규준 (코멘트 근거) 보기"):
        st.text(language_reference_text(age_months))
        st.markdown("**참고 문헌**")
        for r in LANGUAGE_REFERENCES:
            st.caption(f"- {r}")
    if st.button("🧠 LLM 임상 코멘트 생성", disabled=not age_months, key="lang_llm"):
        try:
            with st.spinner("코멘트 생성 중…"):
                st.session_state["lang_insight"] = generate_insight(
                    language=result, api_key=api_key, age_months=age_months)
        except TranscriptionError as e:
            st.error(str(e))
    if age_months and st.session_state.get("lang_insight"):
        st.markdown(st.session_state["lang_insight"])

st.divider()
st.page_link("app.py", label="← 홈으로", icon="🏠")
