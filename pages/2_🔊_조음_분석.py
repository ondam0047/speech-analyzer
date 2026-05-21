"""🔊 조음 분석 — 듀얼 전사(목표어/산출형) → 컨퓨전 매트릭스 · PCC · 위치별 오류 (M3)."""

import os
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.articulation import analyze_articulation  # noqa: E402
from modules.insights import generate_insight, summarize_articulation  # noqa: E402
from modules.intelligibility import compute_intelligibility  # noqa: E402
from modules.norms import REFERENCES, reference_text  # noqa: E402
from modules.shared_ui import (  # noqa: E402
    api_key_input,
    child_pairs,
    child_targets,
    manual_dual_entry,
    render_articulation_results,
    report_download_button,
    require_password,
    voice_dual_review,
)
from modules.transcription import TranscriptionError  # noqa: E402

st.set_page_config(page_title="조음 분석", page_icon="🔊", layout="wide")
require_password()
api_key = api_key_input()

st.title("🔊 조음 분석")
st.caption("목표어/산출형 듀얼 전사 → 임상가 검수 → PCC·PVC·컨퓨전 매트릭스·오류 음운변동")

mode = st.radio("입력 방식", ["직접 입력 (권장)", "음성 업로드 (자동 전사)"],
                horizontal=True, key="artic_mode")
if mode.startswith("직접"):
    edited = manual_dual_entry("artic")
else:
    edited = voice_dual_review("artic", api_key)

if edited is not None and st.button("📊 분석 실행", type="primary"):
    pairs = child_pairs(edited)
    if not pairs:
        st.warning("목표어·산출형이 모두 입력된 아동 발화가 없습니다.")
    else:
        st.session_state["artic_result"] = analyze_articulation(pairs)
        # 말명료도: 목표어 기준(목표어를 알아들었는지 = '*' 여부). 산출 오류가 있어도
        # 목표어를 알아들었으면 명료한 것이므로 목표어 열로 계산한다.
        st.session_state["intelligibility"] = compute_intelligibility(child_targets(edited))
        st.session_state.pop("artic_insight", None)

result = st.session_state.get("artic_result")
if result:
    st.divider()
    render_articulation_results(result)

    intel = st.session_state.get("intelligibility")
    if intel and intel.get("total_words"):
        st.divider()
        st.subheader("🗣️ 말명료도 (이해가능도)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("어절 명료도", f"{intel['word_intelligibility']}%")
        m2.metric("음절 명료도", f"{intel['syllable_intelligibility']}%")
        m3.metric("이해 어절 / 전체", f"{intel['intelligible_words']} / {intel['total_words']}")
        m4.metric("불명료 어절", intel["unintelligible_words"])
        st.caption("못 알아들은 목표어는 음절 수만큼 ‘*’로 표기하세요(예: 3음절 → ***). "
                   "어절 명료도 = ‘*’ 없는 어절 / 전체 어절 × 100. "
                   "산출에 오류가 있어도 목표어를 알아들었으면 명료한 것으로 봅니다(명료도 ≠ 정확도).")

    st.divider()
    report_download_button(articulation=result, key="artic")

    st.divider()
    st.subheader("임상 코멘트")
    st.text(summarize_articulation(result))

    patient = st.session_state.get("patient_info") or {}
    age_months = patient.get("age_months")
    if age_months:
        st.caption(f"LLM 코멘트는 생활연령 **{patient.get('age', '')}** 기준 발달 규준으로 "
                   "연령 대비 적절성을 함께 해석합니다.")
    else:
        st.warning("⚠️ 생활연령이 계산되지 않아 LLM 임상 코멘트를 생성할 수 없습니다. "
                   "사이드바에 **생년월일·검사일**을 입력하세요(연령 기반 발달 규준 해석에 필요).")
    with st.expander("📚 연령 기반 발달 규준 (코멘트 근거) 보기"):
        st.text(reference_text(age_months))
        st.markdown("**참고 문헌**")
        for r in REFERENCES:
            st.caption(f"- {r}")

    if st.button("🧠 LLM 임상 코멘트 생성", disabled=not age_months):
        try:
            with st.spinner("코멘트 생성 중…"):
                st.session_state["artic_insight"] = generate_insight(
                    articulation=result, api_key=api_key, age_months=age_months)
        except TranscriptionError as e:
            st.error(str(e))
    if age_months and st.session_state.get("artic_insight"):
        st.markdown(st.session_state["artic_insight"])

st.divider()
st.page_link("app.py", label="← 홈으로", icon="🏠")
