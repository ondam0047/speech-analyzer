"""🔊 조음 분석 — 듀얼 전사(목표어/산출형) → 컨퓨전 매트릭스 · PCC · 위치별 오류 (M3)."""

import os
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.articulation import analyze_articulation  # noqa: E402
from modules.insights import generate_insight, summarize_articulation  # noqa: E402
from modules.shared_ui import (  # noqa: E402
    api_key_input,
    child_pairs,
    manual_dual_entry,
    render_articulation_results,
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
        st.session_state.pop("artic_insight", None)

result = st.session_state.get("artic_result")
if result:
    st.divider()
    render_articulation_results(result)

    st.divider()
    st.subheader("임상 코멘트")
    st.text(summarize_articulation(result))
    if st.button("🧠 LLM 임상 코멘트 생성"):
        try:
            with st.spinner("코멘트 생성 중…"):
                st.session_state["artic_insight"] = generate_insight(
                    articulation=result, api_key=api_key)
        except TranscriptionError as e:
            st.error(str(e))
    if st.session_state.get("artic_insight"):
        st.markdown(st.session_state["artic_insight"])

st.divider()
st.page_link("app.py", label="← 홈으로", icon="🏠")
