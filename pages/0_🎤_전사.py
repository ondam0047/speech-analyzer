"""🎤 전사 — 음성 → 자동 전사 → 화자 지정(아동만) → 검수 → 분석에 사용 / 엑셀 저장.

분석(언어/조음) 전 단계. 아동 발화만 골라 저장하면 언어·조음 분석에서 바로 불러올 수 있다.
"""

import os
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.shared_ui import (  # noqa: E402
    SHARED_TRANSCRIPT,
    XLSX_MIME,
    api_key_input,
    child_utterances_from,
    require_password,
    transcript_to_excel,
    voice_target_review,
)

st.set_page_config(page_title="전사", page_icon="🎤", layout="wide")
require_password()
api_key = api_key_input()

st.title("🎤 전사")
st.caption("음성 → 자동 전사 → 화자(아동/치료사/제외) 지정 → 검수 → 분석에 사용 또는 엑셀 저장")

st.markdown(
    "1) 음성을 업로드해 자동 전사 → 2) **아동 발화만** 화자 지정 + 검수(마침표로 발화 나누기) → "
    "3) **분석에 사용** 또는 **엑셀로 저장**. 저장한 아동 발화는 언어·조음 분석에서 불러올 수 있습니다.")

edited = voice_target_review("trans", api_key)

if edited is not None:
    child = child_utterances_from(edited)
    st.divider()
    st.subheader("저장 / 분석에 사용")
    st.caption(f"아동 발화 {len(child)}개가 저장 대상입니다.")
    c1, c2 = st.columns(2)
    if c1.button(f"💾 분석에 사용 (세션 저장 · {len(child)}개)", type="primary",
                 use_container_width=True, disabled=not child):
        st.session_state[SHARED_TRANSCRIPT] = child
        st.success("저장 완료. 언어/조음 분석 페이지에서 ‘전사 결과 불러오기/전사에서 목표어 불러오기’로 사용하세요.")
    c2.download_button(
        "⬇️ 엑셀로 저장 (목표어)", data=transcript_to_excel(child),
        file_name="전사_아동발화.xlsx", mime=XLSX_MIME,
        use_container_width=True, disabled=not child)

    if st.session_state.get(SHARED_TRANSCRIPT):
        with st.expander(f"현재 저장된 아동 발화 ({len(st.session_state[SHARED_TRANSCRIPT])}개)"):
            for i, u in enumerate(st.session_state[SHARED_TRANSCRIPT], 1):
                st.write(f"{i}. {u}")

st.divider()
st.page_link("app.py", label="← 홈으로", icon="🏠")
