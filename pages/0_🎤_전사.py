"""🎤 전사 — 음성 → 자동 전사 → 화자 지정(아동만) → 검수 → 분석에 사용 / 엑셀 저장.

분석(언어/조음) 전 단계. 아동 발화만 골라 저장하면 언어·조음 분석에서 바로 불러올 수 있다.
"""

import os
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd  # noqa: E402

from modules.intelligibility import compute_intelligibility  # noqa: E402
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
st.info("🗣️ **말명료도 표기** — 대상자가 말했으나 알아듣지 못한 부분은 **음절 수만큼 `*`**로 적으세요. "
        "예) 3음절을 못 알아들었으면 `***`, 일부만 이해되면 `엄***`. 아래에서 명료도가 자동 계산됩니다.")

edited = voice_target_review("trans", api_key)

if edited is not None:
    child = child_utterances_from(edited)

    st.divider()
    st.subheader("🗣️ 말명료도 (이해가능도)")
    if child:
        intel = compute_intelligibility(child)
        st.session_state["intelligibility"] = intel
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("어절 명료도", f"{intel['word_intelligibility']}%")
        c2.metric("음절 명료도", f"{intel['syllable_intelligibility']}%")
        c3.metric("이해 어절 / 전체", f"{intel['intelligible_words']} / {intel['total_words']}")
        c4.metric("불명료 어절", intel["unintelligible_words"])
        st.caption("어절 명료도 = ‘*’ 없는 어절 / 전체 어절 × 100 · "
                   "음절 명료도 = (전체 음절 − ‘*’) / 전체 음절 × 100. "
                   "이 값은 보고서에도 포함됩니다.")
        with st.expander("발화별 명료도 상세"):
            st.dataframe(
                pd.DataFrame(intel["per_utterance"]).rename(columns={
                    "utterance": "발화", "words": "어절", "intelligible_words": "이해 어절",
                    "unintelligible_words": "불명료 어절", "syllables": "음절",
                    "unintelligible_syllables": "불명료 음절(*)"}),
                use_container_width=True, hide_index=True)
    else:
        st.session_state.pop("intelligibility", None)
        st.caption("아동 발화를 지정하면 말명료도가 자동 계산됩니다.")

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
