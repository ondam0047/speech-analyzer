"""자발화 분석 도구 — 메인 (홈 + 모드 선택)."""

import os
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from modules.shared_ui import (  # noqa: E402
    BUILD_TAG,
    api_key_input,
    g2p_self_test,
    require_password,
)

st.set_page_config(
    page_title="자발화 분석 도구",
    page_icon="🎙️",
    layout="wide",
)
require_password()
api_key_input()

st.title("🎙️ 자발화 분석 도구")
st.caption("언어치료 자발화 분석 — 음성/텍스트 → 형태소·조음 분석 → 보고서 (로컬 도구)")

with st.expander("🔧 배포 확인 / g2p 작동 테스트", expanded=False):
    st.caption(f"빌드 태그: **{BUILD_TAG}**  ← 이 태그가 보이면 최신 코드가 배포된 것입니다.")
    if st.button("g2p 작동 확인 (국물 → 궁물)"):
        ok, out = g2p_self_test()
        if ok:
            st.success(f"✅ g2p 정상 작동: {out} (발음형 변환·조음 분석 정상)")
        else:
            st.error(f"⚠️ g2p 미작동: {out} (배포 미완료 가능 → 잠시 후 새로고침/Reboot)")

st.caption("아래 칸을 누르면 해당 단계로 바로 이동합니다.")

st.markdown("### 1단계 · 전사 (선택)")
if st.button(
        "🎤 **전사**  \n"
        "음성 업로드 → 자동 전사 → 화자 지정 → 검수 → 분석에 사용 / 엑셀 저장  \n"
        "🗣️ **말명료도(이해가능도) 자동 계산** · 텍스트가 이미 있으면 건너뛰어도 됩니다",
        use_container_width=True, key="go_trans"):
    st.switch_page("pages/0_🎤_전사.py")

st.markdown("### 2단계 · 분석")
col1, col2 = st.columns(2, gap="large")
with col1:
    if st.button(
            "📝 **언어 분석**  \n"
            "MLU · TTR · NDW · 문법형태소  \n"
            "🧠 연령 기반 LLM 코멘트  \n"
            "입력: 전사 · 텍스트 · 음성",
            use_container_width=True, key="go_lang"):
        st.switch_page("pages/1_📝_언어_분석.py")
with col2:
    if st.button(
            "🔊 **조음 분석**  \n"
            "PCC · PVC · 오류 음운변동 · 위치별 오류  \n"
            "🗣️ 말명료도 · 🧠 연령 기반 LLM 코멘트  \n"
            "입력: 직접 입력(목표어/산출형) · 음성",
            use_container_width=True, key="go_artic"):
        st.switch_page("pages/2_🔊_조음_분석.py")

st.divider()

with st.expander("ℹ️ 사용 순서 안내"):
    st.markdown(
        """
        0. **사이드바**에 대상자 **생년월일·검사일**을 입력하면 생활연령이 계산되고,
           이때만 **🧠 LLM 임상 코멘트**(연령 규준 대비 해석)를 생성할 수 있습니다.
        1. **🎤 전사** — 음성을 올려 자동 전사 → 화자(아동/치료사/제외) 지정 → 검수
           (마침표/엔터로 발화 나누기) → **분석에 사용**(세션) 또는 **엑셀 저장**.
           못 알아들은 부분은 음절 수만큼 `*`로 표기 → **🗣️ 말명료도** 자동 계산.
        2. **📝 언어 분석** — 전사 결과 불러오기 / 텍스트 입력 / 음성 업로드 후 분석.
           MLU·TTR·NDW, 의미영역, 조사·어미 빈도, 문장유형 + 연령 기반 LLM 코멘트.
        3. **🔊 조음 분석** — 전사에서 목표어 불러오기(또는 직접 입력) → **산출형을 목표 발음형으로 채운 뒤**
           아동이 다르게 낸 음소만 수정 → PCC·PVC, 오류 음운변동, 위치별 오류,
           **🗣️ 말명료도**(목표어를 알아들었는지) + 연령 기반 LLM 코멘트.
        4. 각 분석 후 **📄 HTML 보고서**로 저장(결과 + LLM 코멘트 포함, 브라우저 인쇄 → PDF 가능).

        > 환자 음성은 업로드 시 서버·OpenAI로 전송됩니다. 보호자 동의·기관 정책을 확인하세요.
        """
    )
