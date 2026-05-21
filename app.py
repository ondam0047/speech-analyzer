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

st.markdown("### 1단계 · 전사 (선택)")
with st.container(border=True):
    st.subheader("🎤 전사")
    st.write("음성 업로드 → 자동 전사 → 화자 지정(아동만) → 검수 → **분석에 사용 / 엑셀 저장**")
    st.caption("여기서 저장한 아동 발화를 아래 언어·조음 분석에서 바로 불러올 수 있습니다. "
               "(텍스트가 이미 있으면 이 단계를 건너뛰어도 됩니다.)")
    st.page_link("pages/0_🎤_전사.py", label="전사 시작하기 →", use_container_width=True)

st.markdown("### 2단계 · 분석")

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    with st.container(border=True):
        st.subheader("📝 언어 분석")
        st.write("**MLU, TTR, NDW**")
        st.write("문법형태소·연결어미 분포")
        st.write("입력: 전사 불러오기 · 텍스트 · 음성")
        st.page_link("pages/1_📝_언어_분석.py", label="시작하기 →", use_container_width=True)

with col2:
    with st.container(border=True):
        st.subheader("🔊 조음 분석")
        st.write("**PCC·PVC, 오류 음운변동**")
        st.write("컨퓨전 매트릭스 · 위치별 오류")
        st.write("입력: 직접 입력(목표어/산출형) · 음성")
        st.page_link("pages/2_🔊_조음_분석.py", label="시작하기 →", use_container_width=True)

with col3:
    with st.container(border=True):
        st.subheader("🎯 통합 분석")
        st.write("**언어 + 조음**")
        st.write("종합 보고서")
        st.write("입력: 직접 입력 · 음성")
        st.page_link("pages/3_🎯_통합_분석.py", label="시작하기 →", use_container_width=True)

st.divider()

with st.expander("ℹ️ 사용 순서 안내"):
    st.markdown(
        """
        1. **🎤 전사** — 음성을 올려 자동 전사 → 화자(아동/치료사/제외) 지정 → 검수
           (마침표/엔터로 발화 나누기) → **분석에 사용**(세션) 또는 **엑셀 저장**.
        2. **📝 언어 분석** — 전사 결과 불러오기 / 텍스트 입력 / 음성 업로드 후 분석.
           MLU-w·MLU-m·TTR·NDW·TNW, 의미영역, 조사·어미(연결어미) 빈도, 문장유형.
        3. **🔊 조음 분석** — 전사에서 목표어 불러오기(또는 직접 입력) → 임상가가 **산출형(실제 발음)**
           전사 → PCC·PVC, 오류 음운변동(상대분석), 위치별 오류. 자연스러운 음운변동은 오류로 잡히지 않음.
        4. **🎯 통합 분석** — 언어 + 조음 동시 + 종합 코멘트.
        5. 각 분석 후 **📄 HTML 보고서**로 저장(브라우저 인쇄 → PDF 가능).

        > 환자 음성은 업로드 시 서버·OpenAI로 전송됩니다. 보호자 동의·기관 정책을 확인하세요.
        """
    )
