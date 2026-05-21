"""모드 간 공유 UI 컴포넌트 (언어·조음 페이지 공통)."""

from __future__ import annotations

import hmac
import io
import json
import os
import re

import pandas as pd
import plotly.express as px
import streamlit as st

from modules.articulation import POSITION_ORDER
from modules.morpheme import (
    BROAD_OF,
    GRAM_ORDER,
    SEMANTIC_ORDER,
    SENTENCE_TYPES,
    MorphemeAnalyzer,
)
from modules.transcription import (
    TranscriptionError,
    format_ts,
    slice_audio,
    transcribe_produced,
    transcribe_target,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 전사 페이지 → 언어/조음 분석으로 넘기는 아동 발화(목표어) 세션 키
SHARED_TRANSCRIPT = "shared_child_utterances"

# 배포 확인용 빌드 태그(수정 때마다 갱신). 홈 화면에 표시되어 새 배포 반영 여부를 눈으로 확인.
BUILD_TAG = "2026-05-21q · 전사 검수에 발화 합치기(⬆합치기) 추가"


def g2p_self_test() -> tuple[bool, str]:
    """g2p가 동작하는지 확인(국물→궁물). 규칙 기반이라 네이티브 의존성 없음."""
    try:
        out = get_g2p().to_pronunciation("국물")
        return (out == "궁물"), f"국물→{out}"
    except Exception as e:  # pragma: no cover
        return False, f"{type(e).__name__}: {e}"


@st.cache_resource(show_spinner="형태소 분석기 로딩 중…")
def get_analyzer() -> MorphemeAnalyzer:
    return MorphemeAnalyzer()


@st.cache_resource(show_spinner="g2p 발음 변환기 로딩 중…")
def get_g2p():
    from modules.g2p import G2PConverter
    return G2PConverter()


@st.cache_data
def load_few_shot() -> dict:
    with open(os.path.join(_ROOT, "data", "few_shot_examples.json"), encoding="utf-8") as f:
        return json.load(f)


def get_openai_key() -> str:
    """공용 키(운영자) 우선: st.secrets → 환경변수 → 세션 입력."""
    try:
        k = st.secrets.get("OPENAI_API_KEY", "")
        if k:
            return str(k).strip()
    except Exception:
        pass
    env = (os.getenv("OPENAI_API_KEY") or "").strip()
    if env and not env.startswith("sk-..."):
        return env
    return (st.session_state.get("user_api_key") or "").strip()


def _configured_password() -> str:
    try:
        p = st.secrets.get("APP_PASSWORD", "")
        if p:
            return str(p)
    except Exception:
        pass
    return os.getenv("APP_PASSWORD", "")


def require_password() -> None:
    """비밀번호 게이트. APP_PASSWORD 미설정이면 통과(로컬 개발)."""
    pw = _configured_password()
    if not pw or st.session_state.get("authenticated"):
        return
    st.markdown("## 🔒 접근 비밀번호")
    st.caption("승인된 사용자만 이용할 수 있습니다.")
    entered = st.text_input("비밀번호", type="password", key="pw_input")
    if entered:
        if hmac.compare_digest(entered, pw):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("비밀번호가 올바르지 않습니다.")
    st.stop()


def _server_key() -> str:
    try:
        k = str(st.secrets.get("OPENAI_API_KEY", "") or "")
        if k:
            return k.strip()
    except Exception:
        pass
    env = (os.getenv("OPENAI_API_KEY") or "").strip()
    return env if env and not env.startswith("sk-...") else ""


def _chrono_age_months(birth, test) -> int | None:
    """생년월일·검사일 → 생활연령(총 개월). 계산 불가 시 None."""
    if not birth or not test or test < birth:
        return None
    months = (test.year - birth.year) * 12 + (test.month - birth.month)
    if test.day < birth.day:
        months -= 1
    return max(months, 0)


def _chrono_age(birth, test) -> str:
    """생년월일·검사일 → 생활연령 'N세 M개월'."""
    m = _chrono_age_months(birth, test)
    if m is None:
        return ""
    years, months = divmod(m, 12)
    return f"{years}세 {months}개월"


def patient_info_sidebar() -> dict:
    """사이드바: 대상자 정보(이름/생년월일/검사일/생활연령) + (고급) OpenAI 키."""
    import datetime as _dt
    today = _dt.date.today()
    with st.sidebar:
        st.markdown("### 🧒 대상자 정보")
        name = st.text_input("이름", key="pt_name")
        birth = st.date_input(
            "생년월일", value=None, format="YYYY-MM-DD",
            min_value=_dt.date(1920, 1, 1), max_value=today, key="pt_birth")
        test = st.date_input(
            "검사일", value=today, format="YYYY-MM-DD",
            min_value=_dt.date(1920, 1, 1), max_value=today, key="pt_test")
        age = _chrono_age(birth, test)
        if age:
            st.success(f"생활연령: **{age}**")
        else:
            st.caption("생년월일을 입력하면 생활연령이 자동 계산됩니다.")

        with st.expander("⚙️ 고급 — 음성 기능용 OpenAI 키"):
            if _server_key():
                st.caption("🔑 OpenAI 키: 운영자 설정됨")
            else:
                entered = st.text_input(
                    "sk-...", type="password", key="api_key_field",
                    help="음성 전사·AI 코멘트에만 필요. 직접 입력·텍스트 분석은 키 없이 동작합니다.")
                if entered.strip():
                    st.session_state["user_api_key"] = entered.strip()

    info = {
        "name": (name or "").strip(),
        "birth": birth.isoformat() if birth else "",
        "test": test.isoformat() if test else "",
        "age": age,
        "age_months": _chrono_age_months(birth, test),
    }
    st.session_state["patient_info"] = info
    return info


def api_key_input() -> str:
    """사이드바(대상자 정보 + 고급 OpenAI 키) 렌더 후 사용할 OpenAI 키 반환."""
    patient_info_sidebar()
    return get_openai_key()


# ---------- 음성 듀얼 검수 (목표어/산출형) : 조음 공용 ----------

_SENT_BOUNDARY = re.compile(r"[.!?。！？]+\s*|\n+")


def split_sentences(text) -> list[str]:
    """텍스트를 마침표·물음표·느낌표·줄바꿈 기준으로 분리(빈 조각 제외)."""
    parts = [p.strip() for p in _SENT_BOUNDARY.split(str(text or ""))]
    return [p for p in parts if p]


def _split_pair(target, produced) -> list[tuple[str, str]]:
    """목표어를 기준으로 나누고 산출형을 함께 분리.

    - 목표어를 마침표/줄바꿈으로 분리.
    - 산출형에 같은 수의 종결부호가 있으면 그대로 짝짓는다.
    - 없으면(보통 산출형엔 마침표가 없음) 목표어 각 조각의 어절 수만큼
      산출형 어절을 순서대로 분배한다(조음 정렬이 어절 단위이므로).
    """
    t_parts = split_sentences(target)
    if len(t_parts) <= 1:
        return [(str(target or "").strip(), str(produced or "").strip())]
    p_parts = split_sentences(produced)
    if len(p_parts) == len(t_parts):
        return list(zip(t_parts, p_parts))
    p_words = str(produced or "").split()
    out: list[tuple[str, str]] = []
    idx = 0
    for k, tp in enumerate(t_parts):
        if k == len(t_parts) - 1:
            chunk = p_words[idx:]  # 마지막 조각이 남은 산출형 어절을 모두 가져감
        else:
            n = len(tp.split())
            chunk = p_words[idx:idx + n]
            idx += n
        out.append((tp, " ".join(chunk)))
    return out


def _time_label(start: float, end: float) -> str:
    return f"{format_ts(start)}–{format_ts(end)}"


def _se_of(time_str, lookup: dict) -> tuple[float, float]:
    """편집된 행의 '시간' 문자열 → (start, end). 수동 추가 행은 (0, 0)."""
    key = "" if pd.isna(time_str) else str(time_str)
    return lookup.get(key, (0.0, 0.0))


def child_pairs(edited: pd.DataFrame) -> list[tuple[str, str]]:
    """아동 발화 중 목표어·산출형이 모두 있는 (목표어, 산출형) 쌍."""
    rows = edited[edited["화자"] == "아동"]
    return [
        (str(t).strip(), str(p).strip())
        for t, p in zip(rows["목표어"], rows["산출형"])
        if str(t).strip() and str(p).strip()
    ]


def child_targets(edited: pd.DataFrame) -> list[str]:
    """아동 발화의 목표어(표준어) 리스트."""
    rows = edited[edited["화자"] == "아동"]
    return [str(t).strip() for t in rows["목표어"] if str(t).strip()]


def child_utterances_from(edited: pd.DataFrame) -> list[str]:
    """전사 검수표(화자/전사)에서 아동 발화만 추출."""
    rows = edited[edited["화자"] == "아동"]
    return [str(t).strip() for t in rows["전사"] if str(t).strip()]


def voice_target_review(prefix: str, api_key: str = "") -> pd.DataFrame | None:
    """음성 → Whisper 목표어 전사 + 화자 지정 + 발화 나누기(목표어 전용).

    언어 분석·전사 페이지 공용. 반환: 편집된 DataFrame(화자/전사) 또는 None.
    """
    rows_key, ver_key = f"{prefix}_trows", f"{prefix}_tver"
    uploaded = st.file_uploader(
        "음성 파일 (.mp3, .wav, .m4a)", type=["mp3", "wav", "m4a"], key=f"{prefix}_tupl")
    if uploaded is not None:
        st.audio(uploaded)
        if st.button("🎙️ 자동 전사 시작 (Whisper)", type="primary", key=f"{prefix}_ttr"):
            try:
                with st.spinner("Whisper 전사 중…"):
                    segs = transcribe_target(
                        uploaded.name, uploaded.getvalue(), api_key=api_key)
                st.session_state[rows_key] = [
                    {"start": s["start"], "end": s["end"], "화자": "아동", "전사": s["text"]}
                    for s in segs
                ]
                st.session_state[ver_key] = st.session_state.get(ver_key, 0) + 1
                st.success(f"{len(segs)}개 발화 전사 완료. 화자를 지정하고 검수하세요.")
            except TranscriptionError as e:
                st.error(str(e))

    rows = st.session_state.get(rows_key)
    if not rows:
        st.info("음성을 업로드하고 자동 전사를 시작하세요.")
        return None

    st.markdown("**발화별 검수** — 화자(아동/치료사/제외) 지정 · 전사 수정 · "
                "한 칸에 여러 문장이 있으면 마침표(.)를 넣고 ‘✂️ 발화 나누기’ · "
                "끊긴 발화는 ‘⬆합치기’를 체크하고 ‘🔗 발화 합치기’")
    ver = st.session_state.get(ver_key, 0)
    # 같은 ver 동안 동일한 DataFrame 객체를 재사용해야 data_editor 입력이 유지된다.
    disp_key = f"{prefix}_tdisp_{ver}"
    if disp_key not in st.session_state:
        st.session_state[disp_key] = pd.DataFrame([
            {"#": i + 1, "시간": _time_label(r["start"], r["end"]),
             "화자": r["화자"], "전사": r["전사"], "⬆합치기": False}
            for i, r in enumerate(rows)
        ])
    edited = st.data_editor(
        st.session_state[disp_key], key=f"{prefix}_ttable_{ver}",
        use_container_width=True, hide_index=True,
        num_rows="dynamic", disabled=["#", "시간"],
        column_config={
            "화자": st.column_config.SelectboxColumn(
                "화자", options=["아동", "치료사", "제외"], required=True, width="small"),
            "전사": st.column_config.TextColumn("전사 (수정 가능)", width="large"),
            "⬆합치기": st.column_config.CheckboxColumn(
                "⬆합치기", help="체크한 행을 바로 위 발화에 합칩니다", width="small", default=False),
        },
    )
    lookup = {_time_label(r["start"], r["end"]): (r["start"], r["end"]) for r in rows}
    c_split, c_merge = st.columns(2)
    if c_split.button("✂️ 발화 나누기 (마침표·줄바꿈 기준)", key=f"{prefix}_tsplit",
                      use_container_width=True):
        new_rows = []
        for _, r in edited.iterrows():
            start, end = _se_of(r.get("시간"), lookup)
            spk = r.get("화자") or "아동"
            for part in split_sentences(r.get("전사")):
                new_rows.append({"start": start, "end": end, "화자": spk, "전사": part})
        if new_rows:
            st.session_state[rows_key] = new_rows
            st.session_state[ver_key] = ver + 1
            st.rerun()
    if c_merge.button("🔗 발화 합치기 (⬆합치기 체크한 행을 위에 붙임)", key=f"{prefix}_tmerge",
                      use_container_width=True):
        new_rows: list[dict] = []
        for _, r in edited.iterrows():
            start, end = _se_of(r.get("시간"), lookup)
            spk = r.get("화자") or "아동"
            text = str(r.get("전사") or "").strip()
            if bool(r.get("⬆합치기")) and new_rows:
                prev = new_rows[-1]
                prev["전사"] = (prev["전사"] + " " + text).strip()
                prev["end"] = max(prev["end"], end)
            else:
                new_rows.append({"start": start, "end": end, "화자": spk, "전사": text})
        new_rows = [r for r in new_rows if r["전사"]]
        if new_rows:
            st.session_state[rows_key] = new_rows
            st.session_state[ver_key] = ver + 1
            st.rerun()
    n_child = int((edited["화자"] == "아동").sum())
    st.caption(f"아동 발화로 지정된 항목: {n_child}개 (이 항목만 분석/저장)")
    return edited


def _rows_from_edited(edited: pd.DataFrame, lookup: dict) -> list[dict]:
    """편집된 표 → 표준 행 리스트(화자/목표어/산출형 + 시간 복원)."""
    rows = []
    for _, r in edited.iterrows():
        start, end = _se_of(r.get("시간"), lookup)
        rows.append({
            "start": start, "end": end,
            "화자": r.get("화자") or "아동",
            "목표어": str(r.get("목표어") or "").strip(),
            "산출형": str(r.get("산출형") or "").strip(),
        })
    return rows


def voice_dual_review(prefix: str, api_key: str = "") -> pd.DataFrame | None:
    """음성 업로드 → Whisper 목표어 + 화자 지정 + 산출형 듀얼 검수 표.

    반환: 편집된 DataFrame(화자/목표어/산출형) 또는 None(아직 전사 전).
    """
    rows_key, audio_key = f"{prefix}_rows", f"{prefix}_audio"
    ver_key, msg_key = f"{prefix}_ver", f"{prefix}_msg"

    uploaded = st.file_uploader(
        "음성 파일 (.mp3, .wav, .m4a)", type=["mp3", "wav", "m4a"], key=f"{prefix}_upl")
    if uploaded is not None:
        st.audio(uploaded)
        if st.button("🎙️ 목표어 전사 시작 (Whisper)", type="primary", key=f"{prefix}_tr"):
            try:
                with st.spinner("Whisper 전사 중…"):
                    segs = transcribe_target(
                        uploaded.name, uploaded.getvalue(), api_key=api_key)
                st.session_state[rows_key] = [
                    {"start": s["start"], "end": s["end"], "화자": "아동",
                     "목표어": s["text"], "산출형": ""}
                    for s in segs
                ]
                st.session_state[audio_key] = (uploaded.name, uploaded.getvalue())
                st.session_state[ver_key] = st.session_state.get(ver_key, 0) + 1
                st.session_state.pop(msg_key, None)
                st.success(f"{len(segs)}개 발화 전사 완료. 화자 지정 후 산출형을 입력/생성하세요.")
            except TranscriptionError as e:
                st.error(str(e))

    rows = st.session_state.get(rows_key)
    if not rows:
        st.info("음성을 업로드하고 목표어 전사를 시작하세요.")
        return None

    for level, text in st.session_state.pop(msg_key, []):
        getattr(st, level)(text)

    st.markdown(
        "**듀얼 검수** — 화자(아동/치료사/제외) 지정 · 목표어/산출형 수정 · "
        "목표어 한 칸에 여러 문장이 있으면 마침표(.)를 넣고 **‘✂️ 발화 나누기’**를 누르면 "
        "산출형도 어절 단위로 함께 나뉩니다.")

    ver = st.session_state.get(ver_key, 0)
    # 같은 ver 동안 동일한 DataFrame 객체를 재사용해야 data_editor 입력이 유지된다.
    disp_key = f"{prefix}_disp_{ver}"
    if disp_key not in st.session_state:
        st.session_state[disp_key] = pd.DataFrame([
            {"#": i + 1, "시간": _time_label(r["start"], r["end"]),
             "화자": r["화자"], "목표어": r["목표어"], "산출형": r["산출형"]}
            for i, r in enumerate(rows)
        ])
    edited = st.data_editor(
        st.session_state[disp_key], key=f"{prefix}_table_{ver}",
        use_container_width=True, hide_index=True,
        num_rows="dynamic", disabled=["#", "시간"],
        column_config={
            "화자": st.column_config.SelectboxColumn(
                "화자", options=["아동", "치료사", "제외"], required=True, width="small"),
            "목표어": st.column_config.TextColumn("목표어 (표준어)", width="medium"),
            "산출형": st.column_config.TextColumn("산출형 (실제 발음)", width="medium"),
        },
    )

    lookup = {_time_label(r["start"], r["end"]): (r["start"], r["end"]) for r in rows}
    c_split, c_gen = st.columns(2)

    if c_split.button("✂️ 발화 나누기 (목표어 기준 · 산출형 동반)", key=f"{prefix}_split",
                      use_container_width=True):
        new_rows = []
        for r in _rows_from_edited(edited, lookup):
            for tgt, prod in _split_pair(r["목표어"], r["산출형"]):
                if tgt or prod:
                    new_rows.append({"start": r["start"], "end": r["end"],
                                     "화자": r["화자"], "목표어": tgt, "산출형": prod})
        if new_rows:
            st.session_state[rows_key] = new_rows
            st.session_state[ver_key] = ver + 1
            st.rerun()

    if c_gen.button("🤖 GPT-4o로 산출형 자동 생성 (아동·빈칸만)", key=f"{prefix}_gen",
                    use_container_width=True):
        fname, abytes = st.session_state[audio_key]
        few = load_few_shot()
        new_rows = _rows_from_edited(edited, lookup)
        todo = [i for i, r in enumerate(new_rows)
                if r["화자"] == "아동" and not r["산출형"] and r["end"] > r["start"]]
        done, errors = 0, []
        with st.spinner(f"산출형 생성 중… ({len(todo)}건)"):
            for i in todo:
                r = new_rows[i]
                try:
                    clip, _ = slice_audio(abytes, fname, r["start"], r["end"])
                    new_rows[i]["산출형"] = transcribe_produced(
                        "seg.wav", clip, few, api_key=api_key)
                    done += 1
                except TranscriptionError as e:
                    errors.append(str(e))
        st.session_state[rows_key] = new_rows
        st.session_state[ver_key] = ver + 1
        msgs = []
        if done:
            msgs.append(("success", f"{done}건 산출형 생성 완료."))
        if errors:
            msgs.append(("error", f"{len(errors)}건 실패 — {errors[0]}"))
        if not todo:
            msgs.append(("info",
                         "자동 생성 대상이 없습니다. (화자=아동 · 산출형이 비어 있음 · "
                         "시간 정보가 있는 행만 대상; 수동 추가 행은 제외)"))
        st.session_state[msg_key] = msgs
        st.rerun()

    st.caption("‘발화 나누기’는 목표어의 마침표/줄바꿈을 기준으로 행을 나누고, 산출형은 "
               "어절 수에 맞춰 함께 분리합니다(시간·화자 유지). 산출형에도 마침표가 있으면 그 기준을 우선합니다. "
               "자동 생성은 빈 산출형(아동)만 채우며 입력한 산출형은 보존됩니다.")
    return edited


_TARGET_KW = ("목표", "표준", "target", "standard", "낱말", "단어", "어절", "발화", "문장", "어휘")
_PRODUCED_KW = ("산출", "produced", "실제", "발음", "오류")


def _read_manual_table(file) -> list[dict]:
    """업로드한 엑셀/CSV → [{목표어, 산출형}] 리스트.

    헤더에 '목표어/표준철자', '산출형' 등이 있으면 그 열을 매핑하고,
    헤더가 없으면 1열=목표어, 2열=산출형으로 본다.
    """
    name = (getattr(file, "name", "") or "").lower()
    if name.endswith(".csv"):
        try:
            raw = pd.read_csv(file, header=None, dtype=object, encoding="utf-8-sig")
        except Exception:
            file.seek(0)
            raw = pd.read_csv(file, header=None, dtype=object, encoding="cp949")
    else:
        raw = pd.read_excel(file, header=None, engine="openpyxl")
    if raw is None or raw.empty:
        return []
    raw = raw.fillna("")
    first = [str(x).strip() for x in raw.iloc[0].tolist()]
    has_header = any(any(k in c.lower() for k in _TARGET_KW + _PRODUCED_KW) for c in first)
    tcol, pcol = 0, (1 if raw.shape[1] > 1 else None)
    if has_header:
        tcol, pcol = None, None
        for i, name_i in enumerate(first):
            low = name_i.lower()
            if tcol is None and any(k in low for k in _TARGET_KW):
                tcol = i
            if pcol is None and any(k in low for k in _PRODUCED_KW):
                pcol = i
        if tcol is None:
            tcol = 0
        body = raw.iloc[1:]
    else:
        body = raw
    rows = []
    for _, r in body.iterrows():
        tgt = str(r.iloc[tcol]).strip() if tcol is not None and tcol < len(r) else ""
        prod = str(r.iloc[pcol]).strip() if pcol is not None and pcol < len(r) else ""
        if tgt or prod:
            rows.append({"목표어": tgt, "산출형": prod})
    return rows


def _editor_current_rows(disp_key: str, editor_key: str) -> list[dict]:
    """data_editor의 현재 내용(직전 편집 포함)을 [{목표어, 산출형}]로 반환.

    버튼이 표보다 위에 있어도 직전 run의 위젯 편집 상태를 합쳐 최신 입력을 잃지 않는다.
    """
    base = st.session_state.get(disp_key)
    if base is None:
        return []
    rows = base.to_dict("records")
    state = st.session_state.get(editor_key) or {}
    for k, ch in (state.get("edited_rows") or {}).items():
        try:
            i = int(k)
        except (TypeError, ValueError):
            continue
        if 0 <= i < len(rows):
            rows[i] = {**rows[i], **ch}
    for add in (state.get("added_rows") or []):
        rows.append(dict(add))
    deleted = {int(x) for x in (state.get("deleted_rows") or [])}
    return [
        {"목표어": str(r.get("목표어") or "").strip(),
         "산출형": str(r.get("산출형") or "").strip()}
        for i, r in enumerate(rows) if i not in deleted
    ]


def manual_dual_entry(prefix: str) -> pd.DataFrame:
    """음성 없이 목표어/산출형을 직접 입력하는 검수 표(조음).

    임상가가 귀로 들은 산출형을 직접 전사한다. 목표어(표준 철자)에는 g2p가
    자연스러운 음운변동(연음·경음화·비음화 등)을 자동 적용하므로, 자연 변동은
    오류로 잡히지 않는다. 엑셀/CSV 업로드로 목표어를 일괄 입력할 수 있다.
    반환: 화자/목표어/산출형 DataFrame.
    """
    rows_key, ver_key = f"{prefix}_mrows", f"{prefix}_mver"
    msg_key, sig_key = f"{prefix}_mmsg", f"{prefix}_msig"
    if rows_key not in st.session_state:
        st.session_state[rows_key] = [{"목표어": "", "산출형": ""} for _ in range(6)]

    st.markdown(
        "**직접 입력** — 한 줄에 한 낱말/발화. **목표어**는 표준 철자(예: `먹어요`), "
        "**산출형**은 아동이 실제로 낸 발음을 들리는 대로(예: `머거요`) 한글로 적습니다.")

    up = st.file_uploader(
        "엑셀/CSV로 목표어 일괄 불러오기 (.xlsx, .csv) — 목표어·산출형 열 자동 인식",
        type=["xlsx", "csv"], key=f"{prefix}_mupl")
    if up is not None:
        sig = (up.name, getattr(up, "size", None))
        if st.session_state.get(sig_key) != sig:
            st.session_state[sig_key] = sig
            try:
                imported = _read_manual_table(up)
            except Exception as e:
                imported = []
                st.session_state[msg_key] = ("error", f"파일을 읽지 못했습니다: {e}")
            if imported:
                st.session_state[rows_key] = imported
                st.session_state[ver_key] = st.session_state.get(ver_key, 0) + 1
                st.session_state[msg_key] = (
                    "success", f"{len(imported)}개 행을 불러왔습니다. 목표어·산출형을 검수하세요.")
            elif msg_key not in st.session_state:
                st.session_state[msg_key] = (
                    "warning", "불러올 데이터가 없습니다. 첫 시트에 목표어(또는 목표어/산출형) "
                    "열이 있는지 확인하세요.")
            st.rerun()

    msg = st.session_state.pop(msg_key, None)
    if msg:
        getattr(st, msg[0])(msg[1])

    ver = st.session_state.get(ver_key, 0)
    # 같은 ver 동안 동일한 DataFrame 객체를 재사용해야 data_editor 입력이 유지된다
    # (매 리런마다 새 DataFrame을 넘기면 편집 내용이 초기화됨).
    disp_key = f"{prefix}_mdisp_{ver}"
    editor_key = f"{prefix}_mtable_{ver}"
    if disp_key not in st.session_state:
        st.session_state[disp_key] = pd.DataFrame(
            st.session_state[rows_key], columns=["목표어", "산출형"])

    cb1, cb2 = st.columns(2)
    shared = st.session_state.get(SHARED_TRANSCRIPT) or []
    if cb1.button(f"📋 전사에서 목표어 불러오기 ({len(shared)}개)", key=f"{prefix}_mfromtrans",
                  use_container_width=True, disabled=not shared):
        st.session_state[rows_key] = [{"목표어": t, "산출형": ""} for t in shared]
        st.session_state[ver_key] = st.session_state.get(ver_key, 0) + 1
        st.session_state[msg_key] = ("success",
                                     f"전사에서 {len(shared)}개 목표어를 불러왔습니다. 산출형을 전사하세요.")
        st.rerun()
    # 산출형을 목표 발음형(자연 음운변동 적용)으로 미리 채우고, 임상가가 오류만 수정
    if cb2.button("📝 산출형 = 목표 발음형으로 채우기 (빈칸만)", key=f"{prefix}_mfill",
                  type="primary", use_container_width=True):
        g2p = get_g2p()
        new_rows = []
        for r in _editor_current_rows(disp_key, editor_key):
            tgt, prod = r["목표어"], r["산출형"]
            if tgt and not prod:
                prod = g2p.to_pronunciation_words(tgt)
            if tgt or prod:
                new_rows.append({"목표어": tgt, "산출형": prod})
        if new_rows:
            st.session_state[rows_key] = new_rows
            st.session_state[ver_key] = ver + 1
            st.rerun()
    st.caption("‘산출형 = 목표 발음형으로 채우기’를 누르면 빈 산출형에 표준 발음형(연음·경음화 등 "
               "자연 변동 적용)이 채워집니다. 아동이 다르게 낸 음소만 고치면 됩니다(전부 입력할 필요 없음).")

    edited = st.data_editor(
        st.session_state[disp_key], key=editor_key,
        use_container_width=True, hide_index=True, num_rows="dynamic",
        column_config={
            "목표어": st.column_config.TextColumn("목표어 (표준 철자)", width="medium"),
            "산출형": st.column_config.TextColumn("산출형 (들리는 실제 발음)", width="medium"),
        },
    )

    if st.button("✂️ 발화 나누기 (목표어 기준 · 산출형 동반)", key=f"{prefix}_msplit"):
        new_rows = []
        for _, r in edited.iterrows():
            for tgt, prod in _split_pair(r.get("목표어"), r.get("산출형")):
                if tgt or prod:
                    new_rows.append({"목표어": tgt, "산출형": prod})
        if new_rows:
            st.session_state[rows_key] = new_rows
            st.session_state[ver_key] = ver + 1
            st.rerun()

    out = edited.copy()
    out["화자"] = "아동"
    return out


# ---------- 엑셀 내보내기 ----------

XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def transcript_to_excel(utterances: list[str]) -> bytes:
    """아동 발화(목표어) → 엑셀(목표어·산출형 열). 조음 직접 입력에서 재업로드 가능."""
    df = pd.DataFrame({"목표어": utterances, "산출형": ["" for _ in utterances]})
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="전사", index=False)
    return buf.getvalue()


def report_download_button(language: dict | None = None,
                           articulation: dict | None = None, key: str = "") -> None:
    """분석 결과 → HTML 보고서 다운로드 버튼(브라우저 인쇄로 PDF 가능)."""
    from modules.report import build_report_html
    patient = st.session_state.get("patient_info") or {}
    # 말명료도는 조음 지표 — 조음 보고서에만 포함
    intelligibility = st.session_state.get("intelligibility") if articulation is not None else None
    doc = build_report_html(language=language, articulation=articulation, patient=patient,
                            intelligibility=intelligibility)
    safe = "".join(c for c in (patient.get("name") or "자발화") if c.isalnum() or c in " _-").strip()
    st.download_button(
        "📄 HTML 보고서 저장", data=doc.encode("utf-8"),
        file_name=f"{safe or '자발화'}_분석보고서.html", mime="text/html", key=f"rep_{key}")
    st.caption("다운로드한 HTML을 브라우저에서 열고 인쇄(Ctrl+P) → ‘PDF로 저장’하면 PDF가 됩니다.")


def _language_detail_df(result: dict) -> pd.DataFrame:
    return pd.DataFrame([
        {"#": i + 1, "발화": u["text"], "문장유형": u["sentence_type"],
         "낱말": u["words"], "형태소": u["morphemes"],
         **{c: u["semantic"][c] for c in SEMANTIC_ORDER}}
        for i, u in enumerate(result["utterances"])
    ])


def _word_freq_df(result: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [{"낱말": w["word"], "품사": w["category"], "빈도": w["count"]}
         for w in result["stats"]["word_freq"]])


def language_to_excel(result: dict) -> bytes:
    stats = result["stats"]
    summary = pd.DataFrame({
        "지표": ["발화 수", "MLU-w(평균 낱말)", "MLU-m(평균 형태소)", "TTR",
               "TNW(총 낱말)", "NDW(서로 다른 낱말)", "총 형태소",
               "체언", "용언", "수식언", "독립언"],
        "값": [stats["utterance_count"], stats["mlu_w"], stats["mlu_m"], stats["ttr"],
              stats["tnw"], stats["ndw"], stats["total_morphemes"],
              stats["broad_counts"].get("체언", 0), stats["broad_counts"].get("용언", 0),
              stats["broad_counts"].get("수식언", 0), stats["broad_counts"].get("독립언", 0)],
    })
    sem = pd.DataFrame({
        "품사": SEMANTIC_ORDER,
        "대분류": [BROAD_OF[c] for c in SEMANTIC_ORDER],
        "총 낱말": [stats["semantic_counts"][c] for c in SEMANTIC_ORDER],
        "서로 다른 낱말": [stats["semantic_ndw"][c] for c in SEMANTIC_ORDER],
    })
    gram = pd.DataFrame({"문법형태소": GRAM_ORDER,
                         "빈도": [stats["gram_categories"][c] for c in GRAM_ORDER]})
    order = {c: i for i, c in enumerate(GRAM_ORDER)}
    gmf = stats.get("gram_morpheme_freq", [])
    gram_form = pd.DataFrame(
        [{"범주": g["category"], "형태소": g["morpheme"], "빈도": g["count"]} for g in gmf]) \
        if gmf else pd.DataFrame(columns=["범주", "형태소", "빈도"])
    if not gram_form.empty:
        gram_form = (gram_form.assign(_o=gram_form["범주"].map(order).fillna(99))
                     .sort_values(["_o", "빈도"], ascending=[True, False])
                     .drop(columns="_o").reset_index(drop=True))
    sent = pd.DataFrame({"문장유형": SENTENCE_TYPES,
                         "발화 수": [stats["sentence_types"][s] for s in SENTENCE_TYPES]})
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name="요약", index=False)
        sem.to_excel(xw, sheet_name="의미영역", index=False)
        gram.to_excel(xw, sheet_name="문법형태소", index=False)
        gram_form.to_excel(xw, sheet_name="조사어미빈도", index=False)
        sent.to_excel(xw, sheet_name="문장유형", index=False)
        _language_detail_df(result).to_excel(xw, sheet_name="발화별", index=False)
        _word_freq_df(result).to_excel(xw, sheet_name="낱말빈도", index=False)
    return buf.getvalue()


def articulation_to_excel(result: dict) -> bytes:
    s = result["summary"]
    summary = pd.DataFrame({
        "지표": ["PCC(%)", "목표 자음 수", "정확 자음 수", "오류 수", "첨가",
               "PVC(%)", "목표 모음 수", "정확 모음 수", "모음 오류 수"],
        "값": [result["pcc"], s["total_consonants"], s["correct_consonants"],
              s["error_count"], s["additions"],
              result.get("pvc", 0.0), s.get("total_vowels", 0),
              s.get("correct_vowels", 0), s.get("vowel_error_count", 0)],
    })
    pe, pt = result["position_errors"], result["position_total"]
    pos = pd.DataFrame({
        "위치": POSITION_ORDER,
        "오류": [pe[p] for p in POSITION_ORDER],
        "전체": [pt[p] for p in POSITION_ORDER],
        "오류율(%)": [round(pe[p] / pt[p] * 100, 1) if pt[p] else 0.0 for p in POSITION_ORDER],
    })
    pa = pd.DataFrame({"음소": list(result["phoneme_accuracy"].keys()),
                       "정확도(%)": list(result["phoneme_accuracy"].values())})
    conf_rows = [{"목표": t, "산출": p, "빈도": c}
                 for t, row in result["confusion_matrix"].items() for p, c in row.items()]
    conf = pd.DataFrame(conf_rows) if conf_rows else pd.DataFrame(columns=["목표", "산출", "빈도"])
    errs = pd.DataFrame(result["errors"]).rename(columns={
        "word": "목표어절", "target_pron": "목표발음", "produced_word": "산출어절",
        "target": "목표(음소)", "produced": "산출(음소)", "position": "위치",
        "process": "음운변동"}) \
        if result["errors"] else pd.DataFrame(
            columns=["목표어절", "목표발음", "산출어절", "목표(음소)", "산출(음소)", "위치", "음운변동"])
    pp = result.get("phonological_processes") or []
    proc = pd.DataFrame(
        [{"음운변동": x["process"], "유형": x["type"], "빈도": x["count"]} for x in pp]) \
        if pp else pd.DataFrame(columns=["음운변동", "유형", "빈도"])
    va = result.get("vowel_accuracy") or {}
    vowel_acc = pd.DataFrame({"모음": list(va.keys()), "정확도(%)": list(va.values())})
    vconf_rows = [{"목표": t, "산출": p, "빈도": c}
                  for t, row in (result.get("vowel_confusion_matrix") or {}).items()
                  for p, c in row.items()]
    vconf = pd.DataFrame(vconf_rows) if vconf_rows else pd.DataFrame(columns=["목표", "산출", "빈도"])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name="요약", index=False)
        pos.to_excel(xw, sheet_name="위치별오류", index=False)
        pa.to_excel(xw, sheet_name="음소정확도", index=False)
        conf.to_excel(xw, sheet_name="컨퓨전매트릭스", index=False)
        proc.to_excel(xw, sheet_name="음운변동", index=False)
        errs.to_excel(xw, sheet_name="오류상세", index=False)
        vowel_acc.to_excel(xw, sheet_name="모음정확도", index=False)
        vconf.to_excel(xw, sheet_name="모음컨퓨전", index=False)
    return buf.getvalue()


# ---------- 언어 분석 결과 렌더링 ----------

def render_language_results(result: dict) -> None:
    stats = result["stats"]
    st.download_button(
        "⬇️ 엑셀로 전체 결과 다운로드", data=language_to_excel(result),
        file_name="언어분석.xlsx", mime=XLSX_MIME, key="dl_lang")

    st.subheader("핵심 지표")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("발화 수", stats["utterance_count"])
    c2.metric("MLU-w (평균 낱말)", stats["mlu_w"])
    c3.metric("MLU-m (평균 형태소)", stats["mlu_m"])
    c4.metric("TTR (어휘 다양도)", stats["ttr"])
    c5, c6, c7 = st.columns(3)
    c5.metric("TNW (총 낱말 수)", stats["tnw"])
    c6.metric("NDW (서로 다른 낱말)", stats["ndw"])
    c7.metric("총 형태소 수", stats["total_morphemes"])
    st.caption(
        "낱말 = 체언+용언+수식언+독립언 (용언은 기본형). "
        "MLU-w = 총 낱말 / 발화 수 · MLU-m = 총 형태소 / 발화 수 · TTR = NDW / TNW.")

    st.divider()
    st.subheader("의미 영역 (품사 세분화)")
    b = stats["broad_counts"]
    bc1, bc2, bc3, bc4 = st.columns(4)
    bc1.metric("체언", b.get("체언", 0))
    bc2.metric("용언", b.get("용언", 0))
    bc3.metric("수식언", b.get("수식언", 0))
    bc4.metric("독립언", b.get("독립언", 0))
    sem_df = pd.DataFrame({
        "품사": SEMANTIC_ORDER,
        "대분류": [BROAD_OF[c] for c in SEMANTIC_ORDER],
        "총 낱말": [stats["semantic_counts"][c] for c in SEMANTIC_ORDER],
        "서로 다른 낱말": [stats["semantic_ndw"][c] for c in SEMANTIC_ORDER],
    })
    fig = px.bar(
        sem_df.melt(id_vars=["품사", "대분류"], var_name="구분", value_name="빈도"),
        x="품사", y="빈도", color="구분", barmode="group", text="빈도",
        category_orders={"품사": SEMANTIC_ORDER},
        color_discrete_map={"총 낱말": "#4C78A8", "서로 다른 낱말": "#9ECAE1"})
    fig.update_layout(xaxis_title="", legend_title="", height=380)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(sem_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("문법형태소 (세분류)")
    gcat = stats["gram_categories"]
    gcat_df = pd.DataFrame({"문법형태소": GRAM_ORDER, "빈도": [gcat[c] for c in GRAM_ORDER]})
    gfig = px.bar(gcat_df, x="문법형태소", y="빈도", text="빈도",
                  category_orders={"문법형태소": GRAM_ORDER},
                  color="빈도", color_continuous_scale="Blues")
    gfig.update_layout(xaxis_title="", coloraxis_showscale=False, height=340)
    st.plotly_chart(gfig, use_container_width=True)
    st.caption("피동·사동 접사는 형태소 분석기가 어간에 병합하여 자동 분리되지 않습니다 (임상가 검수).")

    st.markdown("**조사·어미 사용 빈도 (형태소별)**")
    gmf = stats.get("gram_morpheme_freq", [])
    if gmf:
        order = {c: i for i, c in enumerate(GRAM_ORDER)}
        gmf_df = pd.DataFrame(
            [{"범주": g["category"], "형태소": g["morpheme"], "빈도": g["count"]} for g in gmf])
        gmf_df = (gmf_df.assign(_o=gmf_df["범주"].map(order).fillna(99))
                  .sort_values(["_o", "빈도"], ascending=[True, False])
                  .drop(columns="_o").reset_index(drop=True))
        conn = gmf_df[gmf_df["범주"] == "연결어미"]
        if not conn.empty:
            st.success("🔗 연결어미 사용: "
                       + ", ".join(f"{r.형태소}({r.빈도})" for r in conn.itertuples()))
        pick = st.multiselect(
            "범주 필터", GRAM_ORDER, default=GRAM_ORDER, key="gram_form_filter")
        view = gmf_df[gmf_df["범주"].isin(pick)] if pick else gmf_df
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.caption("연결어미·전성어미 등 어미 형태소의 실제 사용형과 빈도입니다 (임상가 검수).")
    else:
        st.info("검출된 조사·어미가 없습니다.")

    with st.expander("문법형태소 상세 (조사·어미 종류별 — 격조사/어미 유형)"):
        gram = stats["grammatical_morphemes"]
        if gram:
            st.dataframe(pd.DataFrame({"종류": list(gram.keys()), "빈도": list(gram.values())}),
                         use_container_width=True, hide_index=True)
        else:
            st.info("검출된 문법형태소가 없습니다.")

    st.divider()
    st.subheader("문장유형 (자동 추정)")
    sent = stats["sentence_types"]
    s1, s2, s3 = st.columns(3)
    s1.metric("단문", sent["단문"])
    s2.metric("이어진문장", sent["이어진문장"])
    s3.metric("안긴문장", sent["안긴문장"])
    sent_df = pd.DataFrame({"문장유형": SENTENCE_TYPES, "발화 수": [sent[s] for s in SENTENCE_TYPES]})
    sfig = px.bar(sent_df, x="문장유형", y="발화 수", text="발화 수",
                  category_orders={"문장유형": SENTENCE_TYPES}, color="문장유형",
                  color_discrete_map={"단문": "#4C78A8", "이어진문장": "#F58518", "안긴문장": "#54A24B"})
    sfig.update_layout(xaxis_title="", showlegend=False, height=320)
    st.plotly_chart(sfig, use_container_width=True)
    st.caption("연결어미·전성어미 기반 자동 추정 — 임상가 검수 필요.")

    st.divider()
    st.subheader("발화별 상세")
    detail_df = pd.DataFrame([
        {"#": i + 1, "발화": u["text"], "문장유형": u["sentence_type"],
         "낱말": u["words"], "형태소": u["morphemes"],
         **{c: u["semantic"][c] for c in SEMANTIC_ORDER}}
        for i, u in enumerate(result["utterances"])
    ])
    st.dataframe(detail_df, use_container_width=True, hide_index=True)
    with st.expander("형태소 분해 보기 (낱말=기본형/품사 표시)"):
        for i, u in enumerate(result["utterances"]):
            st.markdown(f"**{i + 1}. {u['text']}**")
            parts = [
                f"`{t['headword']}`⟨{t['category']}⟩" if t["category"]
                else f"`{t['form']}`/{t['tag']}"
                for t in u["tokens"]
            ]
            st.markdown("  ".join(parts))
    st.divider()
    st.subheader("낱말 빈도표")
    st.caption(f"총 낱말(TNW): {stats['tnw']} · 서로 다른 낱말(NDW): {stats['ndw']} · 표의 빈도 합 = TNW")
    if stats["word_freq"]:
        st.dataframe(_word_freq_df(result), use_container_width=True, hide_index=True)
    else:
        st.info("낱말이 없습니다.")


# ---------- 조음 분석 결과 렌더링 ----------

def render_articulation_results(result: dict) -> None:
    s = result["summary"]
    st.download_button(
        "⬇️ 엑셀로 전체 결과 다운로드", data=articulation_to_excel(result),
        file_name="조음분석.xlsx", mime=XLSX_MIME, key="dl_artic")
    st.subheader("결과")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PCC (자음정확도)", f"{result['pcc']}%")
    c2.metric("목표 자음 수", s["total_consonants"])
    c3.metric("오류 수", s["error_count"])
    c4.metric("첨가", s["additions"])
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("PVC (모음정확도)", f"{result.get('pvc', 0.0)}%")
    v2.metric("목표 모음 수", s.get("total_vowels", 0))
    v3.metric("모음 오류 수", s.get("vowel_error_count", 0))
    v4.metric("정확 모음 수", s.get("correct_vowels", 0))
    st.caption("PCC = 정확 자음 / 목표 자음 × 100 (초성 ㅇ 제외) · PVC = 정확 모음 / 목표 모음 × 100. "
               "목표어는 g2p 발음형으로 변환 후 비교.")

    st.divider()
    st.subheader("컨퓨전 매트릭스 (목표 → 산출)")
    cm = result["confusion_matrix"]
    if cm:
        targets = sorted(cm.keys())
        produced = sorted({p for row in cm.values() for p in row})
        z = [[cm[t].get(p, 0) for p in produced] for t in targets]
        fig = px.imshow(z, x=produced, y=targets, text_auto=True, aspect="auto",
                        color_continuous_scale="Reds",
                        labels=dict(x="산출 음소", y="목표 음소", color="빈도"))
        fig.update_layout(height=max(320, 40 * len(targets)))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("∅ = 생략(대응 산출 음소 없음)")
    else:
        st.success("자음 오류가 없습니다.")

    st.divider()
    st.subheader("위치별 오류")
    pe, pt = result["position_errors"], result["position_total"]
    pos_df = pd.DataFrame({
        "위치": POSITION_ORDER,
        "오류": [pe[p] for p in POSITION_ORDER],
        "전체": [pt[p] for p in POSITION_ORDER],
    })
    pos_df["오류율(%)"] = [round(e / t * 100, 1) if t else 0.0
                         for e, t in zip(pos_df["오류"], pos_df["전체"])]
    pfig = px.bar(pos_df, x="위치", y="오류", text="오류",
                  category_orders={"위치": POSITION_ORDER},
                  color="위치", color_discrete_sequence=px.colors.qualitative.Set2)
    pfig.update_layout(xaxis_title="", showlegend=False, height=320)
    st.plotly_chart(pfig, use_container_width=True)
    st.dataframe(pos_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("음소별 정확도")
    pa = result["phoneme_accuracy"]
    if pa:
        pa_df = pd.DataFrame({"음소": list(pa.keys()), "정확도(%)": list(pa.values())})
        afig = px.bar(pa_df, x="음소", y="정확도(%)", text="정확도(%)",
                      color="정확도(%)", color_continuous_scale="RdYlGn", range_y=[0, 100])
        afig.update_layout(xaxis_title="", coloraxis_showscale=False, height=340)
        st.plotly_chart(afig, use_container_width=True)

    pp = result.get("phonological_processes") or []
    st.divider()
    st.subheader("오류 음운변동 패턴 (상대분석)")
    st.caption(
        "목표 발음형 대비 산출형의 차이를 한국어 음운변동으로 분류한 결과입니다. "
        "경음화·비음화·유음화 등 의무적(자연스러운) 변동은 목표 발음형에 이미 반영되어 "
        "오류로 잡히지 않습니다. ‘비전형’은 정상발달에서 드물어 장애를 시사할 수 있습니다 (임상가 검수).")
    if pp:
        pp_df = pd.DataFrame(
            [{"음운변동": x["process"], "유형": x["type"], "빈도": x["count"]} for x in pp])
        ppfig = px.bar(pp_df, x="음운변동", y="빈도", color="유형", text="빈도",
                       color_discrete_map={"발달적": "#4C78A8", "비전형": "#E45756"})
        ppfig.update_layout(xaxis_title="", legend_title="", height=360)
        st.plotly_chart(ppfig, use_container_width=True)
        st.dataframe(pp_df, use_container_width=True, hide_index=True)
        atyp = [x["process"] for x in pp if x["type"] == "비전형"]
        if atyp:
            st.warning("⚠️ 비전형(비발달적) 패턴: " + ", ".join(atyp) + " — 임상가 검수 권장.")
        if result["summary"].get("syllable_omissions"):
            st.caption(f"음절 생략(추정) {result['summary']['syllable_omissions']}회 — "
                       "모음은 음절핵이라 생략 불가하므로 음절 단위 생략으로 해석합니다.")
    else:
        st.success("분류된 오류 음운변동이 없습니다.")

    va = result.get("vowel_accuracy") or {}
    vcm = result.get("vowel_confusion_matrix") or {}
    if va or vcm:
        st.divider()
        st.subheader("모음 (중성) 분석")
        st.caption("자음 PCC와 분리한 모음 정확도. 모음 대치·생략 경향 확인용 (임상가 검수).")
        if va:
            va_df = pd.DataFrame({"모음": list(va.keys()), "정확도(%)": list(va.values())})
            vfig = px.bar(va_df, x="모음", y="정확도(%)", text="정확도(%)",
                          color="정확도(%)", color_continuous_scale="RdYlGn", range_y=[0, 100])
            vfig.update_layout(xaxis_title="", coloraxis_showscale=False, height=320)
            st.plotly_chart(vfig, use_container_width=True)
        if vcm:
            v_targets = sorted(vcm.keys())
            v_prod = sorted({p for row in vcm.values() for p in row})
            vz = [[vcm[t].get(p, 0) for p in v_prod] for t in v_targets]
            vcfig = px.imshow(vz, x=v_prod, y=v_targets, text_auto=True, aspect="auto",
                              color_continuous_scale="Purples",
                              labels=dict(x="산출 모음", y="목표 모음", color="빈도"))
            vcfig.update_layout(height=max(280, 40 * len(v_targets)))
            st.plotly_chart(vcfig, use_container_width=True)
            st.caption("∅ = 생략(대응 산출 모음 없음)")

    if result["errors"]:
        with st.expander(f"자음 오류 상세 ({len(result['errors'])}건)"):
            st.caption("‘산출어절’은 아동이 실제로 낸 어절입니다 (예: 목표 ‘같이’→발음 ‘가치’, "
                       "산출 ‘다치’이면 어두초성 ㄱ→ㄷ).")
            st.dataframe(
                pd.DataFrame(result["errors"]).rename(columns={
                    "word": "목표어절", "target_pron": "목표발음", "produced_word": "산출어절",
                    "target": "목표(음소)", "produced": "산출(음소)", "position": "위치",
                    "process": "음운변동"}),
                use_container_width=True, hide_index=True)
    if result.get("vowel_errors"):
        with st.expander(f"모음 오류 상세 ({len(result['vowel_errors'])}건)"):
            st.dataframe(
                pd.DataFrame(result["vowel_errors"]).rename(columns={
                    "word": "목표어절", "target_pron": "목표발음", "produced_word": "산출어절",
                    "target": "목표(모음)", "produced": "산출(모음)"}),
                use_container_width=True, hide_index=True)
    if result.get("additions_detail"):
        with st.expander(f"첨가 상세 ({len(result['additions_detail'])}건)"):
            st.dataframe(
                pd.DataFrame(result["additions_detail"]).rename(columns={
                    "produced": "산출(첨가음소)", "position": "위치",
                    "target_word": "목표어절", "produced_word": "산출어절"}),
                use_container_width=True, hide_index=True)
