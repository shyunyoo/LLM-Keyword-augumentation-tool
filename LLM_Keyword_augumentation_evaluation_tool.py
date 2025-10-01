import os, csv, json, time, datetime
from pathlib import Path
from typing import List
import streamlit as st
import streamlit.components.v1 as components  # 추가: 타이머/팝업을 위한 import
import math
import sys
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

# 기본 디렉토리
def get_base_dir() -> Path:
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent

BASE_DIR = get_base_dir()
DATA_PATH = BASE_DIR / "dataset.csv"
CSV_DELIMITER = ","
FILENAME_COL = "filename"
LABEL_COL = "label"
CSV_COL_MAP = {"filename": FILENAME_COL, "label": LABEL_COL}

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
ITEMS_PER_PAGE = 30
TIME_LIMIT_MINUTES = 10
SURVEY_URL = "https://docs.google.com/forms/d/e/1FAIpQLSc3JLpWSRCEhxl8DEo-gqzbWsyyAUajepJOFDv_GRL6-c9JEg/viewform?usp=header"
st.set_page_config(page_title="Phase B Test Page")

#st.set_page_config(page_title="Phase B", layout="wide")

# Fade out 비활성화
st.markdown("""
    <style>
    div.stAlert, div.stSpinner, div.element-container, div.row-widget.stButton, div.row-widget.stCheckbox {
        transition: none !important;
        animation: none !important;
    }
    </style>
""", unsafe_allow_html=True)

ENCODING_CANDIDATES = ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin1"]

def _norm_id(name: str) -> str:
    """중복 판단용 키(대소문자/앞뒤 공백 차이 무시). 필요시 경로 정규화 규칙을 여기서 확장."""
    return (name or "").strip().lower()

def load_saved_from_logs(pid: str) -> tuple[set[str], set[str]]:
    """
    기존 로그를 스캔해 이미 '증거로 저장'된 파일들을 복원(세션 재시작 대비).
    evidence_mark / evidence_mark_on_timeout 이벤트의 payload를 합집합으로 반영.
    """
    saved, keys = set(), set()
    log_path = LOG_DIR / "phase_b" / f"{pid}.csv"
    if not log_path.exists():
        return saved, keys

    with log_path.open("r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            if row.get("event") not in ("evidence_mark", "evidence_mark_on_timeout"):
                continue
            try:
                payload = json.loads(row.get("payload") or "[]")
            except json.JSONDecodeError:
                continue
            if isinstance(payload, list):
                for name in payload:
                    k = _norm_id(name)
                    if k not in keys:
                        keys.add(k)
                        saved.add(name)
    return saved, keys

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not Path(DATA_PATH).exists():
        st.error(f"{DATA_PATH} 파일을 찾을 수 없습니다."); st.stop()

    last_err = None
    for enc in ENCODING_CANDIDATES:
        try:
            df = pd.read_csv(
                DATA_PATH,
                delimiter=CSV_DELIMITER,
                encoding=enc,
                dtype=str
            ).fillna("")
            #st.info(f"CSV 인코딩: {enc}")
            break
        except UnicodeDecodeError as e:
            last_err = e
            continue
    else:
        st.error(f"Data file loading failed!! Please contact the administrator: {last_err}"); st.stop()

    norm = (
        df.columns
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
    )
    df.columns = norm

    rename_map = {}
    for std_name, real_name in CSV_COL_MAP.items():
        real_norm = real_name.lower().strip()
        match_cols = [c for c in df.columns if c == real_norm]
        if match_cols:
            rename_map[match_cols[0]] = std_name
    df = df.rename(columns=rename_map)

    if "filename" not in df.columns:
        st.error("CSV에서 'filename' 열을 찾지 못했습니다.\n"
                 f"→ 실제 헤더: {list(df.columns)}")
        st.stop()

    if "label" not in df.columns:
        df["label"] = ""

    return df

@st.cache_data(show_spinner=False)
def search(
    df: pd.DataFrame,
    keywords: List[str],
    threshold: float,
    *,                       # 키워드 필터 옵션은 키워드 인자로만 전달
    min_len: int = 2,        # N글자 미만은 무시
    ignore_single_digit: bool = True  # 한 자리 숫자 필터
) -> pd.DataFrame:
    """
    - min_len:  이 길이보다 짧은 키워드는 검색에서 제외
    - ignore_single_digit: True 이면 0~9 단독 키워드는 무시
    """

    # ── 1 키워드 필터링 ──────────────────────
    filtered_kw = []
    for kw in keywords:
        kw = kw.strip()
        if len(kw) < min_len:
            continue
        if ignore_single_digit and kw.isdigit() and len(kw) == 1:
            continue
        filtered_kw.append(kw)

    # 필터링 결과가 없으면 빈 DF 반환
    if not filtered_kw:
        return df.iloc[0:0]

    # ── 2 검색용 헬퍼 ──────────────────────
    def normalize(txt: str) -> str:
        return txt.lower().replace(" ", "")

    def score(name: str) -> float:
        norm_name = normalize(name)
        for kw in filtered_kw:
            if normalize(kw) in norm_name:
                return 1.0
        return 0.0

    # ── 3 스코어 계산 & 필터링 ──────────────
    out = df.copy()
    out["score"] = out["filename"].apply(score)
    return out[out["score"] >= threshold].sort_values("score", ascending=False)

def open_new_tab(url: str):
    components.html(
        f"""
        <script>
            window.open('{url}', '_blank', 'noopener,noreferrer');
        </script>
        <!-- dummy {time.time_ns()} -->
        """,
        height=0, width=0
    )

SYSTEM_PROMPT = """
# Objective
You are an investigator with many years of practical experience in the field of digital forensics. Analyze the keywords entered by the user, derive semantically, thematically, and contextually related single-word keywords, and output exactly N according to the specified JSON schema. If you need a more formal or detailed version, here is a slightly longer and clarified option:

## Input
- Format: keyword1, keyword2, keyword3, number_of_keywords_to_output
- Example 1: police agency, sexual harassment, statistics, 30
- Example 2: drugs, smuggling, BTC, 30
- Example 3: military, blueprint, operation, 30

# Output
```json
{
  "keywords": [
    "keyword1",
    "keyword2",
    ...
  ]
}

# Instructions
1. Each keyword must be a single word and cannot contain spaces, hyphens, or underscores.
2. Every keyword should appear only once, with no duplicates (including homonyms and alternate spellings).
3. Ensure semantic diversity: include a balanced mix of synonyms, hypernyms (broader terms), hyponyms (narrower terms), and related words.
4. Arrange keywords in a logical order considering relevance and usefulness; do not randomize the sequence.
5. Hierarchy ratio (for N=30):
   - Hypernyms: at least 12
   - Mid-level terms: 10-12
   - Hyponyms: no more than 8
6. Include at least 5 specialized/professional terms, slang, or abbreviations (from any domain).
7. Self-check before output:
   - Remove banned words, duplicates, and typos to ensure exactly N keywords
   - If ratio or rules are violated, regenerate automatically

# Output-only
Do not include any additional explanations, comments, or line breaks other than the JSON object.
Output the result only once.
"""

def fetch_llm_keywords(base_keywords: List[str], n: int = 30, retry: int = 3) -> List[str]:
    user_input = ", ".join(base_keywords + [f"{n}개"])
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 환경변수가 없습니다.")
    client = OpenAI(api_key=api_key, timeout=45)

    for _ in range(retry):
        try:
            resp = client.chat.completions.create(
                model="gpt-5-chat-latest",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.3,
                presence_penalty=0.8,
                max_tokens=800,
                response_format={"type": "json_object"}
            )
            data = json.loads(resp.choices[0].message.content)
            kw = [k.strip() for k in data.get("keywords", []) if k.strip()]
            return list(dict.fromkeys(kw))[:n]
        except Exception:
            time.sleep(2)
    return []

def log_event(pid: str, event: str, payload: any):
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    phase_dir = LOG_DIR / "phase_b"
    phase_dir.mkdir(exist_ok=True)
    log_path = phase_dir / f"{pid}.csv"
    
    if isinstance(payload, (dict, list)):
        payload_str = json.dumps(payload, ensure_ascii=False)
    else:
        payload_str = str(payload)
    
    if not log_path.exists():
        with log_path.open("w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(["timestamp", "event", "payload"])
    
    with log_path.open("a", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerow([ts, event, payload_str])

st.sidebar.title("File Explorer with LLM Integration")

pid = st.sidebar.text_input("Name", placeholder="e.g.) Smith").strip()
if not pid:
    st.sidebar.warning("Input your name"); st.stop()

#st.sidebar.caption("※ 로그는 logs/phase_b/ 에 저장됩니다")

# ---- Session State 초기화 (pid 입력 직후에 위치) ----
if "manual_selected" not in st.session_state:
    st.session_state.manual_selected = set()

# 제가 제안드린 누적 저장 기능을 쓰신다면 이것도 함께 초기화
if "evidence_saved" not in st.session_state or "evidence_saved_keys" not in st.session_state:
    # 로그에서 복원하는 유틸을 추가하셨다면 그걸 호출
    try:
        saved, keys = load_saved_from_logs(pid)   # 없으면 except로 빠짐
    except Exception:
        saved, keys = set(), set()
    st.session_state.evidence_saved = saved
    st.session_state.evidence_saved_keys = keys

if "evidence_saved" not in st.session_state or "evidence_saved_keys" not in st.session_state:
    saved, keys = load_saved_from_logs(pid)
    st.session_state.evidence_saved = saved
    st.session_state.evidence_saved_keys = keys

# 증거 관리
saved_count = len(st.session_state.evidence_saved)
st.sidebar.success(f"Selected Evidence: **{saved_count}**")
with st.sidebar.expander("List of Stored Evidence (Most Recent 10)", expanded=False):
    for i, name in enumerate(list(sorted(st.session_state.evidence_saved))[:10], 1):
        ell = "..." if len(name) > 50 else ""
        st.write(f"{i}. {name[:50]}{ell}")
    if saved_count > 10:
        st.caption(f"… more {saved_count - 10}")

# 현재 선택 현황
selected_count = len(st.session_state.manual_selected)
if selected_count > 0:
    st.sidebar.info(f"Receently selected: **{selected_count}**")
    with st.sidebar.expander("List of Selected files"):
        for idx, filename in enumerate(list(st.session_state.manual_selected)[:5], 1):
            ell = "..." if len(filename) > 50 else ""
            st.write(f"{idx}. {filename[:50]}{ell}")
        if selected_count > 5:
            st.write(f"... more {selected_count - 5}")
else:
    st.sidebar.info("no selected files")

if st.sidebar.button("📋 selected items to evidence", type="primary",
                     disabled=(selected_count == 0), use_container_width=True, key="evidence_save_btn"):
    try:
        selected_files = list(st.session_state.manual_selected)
        # 중복 제거: 이미 저장된 것은 제외
        new_items = [fn for fn in selected_files if _norm_id(fn) not in st.session_state.evidence_saved_keys]

        if new_items:
            log_event(pid, "evidence_mark", new_items)
            for fn in new_items:
                st.session_state.evidence_saved.add(fn)
                st.session_state.evidence_saved_keys.add(_norm_id(fn))
            st.toast(f"✅ 신규 {len(new_items)}개 저장 완료 (누적 {len(st.session_state.evidence_saved)}개)", icon="✅")
        else:
            st.toast("⚠️ 이미 저장된 항목만 선택되었습니다. 신규 저장 없음.", icon="⚠️")

        # 다음 검색/선택을 편하게 하기 위해 선택 목록은 비움
        st.session_state.manual_selected = set()
    except Exception as e:
        st.sidebar.error(f"저장 중 오류: {str(e)}")

if selected_count > 0:
    if st.sidebar.button("🗑️ Clear All Selections", use_container_width=True, key="clear_selection_btn"):
        st.session_state.manual_selected = set()

st.title("LLM Augment Tool Test")
df = load_data()
#st.markdown(f"📝 [사후 설문지 열기]({SURVEY_URL})")

# ── 타이머 초기화 ──────────────────────────
if "start_time" not in st.session_state:
    st.session_state["start_time"] = time.time()
    st.session_state["time_up"] = False
    log_event(pid, "phase_B_start", f"Phase B started: {TIME_LIMIT_MINUTES} minutes")

elapsed = time.time() - st.session_state["start_time"]
remaining_sec = TIME_LIMIT_MINUTES * 60 - elapsed
if remaining_sec <= 0:
    st.session_state["time_up"] = True

initial_remaining = max(int(remaining_sec), 0)

timeout_message = 'Phase B가 종료되었습니다! 실험이 완료되었습니다.'
post_timeout_action = f"""
alert('You must click the "Save Selected Items as Evidence" button in the left panel! If you do not press the save button, the checked file list will not be saved as evidence!');
alert('Phase B has ended! The experiment has been completed.');
if (confirm('Please fill out the survey. If the page does not load, please use the separate link on the main page!')) {{
    window.open('{SURVEY_URL}', '_blank', 'noopener,noreferrer');
}}
setTimeout(function() {{ window.location.href = 'about:blank'; }}, 3000);
"""

# 타이머를 사이드바로 이동 (상단 배치, 원본 파란색 텍스트 스타일 유지)
with st.sidebar:
    st.markdown("### ⏳ Timer")  # 상단 배치
    js_code = f"""
    <div id="timer" style="font-size: 20px; color: blue;">Time Remain: 00:00</div>
    <script>
        var remaining = {initial_remaining};
        var timerElement = document.getElementById('timer');
        var alerted10 = false;
        var interval = setInterval(function() {{
            if (remaining <= 0) {{
                clearInterval(interval);
                timerElement.innerHTML = 'Time Exceed! {timeout_message}';
                timerElement.style.color = 'red';
                {post_timeout_action}
            }} else {{
                if (remaining === 10 && !alerted10) {{
                    alert('10 seconds remaining! Time is almost up.');
                    alerted10 = true;
                }}
                var mins = Math.floor(remaining / 60);
                var secs = remaining % 60;
                timerElement.innerHTML = 'Time Remain: ' + (mins < 10 ? '0' : '') + mins + ':' + (secs < 10 ? '0' : '') + secs;
                remaining--;
            }}
        }}, 1000);
    </script>
    """
    components.html(js_code, height=50)

js_code = f"""
<div id='timer' style='font-size:20px;color:blue;'>Time Remain: 00:00</div>
<script>
  var remaining      = {initial_remaining};
  var timerEl        = document.getElementById('timer');
  var toast60Shown   = false;   // 60초 알림 1-회용
  var toast30Shown   = false;   // 30초 알림 1-회용

  var interval = setInterval(function () {{
    if (remaining <= 0) {{
      clearInterval(interval);
      timerEl.innerHTML = '시간 초과! {timeout_message}';
      timerEl.style.color = 'red';
      {post_timeout_action}
      return;
    }}

    /* ---------- 타이머 표시 ---------- */
    var m = Math.floor(remaining / 60),
        s = remaining % 60;
    timerEl.innerHTML =
      'Time Remain: ' +
      (m < 10 ? '0':'') + m + ':' +
      (s < 10 ? '0':'') + s;

    /* ---------- 토스트 경고 ---------- */
    if (remaining === 60 && !toast60Shown) {{
      toast60Shown = true;
      showGlobalToast('1 minute remaining', 3000);
    }}
    if (remaining === 30 && !toast30Shown) {{
      toast30Shown = true;
      showGlobalToast('30 seconds remaining! Please begin summarizing your results now!', 3000);
    }}

    remaining--;
  }}, 1000);

  /* ---------- 토스트 함수 ---------- */
  function showGlobalToast(msg, dur) {{
    dur = dur || 3000;
    var doc = window.parent.document;
    var id  = 'global-toast';

    // 이미 다른 토스트가 떠 있으면 잠깐 건너뜀
    // if (doc.getElementById(id)) return;

    var t = doc.createElement('div');
    t.id = id;
    t.textContent = msg;
    t.style.cssText =
    'position:fixed;left:50%;bottom:30px;transform:translateX(-50%);' +
    'background:rgba(0,0,0,.85);color:#fff;' +
    /* ──────────────▼ 여기 둘을 키워 보세요 ─────────────── */
    'padding:20px 28px;' +        // 안쪽 여백(세로 20px, 가로 28px)
    'font-size:18px;' +           // 글자 크기
    /* ──────────────────────────────────────────────────── */
    'border-radius:8px;' +        // (모서리 둥글기도 필요하면)
    'min-width:240px;' +          // 원하는 경우 최소폭 지정
    'z-index:2147483647;' +
    'opacity:0;transition:opacity .3s,transform .3s;';

    doc.body.appendChild(t);

    // fade-in
    setTimeout(function () {{
      t.style.opacity   = '1';
      t.style.transform = 'translateX(-50%) translateY(-10px)';
    }}, 10);

    // dur 뒤 fade-out 후 DOM 제거
    setTimeout(function () {{
      t.style.opacity   = '0';
      t.style.transform = 'translateX(-50%) translateY(0)';
      setTimeout(function () {{
        if (t.parentNode) t.parentNode.removeChild(t);
      }}, 300);
    }}, dur);
  }}
</script>
"""

# 타이머 + 토스트 영역 높이
st.components.v1.html(js_code, height=120)

if st.session_state["time_up"]:
    log_event(pid, "phase_B_end", "Time limit exceeded for B")
    
    # [CHANGED] 타이머 종료 시 '신규'만 자동 저장
    selected_files = list(st.session_state.manual_selected)
    new_items = [fn for fn in selected_files if _norm_id(fn) not in st.session_state.evidence_saved_keys]
    if new_items:
        log_event(pid, "evidence_mark_on_timeout", new_items)
        for fn in new_items:
            st.session_state.evidence_saved.add(fn)
            st.session_state.evidence_saved_keys.add(_norm_id(fn))
        st.toast(f"✅ Time exceed: Auto saved : {len(new_items)} (cumulative total {len(st.session_state.evidence_saved)}개)", icon="✅")
    else:
        st.toast("⏰ Timer ended: There are no new items to auto-save.", icon="⏰")
    st.error("Phase B has ended. The experiment has been completed.")
    st.stop()

# ── 1단계 팝업 (앱 로드 시 1회) ──────────────────────────
phase_step1_key = "phase_B_step1_popup"
if not st.session_state.get(phase_step1_key, False):
    st.session_state[phase_step1_key] = True
    msg = """
🤖 Phase B begins!\n
🎯 Read the description of the crime and enter as many keywords as you can think of.
→ Use the keywords you entered along with the "ChatGPT recommended keywords" to search!
Try to find evidence!\n
    """
    safe_msg = json.dumps(msg, ensure_ascii=False)
    components.html(
        f"""
        <script>
            setTimeout(function() {{
                alert({safe_msg});
            }}, 800);
        </script>
        """,
        height=0, width=0
    )
    log_event(pid, "phase_B_step1_popup", "shown")

# --- ❶ 2단계 팝업 처리 ---------------------------------
#if st.session_state.pop("show_step2_popup", False):
#    msg = """
#🔄 2단계: 다수 입력 방식 (1단계 완료 후 남은 시간)  
#목적: 반복 검색을 통한 증거 수집 능력 평가  
#방법: 여러 번 키워드 입력/수정 가능 (LLM 추천 활용 가능)  
#팁: 검색 결과 확인 후 즉시 키워드 조정

#최대한 많은 증거를 찾아보세요!
#    """
#    safe_msg = json.dumps(msg, ensure_ascii=False)
#    components.html(
#        f"""
#        <script>
#            alert({safe_msg});
#        </script>
#        """,
#        height=0, width=0
#    )
#    # (선택) 로깅
#    log_event(st.session_state.get("pid",""), "phase_B_step2_popup", "shown")
# --------------------------------------------------------

#사건내용
QUESTION_MD = """
Suspect A (XX years old, contract worker at the Ministry of Land, Infrastructure and Transport) took advantage of the opportunity when most employees had left work during the holiday period. On XX:XX on Month XX, Day XX, XXXX, the suspect unlawfully entered the offices of real estate policy-related departments including the Land Policy Division and Housing Policy Division at the Government Complex Sejong of the Ministry of Land, Infrastructure and Transport. Using unspecified computers that were present, the suspect used a pre-prepared external hard drive to <span style="color:red; font-weight:900;"> unlawfully copy and store internal documents including internal reports, approval documents, and press releases </span>  that were stored in each department's shared folders and personal folders.\n

During the investigation, a total of approximately 1,500 files were discovered on the external hard drive confiscated from the suspect. Of these, approximately 500 documents are estimated to have been stolen from real estate policy-related departments.

**[Experimental Objective]**
**From the approximately 1,500 files on the confiscated external hard drive, identify and classify the approximately 500 documents estimated to have been obtained from real estate policy-related departments through keyword searches.**

(Document examples: **Internal reports**, **approval documents**, **press releases**, and other **files presumed to be held on department staff PCs** related to the work of **real estate policy-related departments**
<span style="color:red; font-weight:900;"> However, other administrative files unrelated to real estate policy work should be excluded from the evidence to be collected </span>)\n
"""
with st.expander("📝 View Scenario / Close", expanded=True):
    st.markdown(QUESTION_MD, unsafe_allow_html=True)
st.divider()


# 1단계/2단계 헤더 변환 (개선 4)
if "first_search_done" not in st.session_state:
    st.session_state.first_search_done = False

if not st.session_state.first_search_done:
    st.subheader("Please enter keywords")
else:
    #st.subheader("키워드 입력 ")
    st.sidebar.markdown(
    f'<div class="emphasized-link">📝 <a href="{SURVEY_URL}" target="_blank"> Open survey</a></div>',
    unsafe_allow_html=True)

# ── 키워드 입력 폼 ────────────────────────────────────────────
with st.form("base_kw_form", clear_on_submit=False):
    base_kw_raw = st.text_input(
        "Basic Keywords (separate with commas, all keywords use OR logic)",
        placeholder="Enter at least 3 keywords together, e.g., police, traffic, enforcement, investigation",
        help="Use commas or spaces as separators. Example: police, traffic, enforcement"
    )
    submitted = st.form_submit_button("Input Initial Keywords")  # ← 새 버튼

# Enter 키 ↔ 버튼 클릭: 어떤 방법이든 submitted == True
if submitted or base_kw_raw:          # 입력·제출 둘 중 하나라도 있으면
    base_kw = base_kw_raw.replace(",", " ").split()
else:
    st.info("Please enter your keywords and press Enter or click the **Enter Default Keywords** button.")
    st.stop()

N_OUT = 30
st.write("##### LLM Augmented Keywords")
if st.button(f"Generate {N_OUT}Augmented Keywords", disabled=len(base_kw)==0):
    log_event(pid, "click_generate", ",".join(base_kw))
    with st.spinner("Calling the model..."):
        try:
            rec_kw = fetch_llm_keywords(base_kw, n=N_OUT)
            st.session_state["rec_kw"] = rec_kw
            log_event(pid, "llm_keywords", "|".join(rec_kw))
        except Exception as e:
            st.error(f"Model error: {e}")
rec_kw = st.session_state.get("rec_kw", [])
picked = st.multiselect("Select Additional Keywords", rec_kw, default=rec_kw) if rec_kw else []

final_kw = list(dict.fromkeys(base_kw + picked))

if final_kw:
    st.success("keywords: " + ", ".join(final_kw))  # 복원: 입력 키워드 표시
else:
    st.info("Input keywords or select"); st.stop()

if st.button("Search"):
    st.session_state.first_search_done = True  # 2단계 전환 (개선 4)
    keyword_payload = {
        "base_keywords": base_kw,
        "llm_keywords": picked
    }
    log_event(pid, "search", keyword_payload)
    
    res_df = search(df, final_kw, 1.0)
    st.session_state["result"] = res_df

    # ---- 팝업 플래그: 아직 안 보여줬을 때만 ----
    if not st.session_state.get("step2_popup_shown", False):
        st.session_state["show_step2_popup"] = True   # ❶에서 읽음
        st.session_state["step2_popup_shown"] = True  # 영구 표시
    
    if not res_df.empty:
        hit_files = res_df["filename"].tolist()
        log_event(pid, "search_results", hit_files)
    else:
        log_event(pid, "search_results", [])
        safe_msg = json.dumps(msg, ensure_ascii=False)
        components.html(
            f"""
            <script>
                alert({safe_msg});
            </script>
            """,
            height=0, width=0
        )
        log_event(pid, "phase_B_step2_popup", "shown")
        
    
    # rerun으로 헤더 즉시 변환 반영 (검색 결과 유지)
    st.rerun()

##검색결과 토글 리스트업하는 페이지 시작

res_df = st.session_state.get("result")
if res_df is not None:
    if res_df.empty:
        st.info("검색 결과가 없습니다.")
    else:
        st.subheader(f"Search Result - {len(res_df)}")

        if "current_page" not in st.session_state:
            st.session_state.current_page = 1

        total_pages = math.ceil(len(res_df) / ITEMS_PER_PAGE)
        start_idx = (st.session_state.current_page - 1) * ITEMS_PER_PAGE
        end_idx = start_idx + ITEMS_PER_PAGE
        view = res_df.iloc[start_idx:end_idx].reset_index(drop=True)

        col_all_on, col_all_off = st.columns(2)
        with col_all_on:
            if st.button("Select all in page"):
                for fn in view["filename"]:
                    st.session_state.manual_selected.add(fn)
                st.rerun()
        with col_all_off:
            if st.button("Deselect all in page"):
                for fn in view["filename"]:
                    st.session_state.manual_selected.discard(fn)
                st.rerun()

        for i, row in view.iterrows():
            fn = row["filename"]
            chk_key = f"chk_B_{fn}_{id(fn)}"
            btn_key = f"btn_B_{fn}_{id(fn)}"
            
            col_flag, col_txt = st.columns([0.06, 0.94])
            
            is_selected = fn in st.session_state.manual_selected

            def toggle_file(fn, key):
                """체크박스 on_change 콜백"""
                if st.session_state[key]:      # 체크 ON
                    st.session_state.manual_selected.add(fn)
                else:                          # 체크 OFF
                    st.session_state.manual_selected.discard(fn)
            
            # 1 체크박스  (value 파라미터 X, on_change 로만 동기화)
            with col_flag:
                st.checkbox(
                    label="",
                    key=chk_key,
                    value=fn in st.session_state.manual_selected,  # 최초 1회만 쓰임
                    on_change=toggle_file,
                    args=(fn, chk_key)
                )
            # 2 파일명 버튼 (기존 기능 그대로)
            with col_txt:
                if st.button(f"**{fn}** (score={row['score']:.2f})", key=btn_key, help="Click to toggle selection"):
                    if fn in st.session_state.manual_selected:
                        st.session_state.manual_selected.discard(fn)
                    else:
                        st.session_state.manual_selected.add(fn)
                    st.rerun()
                
                st.caption(f"`{fn}`")

    # ── 페이지 네비게이션 ─────────────────────
    st.divider()
    col_prev, col_page, col_next = st.columns([2, 7, 2])
    with col_prev:
        if st.session_state.current_page > 1 and st.button("<< Before"):
            st.session_state.current_page -= 1
            st.rerun()
    with col_page:
        st.markdown(
            f"**Current {st.session_state.current_page} / {total_pages}**",
            unsafe_allow_html=True
        )
    with col_next:
        if st.session_state.current_page < total_pages and st.button("Next >>"):
            st.session_state.current_page += 1
            st.rerun()

##검색결과 토글 리스트업하는 페이지 종료

else:
    st.info("검색 버튼을 눌러 결과를 확인하세요.")

st.markdown("""
        <style>
        /* 강조용 링크(버튼) */
        .emphasized-link{
            background-color:#2F14B8;          /* 스트림릿 기본 포인트 컬러 */
            color:#ffffff;                     /* 흰색 글자 */
            font-size:16px;                    /* 적당한 크기 */
            font-weight:600;                   /* semi-bold */
            padding:0.5rem 1rem;               /* 균형 잡힌 패딩 */
            border-radius:4px;                 /* 살짝 둥글게 */
            display:inline-block;              /* 내용 길이만큼만 차지 */
            text-align:center;                 /* 가운데 정렬 */
            text-decoration:none;              /* 밑줄 제거 */
            box-shadow:0 2px 4px rgba(0,0,0,.15); /* 은은한 그림자 */
            transition:
                background-color .2s ease,
                transform .1s ease;
        }

        /* 호버 시 살짝 강조 */
        .emphasized-link:hover{
            background-color:#d63c3c;          /* 조금 더 진한 빨간색 */
            transform:translateY(-2px);        /* 2px 위로 띄우기 */
        }

        /* 링크 내부의 a 태그에도 동일 스타일 적용 */
        .emphasized-link a{
            color:inherit;
            text-decoration:none;
        }
        </style>
        """, unsafe_allow_html=True)

#st.sidebar.markdown("---")
#st.sidebar.markdown(f"📝 [사후 설문지 열기]({SURVEY_URL})")
st.sidebar.markdown("---")
log_file = LOG_DIR / "phase_b" / f"{pid}.csv"
if log_file.exists():
    st.sidebar.download_button("Log download", log_file.read_bytes(), file_name=f"{pid}_phase_b.csv")
