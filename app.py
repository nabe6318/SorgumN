# app.py（N吸収量を26.6で上限 / GNDVI入力を-1〜+1に制限 / 可変施肥マップを先頭タブ）
import io
import streamlit as st
import numpy as np
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt

st.set_page_config(page_title="ソルガム可変施肥量計算（GNDVI→N吸収量）", layout="wide")
st.markdown(
    "<h3 style='text-align: center;'>🌾 ソルガム可変施肥量計算（GNDVI→窒素吸収量→可変施肥量）緑肥プロ O.Watanabe, Shinshu Univ.</h3>",
    unsafe_allow_html=True
)

# -----------------------------
# ユーティリティ
# -----------------------------
def make_df(r, c, like=None):
    df = pd.DataFrame(np.full((r, c), np.nan), dtype="float64")
    df.columns = [f"C{j+1}" for j in range(c)]
    df.index = [f"R{i+1}" for i in range(r)]
    if like is not None:
        rmin = min(r, like.shape[0])
        cmin = min(c, like.shape[1])
        df.iloc[:rmin, :cmin] = like.iloc[:rmin, :cmin].values
    return df

def read_csv_safely(file) -> pd.DataFrame:
    """CSVを読み込み、数値化し、[-1, 1] にクリップして返す。"""
    # file は一度読み込むと内部ポインタが進むので先頭へ
    try:
        file.seek(0)
    except Exception:
        pass
    # 文字コード・区切りの自動推定を含む頑健読込
    content = file.read()
    if isinstance(content, bytes):
        raw = content
    else:
        raw = content.encode("utf-8", errors="ignore")
    for enc in ["utf-8-sig", "utf-8", "cp932", "latin1"]:
        try:
            buf = io.StringIO(raw.decode(enc, errors="strict"))
            df = pd.read_csv(buf, sep=None, engine="python", on_bad_lines="skip", header=0, index_col=0)
            if df.empty:
                # ヘッダーなしの可能性
                buf = io.StringIO(raw.decode(enc, errors="strict"))
                df = pd.read_csv(buf, sep=None, engine="python", on_bad_lines="skip", header=None)
            break
        except Exception:
            df = None
            continue
    if df is None:
        raise ValueError("CSVの読み込みに失敗しました。")

    df = df.apply(pd.to_numeric, errors="coerce")
    df.index = [f"R{i+1}" for i in range(df.shape[0])]
    df.columns = [f"C{j+1}" for j in range(df.shape[1])]
    df = df.clip(lower=-1.0, upper=1.0)
    return df

def to_excel_bytes(sheets: dict) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, index=True, sheet_name=sheet_name)
    bio.seek(0)
    return bio.read()

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8")

# -----------------------------
# セッション初期化
# -----------------------------
if "rows" not in st.session_state:
    st.session_state.rows = 5
if "cols" not in st.session_state:
    st.session_state.cols = 5
if "gndvi_df" not in st.session_state:
    st.session_state.gndvi_df = make_df(st.session_state.rows, st.session_state.cols)

# -----------------------------
# サイドバー：設定 & CSV I/O
# -----------------------------
st.sidebar.header("設定 / Settings")

rows = st.sidebar.slider("行数（縦） / Rows", 1, 50, st.session_state.rows, 1)
cols = st.sidebar.slider("列数（横） / Columns", 1, 50, st.session_state.cols, 1)

if rows != st.session_state.rows or cols != st.session_state.cols:
    st.session_state.gndvi_df = make_df(rows, cols, like=st.session_state.gndvi_df)
    st.session_state.rows, st.session_state.cols = rows, cols

baseline_N = st.sidebar.number_input(
    "基準施肥量（kg/10a） / Baseline fertilizer",
    min_value=0.0, max_value=100.0, value=7.0, step=0.1, format="%.2f"
)
clip_negative = st.sidebar.checkbox("可変施肥量を 0 未満にしない（0で下限）", value=False)

st.sidebar.divider()
st.sidebar.subheader("CSV 入出力 / CSV I/O")

template_df = make_df(st.session_state.rows, st.session_state.cols)
st.sidebar.download_button(
    "📄 空テンプレCSVダウンロード",
    data=df_to_csv_bytes(template_df),
    file_name="GNDVI_template.csv",
    mime="text/csv",
)
st.sidebar.download_button(
    "💾 現在の入力をCSV保存",
    data=df_to_csv_bytes(st.session_state.gndvi_df),
    file_name="GNDVI_current.csv",
    mime="text/csv",
)
uploaded = st.sidebar.file_uploader("CSV 読み込み（置き換え）", type=["csv"])
if uploaded is not None:
    try:
        df_in = read_csv_safely(uploaded)
        st.session_state.gndvi_df = df_in
        st.session_state.rows, st.session_state.cols = df_in.shape
        st.success(f"CSV 読み込み成功（{df_in.shape[0]} 行 × {df_in.shape[1]} 列）")
        st.toast("CSVの値を[-1, 1]にクリップしました。", icon="ℹ️")
    except Exception as e:
        st.error(f"CSV 読み込み失敗: {e}")

# --- リセットオプション ---
st.sidebar.divider()
st.sidebar.subheader("シート操作 / Sheet ops")

reset_mode = st.sidebar.selectbox(
    "リセット方法を選択",
    ["空欄（NaN）にする", "0で埋める", "一定値で埋める"],
    index=0
)
const_val = st.sidebar.number_input(
    "一定値 (Reset value)",
    value=0.0, step=0.1, format="%.3f",
    disabled=(reset_mode != "一定値で埋める")
)

if st.sidebar.button("🔄 リセットを実行", use_container_width=True):
    r, c = st.session_state.rows, st.session_state.cols
    df = make_df(r, c)
    if reset_mode == "0で埋める":
        df[:] = 0.0
    elif reset_mode == "一定値で埋める":
        df[:] = const_val
    st.session_state.gndvi_df = df
    st.toast(f"植生指数シートを『{reset_mode}』でリセットしました。")

st.sidebar.caption("計算式: N吸収量 = min( 0.2567 × exp(5.125 × GNDVI), 26.6 )")

# -----------------------------
# ① 入力シート（-1〜+1に制限）
# -----------------------------
st.subheader("① 植生指数シート（GNDVI を入力：-1.0〜+1.0）")

# 列ごとに入力制限を付ける（全列共通）
col_cfg = {
    col: st.column_config.NumberColumn(
        label=col, min_value=-1.0, max_value=1.0, step=0.001, format="%.3f"
    )
    for col in st.session_state.gndvi_df.columns
}

gndvi_df = st.data_editor(
    st.session_state.gndvi_df.astype("float64"),
    num_rows="fixed",
    use_container_width=True,
    key="gndvi_editor",
    column_config=col_cfg
)
# 安全側：最終的に数値化＋クリップ
gndvi_df = gndvi_df.apply(pd.to_numeric, errors="coerce").clip(lower=-1.0, upper=1.0)
st.session_state.gndvi_df = gndvi_df

# -----------------------------
# ② 計算（★N吸収量を26.6で上限）
# -----------------------------
def safe_exp(x):
    with np.errstate(over="ignore", invalid="ignore"):
        return np.exp(x)

# もとの計算
n_uptake_raw = 0.2567 * safe_exp(5.125 * gndvi_df.astype(float))  # 窒素吸収量（kg/10a）
# 上限クリップ
n_uptake = n_uptake_raw.clip(upper=26.6)

# 上限適用の通知（任意）
if (n_uptake != n_uptake_raw).to_numpy().any():
    st.toast("窒素吸収量シートの上限 26.6 kg/10a を超えたセルを 26.6 に丸めました。", icon="⚠️")

# 下流計算
n_sorghum = n_uptake * 0.3
variable_N = baseline_N - n_sorghum
if clip_negative:
    variable_N = variable_N.clip(lower=0)

for df in (n_uptake, n_sorghum, variable_N):
    df.index = gndvi_df.index
    df.columns = gndvi_df.columns

# -----------------------------
# マップ表示設定（サイドバー）
# -----------------------------
st.sidebar.divider()
st.sidebar.subheader("マップ表示設定 / Map settings")
use_fixed_scale = st.sidebar.checkbox("色スケールを固定する（vmin/vmax 指定）", value=False)
decimals = st.sidebar.number_input("セル数値の小数桁", min_value=0, max_value=6, value=1, step=1)
if use_fixed_scale:
    default_min = float(np.nanmin(variable_N.values)) if np.isfinite(np.nanmin(variable_N.values)) else 0.0
    default_max = float(np.nanmax(variable_N.values)) if np.isfinite(np.nanmax(variable_N.values)) else 1.0
    vmin = st.sidebar.number_input("vmin（最小）", value=round(default_min, 2))
    vmax = st.sidebar.number_input("vmax（最大）", value=round(default_max, 2))
else:
    vmin = None
    vmax = None

# -----------------------------
# タブ（★先頭：可変施肥マップ）
# -----------------------------
tab_map, tab_var, tab2, tab3, tab1 = st.tabs([
    "可変施肥マップ（色分け＋数値）",  # 先頭
    "可変施肥量シート",
    "窒素吸収量シート",
    "ソルガム由来の窒素量シート",
    "植生指数シート",
])

# --- 可変施肥マップ（先頭タブ） ---
with tab_map:
    st.caption("行×列のグリッドをヒートマップ表示し、セルに可変施肥量（kg/10a）を重ねて表示します。")
    data = variable_N.values.astype(float)
    masked = np.ma.masked_invalid(data)

    _vmin = np.nanmin(data) if vmin is None else vmin
    _vmax = np.nanmax(data) if vmax is None else vmax
    if not np.isfinite(_vmin): _vmin = 0.0
    if not np.isfinite(_vmax): _vmax = 1.0
    if _vmin == _vmax:
        _vmax = _vmin + 1.0

    fig, ax = plt.subplots(figsize=(max(5, data.shape[1]*0.7), max(4, data.shape[0]*0.7)))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#e0e0e0")  # NaNは薄灰

    im = ax.imshow(masked, cmap=cmap, vmin=_vmin, vmax=_vmax)
    ax.set_xticks(np.arange(data.shape[1]) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1, alpha=0.7)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    rng = _vmax - _vmin
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                continue
            norm = (val - _vmin) / rng if rng > 0 else 0.5
            text_color = "black" if norm > 0.6 else "white"
            ax.text(j, i, f"{val:.{decimals}f}", ha="center", va="center", fontsize=10, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("可変施肥量 (kg/10a)")
    st.pyplot(fig, use_container_width=True)

# --- テーブル ---
with tab_var:
    st.dataframe(variable_N.round(3), use_container_width=True)
    st.caption(f"基準施肥量 = {baseline_N:.2f} kg/10a")

with tab2:
    st.dataframe(n_uptake.round(3), use_container_width=True)

with tab3:
    st.dataframe(n_sorghum.round(3), use_container_width=True)

with tab1:
    st.dataframe(gndvi_df, use_container_width=True)

# -----------------------------
# Excel ダウンロード
# -----------------------------
excel_bytes = to_excel_bytes({
    "植生指数シート": gndvi_df,
    "窒素吸収量シート": n_uptake.round(6),
    "ソルガム由来の窒素量シート": n_sorghum.round(6),
    "可変施肥量シート": variable_N.round(6),
})
st.download_button(
    label="📥 Excel ダウンロード（4シート）",
    data=excel_bytes,
    file_name="variable_fertilizer.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
