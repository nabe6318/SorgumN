import io
import streamlit as st
import numpy as np
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt

st.set_page_config(page_title="可変施肥量計算（GNDVI→N吸収量）", layout="wide")
st.markdown(
    "<h3 style='text-align: center;'>🌿 可変施肥量計算（GNDVI→窒素吸収量→可変施肥量）緑肥プロ O.Watanabe, Shinshu Univ.</h3>",
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
    try:
        file.seek(0)
    except Exception:
        pass

    content = file.read()
    if isinstance(content, bytes):
        raw = content
    else:
        raw = content.encode("utf-8", errors="ignore")

    df = None
    for enc in ["utf-8-sig", "utf-8", "cp932", "latin1"]:
        try:
            buf = io.StringIO(raw.decode(enc, errors="strict"))
            df = pd.read_csv(
                buf, sep=None, engine="python", on_bad_lines="skip",
                header=0, index_col=0
            )
            if df.empty:
                buf = io.StringIO(raw.decode(enc, errors="strict"))
                df = pd.read_csv(
                    buf, sep=None, engine="python", on_bad_lines="skip",
                    header=None
                )
            break
        except Exception:
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
if "gndvi_df" not in st.session_state:
    st.session_state.gndvi_df = make_df(5, 5)

# -----------------------------
# サイドバー：設定
# -----------------------------
st.sidebar.header("設定 / Settings")

# --- 緑肥の選択と計算式の定義 ---
crop_type = st.sidebar.selectbox(
    "緑肥の種類を選択",
    ["ソルガム", "クロタラリア"]
)

# 【修正箇所】ソルガムの計算式パラメータを更新
if crop_type == "ソルガム":
    coeff_a = 0.2567
    coeff_b = 5.125
else:  # クロタラリア
    coeff_a = 0.0711
    coeff_b = 7.1541

baseline_N = st.sidebar.number_input(
    "基準施肥量（kg/10a）",
    min_value=0.0, max_value=100.0, value=8.0, step=0.1, format="%.2f"
)

efficiency = st.sidebar.number_input(
    "肥効率近似値（緑肥由来窒素の利用率）",
    min_value=0.0, max_value=1.0, value=0.3, step=0.01, format="%.2f"
)

clip_negative = st.sidebar.checkbox("可変施肥量を 0 未満にしない", value=False)

st.sidebar.divider()
st.sidebar.subheader("CSV 入出力")

template_df = make_df(
    st.session_state.gndvi_df.shape[0],
    st.session_state.gndvi_df.shape[1]
)
st.sidebar.download_button(
    "📄 空テンプレCSVダウンロード",
    data=df_to_csv_bytes(template_df),
    file_name="GNDVI_template.csv",
    mime="text/csv",
)

uploaded = st.sidebar.file_uploader("CSV 読み込み", type=["csv"])
if uploaded is not None:
    try:
        df_in = read_csv_safely(uploaded)
        st.session_state.gndvi_df = df_in
        st.success("CSV 読み込み成功")
    except Exception as e:
        st.error(f"CSV 読み込み失敗: {e}")

st.sidebar.download_button(
    "💾 現在の入力をCSV保存",
    data=df_to_csv_bytes(st.session_state.gndvi_df),
    file_name="GNDVI_current.csv",
    mime="text/csv",
)

st.sidebar.divider()
formula_text = f"{coeff_a} × exp({coeff_b} × GNDVI)"
st.sidebar.caption(
    f"現在の計算式 ({crop_type}):\n\n"
    f"N吸収量 = min( {formula_text}, 26.6 )\n\n"
    f"可変施肥量 = 基準施肥量 - (N吸収量 × 肥効率近似値 {efficiency})"
)

# -----------------------------
# ① 入力シート
# -----------------------------
st.subheader("① 植生指数シート（GNDVI を入力）")

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
gndvi_df = gndvi_df.apply(pd.to_numeric, errors="coerce").clip(lower=-1.0, upper=1.0)
st.session_state.gndvi_df = gndvi_df

# -----------------------------
# ② 計算
# -----------------------------
def safe_exp(x):
    with np.errstate(over="ignore", invalid="ignore"):
        return np.exp(x)

# 窒素吸収量 (更新された係数を使用)
n_uptake_raw = coeff_a * safe_exp(coeff_b * gndvi_df.astype(float))
n_uptake = n_uptake_raw.clip(upper=26.6)

if (n_uptake != n_uptake_raw).to_numpy().any():
    st.toast("窒素吸収量の上限 26.6 kg/10a を超えたセルを丸めました。", icon="⚠️")

# 緑肥由来の窒素量
n_green_manure = n_uptake * efficiency

# 可変施肥量
variable_N = baseline_N - n_green_manure
if clip_negative:
    variable_N = variable_N.clip(lower=0)

for df in (n_uptake, n_green_manure, variable_N):
    df.index = gndvi_df.index
    df.columns = gndvi_df.columns

# -----------------------------
# マップ表示
# -----------------------------
st.sidebar.divider()
st.sidebar.subheader("表示設定")
use_fixed_scale = st.sidebar.checkbox("色スケールを固定する", value=False)
decimals = st.sidebar.number_input("小数桁", min_value=0, max_value=6, value=1, step=1)
if use_fixed_scale:
    vmin_input = st.sidebar.number_input("vmin", value=0.0)
    vmax_input = st.sidebar.number_input("vmax", value=10.0)
else:
    vmin_input, vmax_input = None, None

# -----------------------------
# タブ
# -----------------------------
tab_map, tab_var, tab2, tab3, tab1 = st.tabs([
    "可変施肥マップ",
    "可変施肥量シート",
    "窒素吸収量シート",
    "緑肥由来の窒素量シート",
    "植生指数シート",
])

with tab_map:
    st.caption(f"対象緑肥: {crop_type} / 計算式: {formula_text}")
    data = variable_N.values.astype(float)
    masked = np.ma.masked_invalid(data)

    _vmin = np.nanmin(data) if vmin_input is None else vmin_input
    _vmax = np.nanmax(data) if vmax_input is None else vmax_input
    if not np.isfinite(_vmin):
        _vmin = 0.0
    if not np.isfinite(_vmax):
        _vmax = 1.0

    fig, ax = plt.subplots(
        figsize=(max(5, data.shape[1] * 0.7), max(4, data.shape[0] * 0.7))
    )
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#e0e0e0")
    im = ax.imshow(masked, cmap=cmap, vmin=_vmin, vmax=_vmax)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if not np.isnan(val):
                ratio = (val - _vmin) / (_vmax - _vmin + 1e-6)
                ax.text(
                    j, i, f"{val:.{decimals}f}",
                    ha="center", va="center",
                    color="black" if ratio > 0.6 else "white"
                )

    plt.colorbar(im, ax=ax, label="施肥量 (kg/10a)")
    st.pyplot(fig, use_container_width=True)

with tab_var:
    st.dataframe(variable_N.round(3), use_container_width=True)

with tab2:
    st.dataframe(n_uptake.round(3), use_container_width=True)

with tab3:
    st.caption(f"計算: 窒素吸収量 × 肥効率近似値({efficiency})")
    st.dataframe(n_green_manure.round(3), use_container_width=True)

with tab1:
    st.dataframe(gndvi_df, use_container_width=True)

# -----------------------------
# Excel ダウンロード
# -----------------------------
excel_bytes = to_excel_bytes({
    "植生指数シート": gndvi_df,
    "窒素吸収量シート": n_uptake.round(6),
    "緑肥由来の窒素量シート": n_green_manure.round(6),
    "可変施肥量シート": variable_N.round(6),
})

st.download_button(
    label="📥 Excel ダウンロード",
    data=excel_bytes,
    file_name=f"variable_fertilizer_{crop_type}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
