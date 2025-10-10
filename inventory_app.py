import streamlit as st
import pandas as pd
import re
import fitz  # PyMuPDF
from collections import defaultdict
from io import BytesIO
from datetime import datetime
import os

# ---------------- UI ----------------
st.set_page_config(page_title="NailVesta 库存系统", layout="centered")
st.title("ColorFour Inventory 系统（fitz 解析 + bundle 修复 + Mystery 说明）")

# 上传文件（PDF 支持多选）
pdf_files = st.file_uploader("上传 Picking List PDF（可多选）", type=["pdf"], accept_multiple_files=True)
csv_file = st.file_uploader("上传库存表 CSV", type=["csv"])

# 选择要参与统计的 PDF（默认全选）
selected_pdfs = []
if pdf_files:
    selected_names = st.multiselect(
        "选择要参与统计的 Picking List PDF",
        options=[f.name for f in pdf_files],
        default=[f.name for f in pdf_files]
    )
    selected_pdfs = [f for f in pdf_files if f.name in selected_names]

# —— 按钮触发：是否有达人换货 ——
if "show_exchange" not in st.session_state:
    st.session_state.show_exchange = False
if st.button("有达人换货吗？"):
    st.session_state.show_exchange = True

exchange_df = None
if st.session_state.show_exchange:
    st.info("请上传换货记录文件（CSV / Excel），将执行：原款 +1、换货 -1（每行各一件）")
    exchange_file = st.file_uploader("上传换货记录", type=["csv", "xlsx"])
    if exchange_file:
        if exchange_file.name.endswith(".csv"):
            exchange_df = pd.read_csv(exchange_file)
        else:
            exchange_df = pd.read_excel(exchange_file)
        st.success("换货表已上传")

# ---------- 规则与小工具 ----------
# 允许 1–4 件的组合，单件可为标准 6 位 SKU（如 ABC123）或 Mystery 'NM001'，尾随尺码 -S/M/L
SKU_BUNDLE = re.compile(r'((?:[A-Z]{3}\d{3}|NM001){1,4}-[SML])', re.DOTALL)
# SKU 右侧 1–3 位数量
QTY_AFTER = re.compile(r'\b([1-9]\d{0,2})\b')
# “Item quantity”
ITEM_QTY_RE = re.compile(r"Item\s+quantity[:：]?\s*(\d+)", re.I)

def normalize_text(t: str) -> str:
    t = t.replace("\u00ad","").replace("\u200b","").replace("\u00a0"," ")
    t = t.replace("–","-").replace("—","-")
    return t

def fix_orphan_digit_before_size(txt: str) -> str:
    """
    修复形如：NPJ011NPX01\n5-M → NPJ011NPX015-M 的换行折断
    """
    pattern = re.compile(
        r'(?P<prefix>(?:[A-Z]{3}\d{3}|NM001){0,3}(?:[A-Z]{3}\d{2}|NM001))\s*[\r\n]+\s*(?P<d>\d)\s*-\s*(?P<size>[SML])'
    )
    def _join(m):
        # 若前缀最后是 NM001，则不追加 d（NM001 长度为 5）；否则拼接成 6 位
        prefix = m.group('prefix')
        d = m.group('d')
        size = m.group('size')
        if prefix.endswith('NM001'):
            return f"{prefix}-{size}"
        return f"{prefix}{d}-{size}"
    prev = None
    cur = txt
    while prev != cur:
        prev = cur
        cur = pattern.sub(_join, cur)
    return cur

def parse_code_parts(code: str):
    """
    将无连字符的主体按以下顺序切块：
      - 优先匹配前缀 'NM001'
      - 否则匹配标准 6 位块 [A-Z]{3}\d{3}
    允许任意顺序组合，如：
      NM001NPJ011 / NPX015NM001 / NM001NM001 / NPJ011NPX015 等
    全部成功返回 part 列表，否则返回 None
    """
    parts = []
    i = 0
    n = len(code)
    while i < n:
        if code.startswith('NM001', i):
            parts.append('NM001')
            i += 5
            continue
        seg = code[i:i+6]
        if re.fullmatch(r'[A-Z]{3}\d{3}', seg):
            parts.append(seg)
            i += 6
            continue
        # 无法继续匹配
        return None
    # 1–4 件限制
    if 1 <= len(parts) <= 4:
        return parts
    return None

def expand_bundle(counter: dict, sku_with_size: str, qty: int):
    """
    将 1–4 件组合拆成独立 SKU 计数；返回：
      extra_units: 因组合导致的“额外件数”（(件数-1)*qty）
      mystery_units: 其中属于 NM001 的件数（parts 中出现 NM001 的次数 * qty）
    """
    s = re.sub(r'\s+', '', sku_with_size)
    if '-' not in s:
        counter[s] += qty
        return 0, 0
    code, size = s.split('-', 1)
    parts = parse_code_parts(code)
    if parts:
        mystery_units = 0
        for p in parts:
            key = f"{p}-{size}"
            counter[key] += qty
            if p == 'NM001':
                mystery_units += qty
        extra = (len(parts) - 1) * qty
        return extra, mystery_units
    # 回退：非标准就按原样记
    counter[s] += qty
    # 若恰是 NM001-Size，也记录为 Mystery
    mystery_units = qty if code == 'NM001' else 0
    return 0, mystery_units

def parse_pdf_with_fitz(file_bytes: bytes):
    """
    返回：
      expected_total_raw: PDF 里 “Item quantity”
      sku_counts_single: dict，键为 Seller SKU（含尺码），值为累计数量
      bundle_extra_units: 因 bundle（多件被 PDF 记为 1 件）产生的“额外件数”
      mystery_units: 解析中识别到的 NM001 件数（计数含数量）
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages_text = []
    for p in doc:
        pages_text.append(p.get_text("text"))
    text_raw = "\n".join(pages_text)
    text = normalize_text(text_raw)

    # 对账用的 Item quantity
    m_total = ITEM_QTY_RE.search(text)
    expected_total_raw = int(m_total.group(1)) if m_total else 0

    # 修复换行
    text_fixed = fix_orphan_digit_before_size(text)

    # 匹配所有 SKU（允许跨行）
    sku_counts = defaultdict(int)
    bundle_extra_units = 0
    mystery_units = 0
    for m in SKU_BUNDLE.finditer(text_fixed):
        sku_raw = re.sub(r'\s+', '', m.group(1))
        # 在 token 之后的一小段里找 Qty
        after = text_fixed[m.end(): m.end()+50]
        mq = QTY_AFTER.search(after)
        qty = int(mq.group(1)) if mq else 1
        extra, myst = expand_bundle(sku_counts, sku_raw, qty)
        bundle_extra_units += extra
        mystery_units += myst

    return expected_total_raw, sku_counts, bundle_extra_units, mystery_units

# ---------- 主流程 ----------
if selected_pdfs and csv_file:
    st.success("文件上传成功，开始处理...")

    # 读取库存 CSV
    stock_df = pd.read_csv(csv_file)
    stock_df.columns = [col.strip() for col in stock_df.columns]
    stock_col = [col for col in stock_df.columns if re.match(r"\d{2}/\d{2}", col)]
    if not stock_col:
        st.error("未找到库存日期列（如 '06/03'）")
        st.stop()
    stock_date_col = stock_col[0]

    # —— 每个 PDF：解析 ——
    pdf_item_list = []
    pdf_sku_counts = {}
    per_pdf_expected = []      # 原始 expected
    per_pdf_extra = []         # bundle 额外件数
    per_pdf_actual = []        # 实际提取（含 Mystery）
    per_pdf_mystery = []       # Mystery 件数

    for pf in selected_pdfs:
        file_bytes = pf.read()
        expected_total_raw, sku_counts_single, bundle_extra_units, mystery_units = parse_pdf_with_fitz(file_bytes)
        pdf_sku_counts[pf.name] = sku_counts_single

        actual_total = sum(sku_counts_single.values())                    # 含 Mystery
        expected_adjusted = expected_total_raw + bundle_extra_units       # bundle 调整
        actual_for_check = actual_total - mystery_units                   # 扣除 Mystery 后用于对账的件数
        expected_for_check = expected_adjusted - mystery_units            # 对应扣除 Mystery 的期望

        # 状态判定（优先：原始一致 → bundle 调整后 → 扣除 Mystery 后）
        if expected_total_raw == 0:
            status = "未识别到 Item quantity"
        elif actual_total == expected_total_raw:
            status = "一致"
        elif actual_total == expected_adjusted:
            status = f"与PDF标注不一致，但考虑 bundle 后相符（差 {actual_total - expected_total_raw}）"
        elif actual_for_check == expected_for_check:
            status = (
                f"与PDF标注不一致，但考虑 bundle 与 Mystery（NM001）后相符"
                f"（差 {actual_total - expected_total_raw}，其中 Mystery {mystery_units} 件）"
            )
        else:
            diff = actual_total - expected_total_raw
            status = f"不一致（差 {diff}；bundle 影响 {bundle_extra_units}；Mystery {mystery_units} 件）"

        pdf_item_list.append({
            "PDF文件": pf.name,
            "Item quantity（原始）": expected_total_raw,
            "bundle 调整(+额外件数)": bundle_extra_units,
            "Mystery(NM001) 件数": mystery_units,
            "调整后应为（含 bundle）": expected_adjusted,
            "对账用实际（扣除 Mystery）": actual_for_check,
            "提取出货数量（含 Mystery）": actual_total,
            "状态": status
        })

        per_pdf_expected.append(expected_total_raw)
        per_pdf_extra.append(bundle_extra_units)
        per_pdf_actual.append(actual_total)
        per_pdf_mystery.append(mystery_units)

    # —— 显示 PDF 对账表 + 合计行 ——
    st.subheader("各 PDF 的 Item quantity 对账表（含 bundle 与 Mystery）")
    pdf_df = pd.DataFrame(pdf_item_list)

    total_expected_raw = sum(per_pdf_expected) if per_pdf_expected else 0
    total_bundle_extra = sum(per_pdf_extra) if per_pdf_extra else 0
    total_expected_adjusted = total_expected_raw + total_bundle_extra
    total_mystery = sum(per_pdf_mystery) if per_pdf_mystery else 0
    total_actual = sum(per_pdf_actual) if per_pdf_actual else 0
    total_actual_for_check = total_actual - total_mystery
    total_expected_for_check = total_expected_adjusted - total_mystery

    if not pdf_df.empty:
        # 合计状态
        if total_actual == total_expected_raw:
            total_status = "一致"
        elif total_actual == total_expected_adjusted:
            total_status = f"与PDF标注不一致，但考虑 bundle 后相符（差 {total_actual - total_expected_raw}）"
        elif total_actual_for_check == total_expected_for_check:
            total_status = (
                f"与PDF标注不一致，但考虑 bundle 与 Mystery（NM001）后相符"
                f"（差 {total_actual - total_expected_raw}，其中 Mystery {total_mystery} 件）"
            )
        else:
            diff = total_actual - total_expected_raw
            total_status = f"不一致（差 {diff}；bundle 影响 {total_bundle_extra}；Mystery {total_mystery} 件）"

        total_row = {
            "PDF文件": "合计",
            "Item quantity（原始）": total_expected_raw,
            "bundle 调整(+额外件数)": total_bundle_extra,
            "Mystery(NM001) 件数": total_mystery,
            "调整后应为（含 bundle）": total_expected_adjusted,
            "对账用实际（扣除 Mystery）": total_actual_for_check,
            "提取出货数量（含 Mystery）": total_actual,
            "状态": total_status
        }
        pdf_df = pd.concat([pdf_df, pd.DataFrame([total_row])], ignore_index=True)

    st.dataframe(pdf_df, use_container_width=True)

    # —— 汇总所有 PDF 的 SKU ——
    sku_counts_all = defaultdict(int)
    for counts in pdf_sku_counts.values():
        for sku, qty in counts.items():
            sku_counts_all[sku] += qty

    # —— 换货处理：提取替换 + 库存调整（每行原款 +1、换货 -1） ——
    if exchange_df is not None:
        if "原款式" in exchange_df.columns and "换货款式" in exchange_df.columns:
            for _, row in exchange_df.iterrows():
                original_sku = str(row["原款式"]).strip()
                new_sku = str(row["换货款式"]).strip()
                # 替换提取数量（原款 → 换货）
                if sku_counts_all.get(original_sku):
                    qty = sku_counts_all.pop(original_sku)
                    sku_counts_all[new_sku] += qty
                # 直接修改库存（对应日期列）：原款 +1、换货 -1
                stock_df.loc[stock_df["SKU编码"] == original_sku, stock_date_col] += 1
                stock_df.loc[stock_df["SKU编码"] == new_sku, stock_date_col] -= 1
            st.success("换货处理完成：已替换提取数量并调整库存（原款 +1 / 换货 -1）")
        else:
            st.warning("换货表中必须包含“原款式”和“换货款式”两列")

    # —— 合并库存数据 ——
    stock_df["Sold"] = stock_df["SKU编码"].map(sku_counts_all).fillna(0).astype(int)
    stock_df["New Stock"] = stock_df[stock_date_col] - stock_df["Sold"]

    summary_df = stock_df[["SKU编码", stock_date_col, "Sold", "New Stock"]].copy()
    summary_df.columns = ["SKU", "Old Stock", "Sold Qty", "New Stock"]
    summary_df.index += 1
    summary_df.loc["合计"] = [
        "—",
        summary_df["Old Stock"].sum(),
        summary_df["Sold Qty"].sum(),
        summary_df["New Stock"].sum()
    ]

    # 展示库存更新结果
    st.subheader("库存更新结果")
    st.dataframe(summary_df, use_container_width=True)

    # 总对账提示（优先 bundle，再考虑 Mystery）
    total_sold = summary_df.loc["合计", "Sold Qty"]
    if total_expected_raw > 0:
        if total_sold == total_expected_raw:
            st.success(f"提取成功：共 {total_sold} 件，与 PDF 标注汇总一致")
        elif total_sold == total_expected_adjusted:
            st.success(f"提取成功：共 {total_sold} 件。与 PDF 原始汇总不一致，但考虑 bundle 后相符（差 {total_sold - total_expected_raw}）。")
        elif (total_sold - total_mystery) == (total_expected_adjusted - total_mystery):
            st.success(
                f"提取成功：共 {total_sold} 件。与 PDF 原始汇总不一致，但考虑 bundle 与 Mystery（NM001 {total_mystery} 件）后相符"
                f"（差 {total_sold - total_expected_raw}）。"
            )
        else:
            st.error(
                f"提取数量 {total_sold} 与 PDF 标注汇总不一致；"
                f"原始: {total_expected_raw}，bundle 调整后: {total_expected_adjusted}，Mystery（NM001）: {total_mystery} 件。"
            )

    # 一键复制 New Stock
    st.subheader("一键复制 New Stock")
    new_stock_text = "\n".join(summary_df.iloc[:-1]["New Stock"].astype(str).tolist())
    st.code(new_stock_text, language="text")

    # 下载 Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        summary_df.to_excel(writer, index_label="序号")
    st.download_button(
        label="下载库存更新表 Excel",
        data=output.getvalue(),
        file_name="库存更新结果.xlsx"
    )

    # 上传历史记录
    history_file = "upload_history.csv"
    new_record = {
        "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "PDF文件": "; ".join([f.name for f in selected_pdfs]),
        "库存文件": csv_file.name,
        "PDF标注数量（原始）": total_expected_raw if total_expected_raw else "",
        "bundle 额外件数": total_bundle_extra if total_bundle_extra else "",
        "Mystery（NM001）件数": total_mystery if total_mystery else "",
        "PDF标注数量（调整后）": total_expected_adjusted if total_expected_adjusted else "",
        "提取出货数量（含 Mystery）": total_sold
    }
    if os.path.exists(history_file):
        history_df = pd.read_csv(history_file)
        history_df = pd.concat([history_df, pd.DataFrame([new_record])], ignore_index=True)
    else:
        history_df = pd.DataFrame([new_record])
    history_df.to_csv(history_file, index=False)

    st.subheader("上传历史记录")
    st.dataframe(history_df, use_container_width=True)

else:
    st.info("请上传一个或多个 Picking List PDF 和库存 CSV 以开始处理。")
