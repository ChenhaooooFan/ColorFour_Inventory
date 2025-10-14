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
st.title("ColorFour Inventory 系统（fitz 解析 + bundle 修复 + Mystery 抵扣期望）")

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

exchange_df = None          # 旧：成对原款/换货（按行做“全量替换 + 库存 ±1”）
creator_swap_df = None      # 新：达人换货统计表（每行代表 1 件的原→发货调整）

if st.session_state.show_exchange:
    st.info("可上传两类换货数据：\n1) 换货记录（CSV/XLSX）：原款 +1、换货 -1（每行各一件，且会尝试把该 SKU 的“全部提取量”转移到新款）\n2) 达人换货统计表（CSV/XLSX）：每行代表发货了 1 件，逐行按 1 件进行“原款 -1 → 发货 +1”并同步库存 +1/-1。")
    # 旧换货表
    exchange_file = st.file_uploader("上传换货记录（旧格式：包含列“原款式”“换货款式”）", type=["csv", "xlsx"], key="old_exchange")
    if exchange_file:
        if exchange_file.name.endswith(".csv"):
            exchange_df = pd.read_csv(exchange_file)
        else:
            exchange_df = pd.read_excel(exchange_file)
        st.success("换货记录（旧格式）已上传")

    # 新达人换货统计表
    creator_swap_file = st.file_uploader("上传达人换货统计表（新格式：每行=1件，包含列“原款式 SKU”“发货款式SKU”）", type=["csv", "xlsx"], key="creator_swap")
    if creator_swap_file:
        if creator_swap_file.name.endswith(".csv"):
            creator_swap_df = pd.read_csv(creator_swap_file)
        else:
            creator_swap_df = pd.read_excel(creator_swap_file)
        st.success("达人换货统计表（逐行一件）已上传")

# ---------- 规则与小工具 ----------
# 1) 含尺码的组合（bundle）：允许 1–4 件；部件可为标准 6 位或 NM001；结尾必须 -S/M/L
SKU_BUNDLE_WITH_SIZE = re.compile(r'((?:[A-Z]{3}\d{3}|NM001){1,4}-(?:S|M|L))', re.DOTALL)

# 2) 仅允许 NM001 无尺码（你的单独栏场景）
SKU_SOLO_NM_ONLY = re.compile(r'\bNM001\b')

# 数量：1–3 位整数，且后面不能紧跟字母（避免把 “3D” 的 3 当数量）
QTY_AFTER_SHORT = re.compile(r'\b([1-9]\d{0,2})(?![A-Za-z])\b')

# “Item quantity”
ITEM_QTY_RE = re.compile(r"Item\s+quantity[:：]?\s*(\d+)", re.I)

def normalize_text(t: str) -> str:
    t = t.replace("\u00ad","").replace("\u200b","").replace("\u00a0"," ")
    t = t.replace("–","-").replace("—","-")
    return t

def fix_orphan_digit_before_size(txt: str) -> str:
    """
    修复形如：NPJ011NPX01\n5-M → NPJ011NPX015-M 的换行折断
    同时兼容 NM001 出现在组合中
    """
    pattern = re.compile(
        r'(?P<prefix>(?:[A-Z]{3}\d{3}|NM001){0,3}(?:[A-Z]{3}\d{2}|NM001))\s*[\r\n]+\s*(?P<d>\d)\s*-\s*(?P<size>[SML])'
    )
    def _join(m):
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
    """将主体按 NM001 或 6位块切分，限制 1–4 件。全部成功返回列表，否则 None。"""
    parts, i, n = [], 0, len(code)
    while i < n:
        if code.startswith('NM001', i):
            parts.append('NM001'); i += 5; continue
        seg = code[i:i+6]
        if re.fullmatch(r'[A-Z]{3}\d{3}', seg):
            parts.append(seg); i += 6; continue
        return None
    return parts if 1 <= len(parts) <= 4 else None

def expand_bundle(counter: dict, sku_with_size: str, qty: int):
    """
    拆组合：
    返回 extra_units=(件数-1)*qty；mystery_units=NM001 次数 * qty
    """
    s = re.sub(r'\s+', '', sku_with_size)
    if '-' not in s:
        counter[s] += qty
        return 0, (qty if s == 'NM001' else 0)
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
    counter[s] += qty
    return 0, (qty if code == 'NM001' else 0)

def parse_pdf_with_fitz(file_bytes: bytes):
    """
    返回：
      expected_total_raw: PDF 里 “Item quantity”
      sku_counts_single: dict（含尺码/无尺码）
      bundle_extra_units: 因 bundle 产生的额外件数
      mystery_units: 识别到的 NM001 件数（按数量）
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages_text = [p.get_text("text") for p in doc]
    text_raw = "\n".join(pages_text)
    text = normalize_text(text_raw)

    m_total = ITEM_QTY_RE.search(text)
    expected_total_raw = int(m_total.group(1)) if m_total else 0

    text_fixed = fix_orphan_digit_before_size(text)

    sku_counts = defaultdict(int)
    bundle_extra_units = 0
    mystery_units = 0

    # —— 第一轮：带尺码的组合（含单件 -S/M/L）——
    for m in SKU_BUNDLE_WITH_SIZE.finditer(text_fixed):
        token = re.sub(r'\s+', '', m.group(1))
        after = text_fixed[m.end(): m.end()+60]
        mq = QTY_AFTER_SHORT.search(after)
        qty = int(mq.group(1)) if mq else 1
        extra, myst = expand_bundle(sku_counts, token, qty)
        bundle_extra_units += extra
        mystery_units += myst

    # —— 第二轮：仅针对“无尺码 NM001”——
    for m in SKU_SOLO_NM_ONLY.finditer(text_fixed):
        # 若后面紧跟 '-S/M/L'，说明是 NM001-M 等，已在第一轮计入 → 跳过
        next_chunk = text_fixed[m.end(): m.end()+3]
        if '-' in next_chunk:
            continue
        after = text_fixed[m.end(): m.end()+80]
        mq = QTY_AFTER_SHORT.search(after)
        qty = int(mq.group(1)) if mq else 1
        sku_counts['NM001'] += qty
        mystery_units += qty

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
    per_pdf_mystery = []       # Mystery 件数（用于抵扣期望）

    for pf in selected_pdfs:
        file_bytes = pf.read()
        expected_total_raw, sku_counts_single, bundle_extra_units, mystery_units = parse_pdf_with_fitz(file_bytes)
        pdf_sku_counts[pf.name] = sku_counts_single

        actual_total = sum(sku_counts_single.values())            # 含 Mystery
        expected_bundle = expected_total_raw + bundle_extra_units # 仅考虑 bundle
        expected_final = expected_bundle - mystery_units          # ⚠️ bundle − Mystery（抵扣）为最终期望

        # 状态判定：原始 → bundle → bundle−Mystery
        if expected_total_raw == 0:
            status = "未识别到 Item quantity"
        elif actual_total == expected_total_raw:
            status = "一致"
        elif actual_total == expected_bundle:
            status = f"与PDF标注不一致，但考虑 bundle 后相符（差 {actual_total - expected_total_raw}）"
        elif actual_total == expected_final:
            status = (
                f"与PDF标注不一致，但考虑 bundle 与 Mystery 抵扣后相符"
                f"（bundle +{bundle_extra_units}，Mystery −{mystery_units}，最终期望 {expected_final}）"
            )
        else:
            diff = actual_total - expected_total_raw
            status = (
                f"不一致（差 {diff}；bundle 影响 +{bundle_extra_units}；"
                f"Mystery 抵扣 −{mystery_units}；按理应为 {expected_final}）"
            )

        pdf_item_list.append({
            "PDF文件": pf.name,
            "Item quantity（原始）": expected_total_raw,
            "bundle 额外件数(+)": bundle_extra_units,
            "Mystery(NM001) 件数(−)": mystery_units,
            "调整后应为（bundle − Mystery）": expected_final,
            "提取出货数量": actual_total,
            "状态": status
        })

        per_pdf_expected.append(expected_total_raw)
        per_pdf_extra.append(bundle_extra_units)
        per_pdf_actual.append(actual_total)
        per_pdf_mystery.append(mystery_units)

    # —— 显示 PDF 对账表 + 合计行 ——
    st.subheader("各 PDF 的 Item quantity 对账表（bundle − Mystery 口径）")
    pdf_df = pd.DataFrame(pdf_item_list)

    total_expected_raw = sum(per_pdf_expected) if per_pdf_expected else 0
    total_bundle_extra = sum(per_pdf_extra) if per_pdf_extra else 0
    total_mystery = sum(per_pdf_mystery) if per_pdf_mystery else 0
    total_expected_bundle = total_expected_raw + total_bundle_extra
    total_expected_final = total_expected_bundle - total_mystery   # ⚠️ 合计最终期望
    total_actual = sum(per_pdf_actual) if per_pdf_actual else 0

    if not pdf_df.empty:
        # 合计状态
        if total_actual == total_expected_raw:
            total_status = "一致"
        elif total_actual == total_expected_bundle:
            total_status = f"与PDF标注不一致，但考虑 bundle 后相符（差 {total_actual - total_expected_raw}）"
        elif total_actual == total_expected_final:
            total_status = (
                f"与PDF标注不一致，但考虑 bundle 与 Mystery 抵扣后相符"
                f"（bundle +{total_bundle_extra}，Mystery −{total_mystery}，最终期望 {total_expected_final}）"
            )
        else:
            diff = total_actual - total_expected_raw
            total_status = (
                f"不一致（差 {diff}；bundle 影响 +{total_bundle_extra}；"
                f"Mystery 抵扣 −{total_mystery}；按理应为 {total_expected_final}）"
            )

        total_row = {
            "PDF文件": "合计",
            "Item quantity（原始）": total_expected_raw,
            "bundle 额外件数(+)": total_bundle_extra,
            "Mystery(NM001) 件数(−)": total_mystery,
            "调整后应为（bundle − Mystery）": total_expected_final,
            "提取出货数量": total_actual,
            "状态": total_status
        }
        pdf_df = pd.concat([pdf_df, pd.DataFrame([total_row])], ignore_index=True)

    st.dataframe(pdf_df, use_container_width=True)

    # —— 汇总所有 PDF 的 SKU ——
    sku_counts_all = defaultdict(int)
    for counts in pdf_sku_counts.values():
        for sku, qty in counts.items():
            sku_counts_all[sku] += qty

    # —— 换货处理（旧格式）：提取替换 + 库存调整（每行原款 +1、换货 -1） ——
    if exchange_df is not None:
        if "原款式" in exchange_df.columns and "换货款式" in exchange_df.columns:
            for _, row in exchange_df.iterrows():
                original_sku = str(row["原款式"]).strip()
                new_sku = str(row["换货款式"]).strip()
                # 替换提取数量（原款 → 换货）——旧逻辑：将该原款当日全部提取量转移给新款
                if sku_counts_all.get(original_sku):
                    qty = sku_counts_all.pop(original_sku)
                    sku_counts_all[new_sku] += qty
                # 直接修改库存（对应日期列）：原款 +1、换货 -1
                stock_df.loc[stock_df["SKU编码"] == original_sku, stock_date_col] += 1
                stock_df.loc[stock_df["SKU编码"] == new_sku, stock_date_col] -= 1
            st.success("换货处理完成（旧格式）：已替换提取数量并调整库存（原款 +1 / 换货 -1）")
        else:
            st.warning("换货表（旧格式）中必须包含“原款式”和“换货款式”两列")

    # —— 达人换货统计表（新格式，逐行一件）：原款 -1、发货 +1；库存原款 +1、发货 -1 ——
    if creator_swap_df is not None:
        # 兼容不同空格/命名
        rename_map = {}
        cols = [c.strip() for c in creator_swap_df.columns]
        creator_swap_df.columns = cols
        if "原款式 SKU" in cols:
            rename_map["原款式 SKU"] = "原SKU"
        if "原款式SKU" in cols:
            rename_map["原款式SKU"] = "原SKU"
        if "发货款式SKU" in cols:
            rename_map["发货款式SKU"] = "新SKU"
        if "发货款式 SKU" in cols:
            rename_map["发货款式 SKU"] = "新SKU"
        creator_swap_df = creator_swap_df.rename(columns=rename_map)

        if {"原SKU", "新SKU"}.issubset(creator_swap_df.columns):
            # 逐行一件执行：sku_counts_all[原]-=1; sku_counts_all[新]+=1; 库存 原+1 新-1
            applied_rows = 0
            missing_in_sold = 0
            for _, row in creator_swap_df.iterrows():
                original_sku = str(row["原SKU"]).strip()
                new_sku = str(row["新SKU"]).strip()

                # 对“提取/售出计数”的修正（仅逐件）
                if original_sku:
                    if sku_counts_all.get(original_sku, 0) > 0:
                        sku_counts_all[original_sku] -= 1
                        if sku_counts_all[original_sku] == 0:
                            del sku_counts_all[original_sku]
                    else:
                        # 统计表可能有“原SKU”不在当日提取里的情况
                        missing_in_sold += 1
                if new_sku:
                    sku_counts_all[new_sku] += 1

                # 对库存的修正：原款 +1、发货 -1
                if original_sku:
                    stock_df.loc[stock_df["SKU编码"] == original_sku, stock_date_col] += 1
                if new_sku:
                    stock_df.loc[stock_df["SKU编码"] == new_sku, stock_date_col] -= 1

                applied_rows += 1

            msg_tail = "" if missing_in_sold == 0 else f"；其中 {missing_in_sold} 行原SKU未在当日提取中找到，仅执行了库存修正与发货SKU计数增加"
            st.success(f"达人换货处理完成（新格式，逐行一件）：共应用 {applied_rows} 行{msg_tail}")
        else:
            st.warning("达人换货统计表需要包含列：“原款式 SKU/原款式SKU” 与 “发货款式SKU/发货款式 SKU”")

    # —— 合并库存数据（NM001 同样参与扣减） ——
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

    # 总对账提示（顺序：原始 → bundle → bundle−NM001）
    total_sold = summary_df.loc["合计", "Sold Qty"]
    if total_expected_raw > 0:
        if total_sold == total_expected_raw:
            st.success(f"提取成功：共 {total_sold} 件，与 PDF 标注汇总一致")
        elif total_sold == total_expected_bundle:
            st.success(
                f"提取成功：共 {total_sold} 件。与 PDF 原始汇总不一致，但考虑 bundle 后相符（差 {total_sold - total_expected_raw}）。"
            )
        elif total_sold == total_expected_final:
            st.success(
                f"提取成功：共 {total_sold} 件。与 PDF 原始汇总不一致，但考虑 bundle 与 Mystery 抵扣后相符"
                f"（bundle +{total_bundle_extra}，Mystery −{total_mystery}）。"
            )
        else:
            st.error(
                f"提取数量 {total_sold} 与 PDF 标注汇总不一致；"
                f"原始: {total_expected_raw}，bundle 调整后: {total_expected_bundle}，"
                f"bundle−Mystery 调整后: {total_expected_final}（bundle +{total_bundle_extra}，Mystery −{total_mystery}）。"
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
        "bundle 额外件数(+)": total_bundle_extra if total_bundle_extra else "",
        "Mystery（NM001）件数(−)": total_mystery if total_mystery else "",
        "PDF标注数量（最终，bundle−Mystery）": total_expected_final if total_expected_final else "",
        "提取出货数量": total_sold
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
