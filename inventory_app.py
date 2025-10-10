import streamlit as st
import pandas as pd
import pdfplumber
import re
import fitz  # PyMuPDF
from collections import defaultdict
from io import BytesIO
from datetime import datetime
import os

# ---------------- UI ----------------
st.set_page_config(page_title="NailVesta 库存系统", layout="centered")
st.title("ColorFour Inventory 系统")
st.title("ColorFour Inventory 系统（fitz 解析 + bundle 修复）")

# 上传文件（PDF 支持多选）
pdf_files = st.file_uploader("上传 Picking List PDF（可多选）", type=["pdf"], accept_multiple_files=True)
csv_file = st.file_uploader("上传库存表 CSV", type=["csv"])

# 选择要参与统计的 PDF（默认全选）
selected_pdfs = []
if pdf_files:
    selected_names = st.multiselect(
@@ -23,8 +25,10 @@
    )
    selected_pdfs = [f for f in pdf_files if f.name in selected_names]

# —— 按钮触发：是否有达人换货 —— #
if "show_exchange" not in st.session_state:
    st.session_state.show_exchange = False

if st.button("有达人换货吗？"):
    st.session_state.show_exchange = True

@@ -40,12 +44,12 @@
        st.success("换货表已上传")

# ---------- 规则与小工具 ----------
# 1–4 件 bundle（允许跨行/空格）；匹配到的 token 形如 “NPJ011NPX015-M” 或 “NPF001-S”
# 1–4 件 bundle（允许跨行/空格）
SKU_BUNDLE = re.compile(r'((?:[A-Z]{3}\d{3}){1,4}-[SML])', re.DOTALL)
# SKU 后面 1–3 位的数量
# SKU 右侧 1–3 位数量（和你“拣货单汇总工具”的写法保持一致）
QTY_AFTER  = re.compile(r'\b([1-9]\d{0,2})\b')
# “大概率是订单号/条码”的长数字
LONG_ID    = re.compile(r'\d{9,}')
# “Item quantity”
ITEM_QTY_RE = re.compile(r"Item\s+quantity[:：]?\s*(\d+)", re.I)

def normalize_text(t: str) -> str:
    t = t.replace("\u00ad","").replace("\u200b","").replace("\u00a0"," ")
@@ -86,136 +90,86 @@ def expand_bundle(counter: dict, sku_with_size: str, qty: int):
            for p in parts:
                counter[f"{p}-{size}"] += qty
            return
    counter[s] += qty  # 回退：非标准就按原样记
    # 回退：非标准就按原样记
    counter[s] += qty

def parse_pdf_with_fitz(file_bytes: bytes):
    """
    返回： (expected_total, sku_counts_single)
    - expected_total: PDF里“Item quantity”
    - sku_counts_single: dict，键为 Seller SKU（含尺码），值为累计数量
    解析策略 = 你的“拣货单汇总工具”：整文提取 → 标准化 → 修复换行 → 跨行匹配 SKU → 右侧抓 Qty → expand_bundle
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages_text = []
    for i, p in enumerate(doc):
        pages_text.append(p.get_text("text"))
    text_raw = "\n".join(pages_text)
    text = normalize_text(text_raw)

    # 对账用的 Item quantity（只要第一页也行；整文里搜一遍更稳）
    m_total = ITEM_QTY_RE.search(text)
    expected_total = int(m_total.group(1)) if m_total else 0

    # 关键修复：把“最后一位数字换行到下一行”的 SKU 拼回去
    text_fixed = fix_orphan_digit_before_size(text)

    # 匹配所有 SKU（允许跨行）
    sku_counts = defaultdict(int)
    for m in SKU_BUNDLE.finditer(text_fixed):
        sku_raw = re.sub(r'\s+', '', m.group(1))  # 去空白
        # 在 token 之后的一小段里找 Qty（保持与你可用代码一致，不再强制校验长条码）
        after = text_fixed[m.end(): m.end()+50]
        mq = QTY_AFTER.search(after)
        qty = int(mq.group(1)) if mq else 1
        expand_bundle(sku_counts, sku_raw, qty)

    return expected_total, sku_counts

# ---------- 主流程 ----------
if selected_pdfs and csv_file:
    st.success("文件上传成功，开始处理...")

    # 读库存
    # 读取库存 CSV
    stock_df = pd.read_csv(csv_file)
    stock_df.columns = [col.strip() for col in stock_df.columns]
    stock_col = [col for col in stock_df.columns if re.match(r"\d{2}/\d{2}", col)]
    if not stock_col:
        st.error("未找到库存日期列（如 '06/03'）")
        st.stop()
    stock_date_col = stock_col[0]
    stock_skus = set(stock_df["SKU编码"].astype(str).str.strip())

    # —— 每个 PDF：解析 —— #
    pdf_item_list = []
    pdf_sku_counts = {}
    pdf_nm001_counts = {}
    pdf_hb_counts = {}

    def _scan_holiday_bunny_qty(line: str) -> int:
        if not re.search(r'holiday\s*bunny', line, flags=re.I):
            return 0
        m = re.search(r'holiday\s*bunny.*?(\d{1,3})\s+\d{9,}', line, flags=re.I)
        if m:
            return int(m.group(1))
        has_long_digits = re.search(r'\d{9,}', line) is not None
        if has_long_digits:
            nums = re.findall(r'\b(\d{1,3})\b', line)
            if nums:
                return int(nums[0])
        return 0

    for pf in selected_pdfs:
        # 1) Item quantity
        with pdfplumber.open(pf) as pdf:
            first_page_text = pdf.pages[0].extract_text()
            item_match = re.search(r'Item quantity[:：]?\s*(\d+)', first_page_text or "")
            qty_val = int(item_match.group(1)) if item_match else ""

        # 2) 提取 SKU（新版：先修复断行 → 匹配 token → 向后找 Qty + 长条码）
        sku_counts_single = defaultdict(int)
        nm001_qty_scan = 0
        hb_qty_scan = 0

        with pdfplumber.open(pf) as pdf:
            for page in pdf.pages:
                raw = page.extract_text() or ""
                txt = fix_orphan_digit_before_size(normalize_text(raw))
                lines = txt.split("\n")
                n = len(lines)

                for i, line in enumerate(lines):
                    line = line.strip()
                    # NM001 / Holiday Bunny 扫描（仅对账说明）
                    m_nm = re.search(r'\bNM001\b\s+(\d{1,3})\s+\d{9,}', line)
                    if m_nm:
                        nm001_qty_scan += int(m_nm.group(1))
                    hb_qty_scan += _scan_holiday_bunny_qty(line)

                    # 找到本行所有 SKU token
                    for m in SKU_BUNDLE.finditer(line):
                        token = m.group(1)
                        tail = line[m.end():]
                        # 常见：数量和订单号就在后面；但也可能“数量开头跑到下一行”
                        if i + 1 < n:
                            tail += " " + lines[i+1].strip()

                        # 需要一个 Qty 和（同一段中的）一个 ≥9位长数字来确认
                        qty_m = QTY_AFTER.search(tail)
                        has_long = LONG_ID.search(tail) is not None
                        if qty_m and has_long:
                            qty = int(qty_m.group(1))
                            expand_bundle(sku_counts_single, token, qty)
                        # 若这行没有长条码，就忽略（避免误把描述文本里的小数字当成数量）

        file_bytes = pf.read()  # PyMuPDF 需要 bytes
        expected_total, sku_counts_single = parse_pdf_with_fitz(file_bytes)
        pdf_sku_counts[pf.name] = sku_counts_single
        pdf_nm001_counts[pf.name] = nm001_qty_scan
        pdf_hb_counts[pf.name] = hb_qty_scan
        actual_total = sum(sku_counts_single.values())

        actual_total = sum(q for s, q in sku_counts_single.items() if not s.startswith("MISSING_"))

        # 3) 状态判定（考虑 NM001 / Holiday Bunny 解释）
        if qty_val == "":
            status = "无标注"
        # 对账状态
        if expected_total == 0:
            status = "未识别到 Item quantity"
        elif actual_total == expected_total:
            status = "一致"
        else:
            diff = actual_total - qty_val
            if diff == 0:
                status = "一致"
            elif ("NM001" not in stock_skus) and (actual_total + nm001_qty_scan == qty_val):
                status = f"一致（差 {nm001_qty_scan} 件，均为 NM001，库存无此 SKU）"
            elif (actual_total + hb_qty_scan == qty_val):
                status = f"一致（差 {hb_qty_scan} 件，均为 Holiday Bunny，未被正则识别）"
            elif ("NM001" not in stock_skus) and (actual_total + nm001_qty_scan + hb_qty_scan == qty_val):
                status = f"一致（差 {nm001_qty_scan + hb_qty_scan} 件，其中 NM001 {nm001_qty_scan}、Holiday Bunny {hb_qty_scan}）"
            else:
                if hb_qty_scan > 0:
                    status = f"不一致（差 {diff}；Holiday Bunny 扫描到 {hb_qty_scan} 件）"
                else:
                    status = f"不一致（差 {diff}）"
            status = f"不一致（差 {actual_total - expected_total}）"

        pdf_item_list.append({
            "PDF文件": pf.name,
            "Item quantity": qty_val,
            "Item quantity": expected_total,
            "提取出货数量": actual_total,
            "状态": status
        })

    # —— 对账表 —— #
    # —— 显示 PDF 对账表 + 合计行 —— 
    st.subheader("各 PDF 的 Item quantity 对账表")
    pdf_df = pd.DataFrame(pdf_item_list)
    total_expected = pdf_df["Item quantity"].replace("", 0).astype(int).sum() if not pdf_df.empty else 0
    total_actual = pdf_df["提取出货数量"].sum() if not pdf_df.empty else 0
    nm001_total_scan = sum(pdf_nm001_counts.values())
    hb_total_scan = sum(pdf_hb_counts.values())

    if total_expected > 0:
        if total_actual == total_expected:
            total_status = "一致"
        elif ("NM001" not in stock_skus) and (total_actual + nm001_total_scan == total_expected):
            total_status = f"一致（差 {nm001_total_scan} 件，均为 NM001，库存无此 SKU）"
        elif (total_actual + hb_total_scan == total_expected):
            total_status = f"一致（差 {hb_total_scan} 件，均为 Holiday Bunny，未被正则识别）"
        elif ("NM001" not in stock_skus) and (total_actual + nm001_total_scan + hb_total_scan == total_expected):
            total_status = f"一致（差 {nm001_total_scan + hb_total_scan} 件，其中 NM001 {nm001_total_scan}、Holiday Bunny {hb_total_scan}）"
        else:
            total_status = f"不一致（差 {total_actual - total_expected}；Holiday Bunny 扫描到 {hb_total_scan} 件）"
    else:
        total_status = "—"
    total_status = "一致" if total_actual == total_expected else f"不一致（差 {total_actual - total_expected}）"

    if not pdf_df.empty:
        pdf_df = pd.concat([pdf_df, pd.DataFrame({
@@ -224,37 +178,33 @@ def _scan_holiday_bunny_qty(line: str) -> int:
            "提取出货数量": [total_actual],
            "状态": [total_status]
        })], ignore_index=True)
    st.dataframe(pdf_df, use_container_width=True)

    if hb_total_scan > 0:
        st.info(f"提示：扫描到 Holiday Bunny 共 {hb_total_scan} 件。如果未自动识别，请在下面“缺 SKU 补录”输入其对应的 SKU 后确认。")
    st.dataframe(pdf_df, use_container_width=True)

    # —— 汇总所有 PDF 的 SKU —— #
    sku_counts_all = defaultdict(int)
    for counts in pdf_sku_counts.values():
        for sku, qty in counts.items():
            if not sku.startswith("MISSING_"):
                sku_counts_all[sku] += qty
            sku_counts_all[sku] += qty

    # —— 换货处理 —— #
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

    # —— 合并库存并展示 —— #
    # —— 合并库存数据 —— #
    stock_df["Sold"] = stock_df["SKU编码"].map(sku_counts_all).fillna(0).astype(int)
    stock_df["New Stock"] = stock_df[stock_date_col] - stock_df["Sold"]
    summary_df = stock_df[["SKU编码", stock_date_col, "Sold", "New Stock"]].copy()
@@ -267,34 +217,24 @@ def _scan_holiday_bunny_qty(line: str) -> int:
        summary_df["New Stock"].sum()
    ]

    # 展示库存更新结果
    st.subheader("库存更新结果")
    st.dataframe(summary_df, use_container_width=True)

    # —— 总对账 —— #
    # 总对账
    total_sold = summary_df.loc["合计", "Sold Qty"]
    if total_expected and total_expected > 0:
    if total_expected > 0:
        if total_sold == total_expected:
            st.success(f"提取成功：共 {total_sold} 件，与 PDF 标注汇总一致")
        elif ("NM001" not in stock_skus) and (total_sold + nm001_total_scan == total_expected):
            st.success(f"提取成功：共 {total_sold} 件（差 {nm001_total_scan} 件，均为 NM001，库存无此 SKU），与 PDF 标注汇总一致")
        elif (total_sold + hb_total_scan == total_expected):
            st.success(f"提取成功：共 {total_sold} 件（差 {hb_total_scan} 件，均为 Holiday Bunny，未被正则识别），与 PDF 标注汇总一致")
        elif ("NM001" not in stock_skus) and (total_sold + nm001_total_scan + hb_total_scan == total_expected):
            st.success(f"提取成功：共 {total_sold} 件（差 {nm001_total_scan + hb_total_scan} 件，其中 NM001 {nm001_total_scan}、Holiday Bunny {hb_total_scan}），与 PDF 标注汇总一致")
        else:
            if hb_total_scan > 0:
                st.error(f"提取数量 {total_sold} 与 PDF 标注汇总 {total_expected} 不一致；其中 Holiday Bunny 扫描到 {hb_total_scan} 件")
            else:
                st.error(f"提取数量 {total_sold} 与 PDF 标注汇总 {total_expected} 不一致")
    else:
        st.warning("未识别 PDF 中的 Item quantity")
            st.error(f"提取数量 {total_sold} 与 PDF 标注汇总 {total_expected} 不一致")

    # —— 一键复制 New Stock —— #
    # 可复制 New Stock
    st.subheader("一键复制 New Stock")
    new_stock_text = "\n".join(summary_df.iloc[:-1]["New Stock"].astype(str).tolist())
    st.code(new_stock_text, language="text")

    # —— 导出 Excel —— #
    # 下载 Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        summary_df.to_excel(writer, index_label="序号")
@@ -304,7 +244,7 @@ def _scan_holiday_bunny_qty(line: str) -> int:
        file_name="库存更新结果.xlsx"
    )

    # —— 上传历史记录 —— #
    # 上传历史记录
    history_file = "upload_history.csv"
    new_record = {
        "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
