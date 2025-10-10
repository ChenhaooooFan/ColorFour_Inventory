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
st.title("ColorFour Inventory 系统（fitz 解析 + bundle 修复）")

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

# —— 按钮触发：是否有达人换货 —— #
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
# 1–4 件 bundle（允许跨行/空格）
SKU_BUNDLE = re.compile(r'((?:[A-Z]{3}\d{3}){1,4}-[SML])', re.DOTALL)
# SKU 右侧 1–3 位数量（和你“拣货单汇总工具”的写法保持一致）
QTY_AFTER  = re.compile(r'\b([1-9]\d{0,2})\b')
# “Item quantity”
ITEM_QTY_RE = re.compile(r"Item\s+quantity[:：]?\s*(\d+)", re.I)

def normalize_text(t: str) -> str:
    t = t.replace("\u00ad","").replace("\u200b","").replace("\u00a0"," ")
    t = t.replace("–","-").replace("—","-")
    return t

def fix_orphan_digit_before_size(txt: str) -> str:
    """
    修复形如：
        NPJ011NPX01\n5-M  → NPJ011NPX015-M
    的换行折断（最后一段“3位数字”被切成“2位在上一行 + 1位在下一行，再接 -SIZE”）。
    """
    pattern = re.compile(
        r'(?P<prefix>(?:[A-Z]{3}\d{3}){0,3}[A-Z]{3}\d{2})\s*[\r\n]+\s*(?P<d>\d)\s*-\s*(?P<size>[SML])'
    )
    def _join(m):
        return f"{m.group('prefix')}{m.group('d')}-{m.group('size')}"
    prev = None
    cur = txt
    while prev != cur:
        prev = cur
        cur = pattern.sub(_join, cur)
    return cur

def expand_bundle(counter: dict, sku_with_size: str, qty: int):
    """
    将 1–4 件 bundle 拆成独立 SKU 计数。
    例如：NPJ011NPX015-M → NPJ011-M, NPX015-M 各 +qty
    """
    s = re.sub(r'\s+', '', sku_with_size)
    if '-' not in s:
        counter[s] += qty
        return
    code, size = s.split('-', 1)
    if len(code) % 6 == 0 and 6 <= len(code) <= 24:
        parts = [code[i:i+6] for i in range(0, len(code), 6)]
        if all(re.fullmatch(r'[A-Z]{3}\d{3}', p) for p in parts):
            for p in parts:
                counter[f"{p}-{size}"] += qty
            return
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

    # 读取库存 CSV
    stock_df = pd.read_csv(csv_file)
    stock_df.columns = [col.strip() for col in stock_df.columns]
    stock_col = [col for col in stock_df.columns if re.match(r"\d{2}/\d{2}", col)]
    if not stock_col:
        st.error("未找到库存日期列（如 '06/03'）")
        st.stop()
    stock_date_col = stock_col[0]

    # —— 每个 PDF：解析 —— #
    pdf_item_list = []
    pdf_sku_counts = {}

    for pf in selected_pdfs:
        file_bytes = pf.read()  # PyMuPDF 需要 bytes
        expected_total, sku_counts_single = parse_pdf_with_fitz(file_bytes)
        pdf_sku_counts[pf.name] = sku_counts_single
        actual_total = sum(sku_counts_single.values())

        # 对账状态
        if expected_total == 0:
            status = "未识别到 Item quantity"
        elif actual_total == expected_total:
            status = "一致"
        else:
            status = f"不一致（差 {actual_total - expected_total}）"

        pdf_item_list.append({
            "PDF文件": pf.name,
            "Item quantity": expected_total,
            "提取出货数量": actual_total,
            "状态": status
        })

    # —— 显示 PDF 对账表 + 合计行 —— 
    st.subheader("各 PDF 的 Item quantity 对账表")
    pdf_df = pd.DataFrame(pdf_item_list)
    total_expected = pdf_df["Item quantity"].replace("", 0).astype(int).sum() if not pdf_df.empty else 0
    total_actual = pdf_df["提取出货数量"].sum() if not pdf_df.empty else 0
    total_status = "一致" if total_actual == total_expected else f"不一致（差 {total_actual - total_expected}）"

    if not pdf_df.empty:
        pdf_df = pd.concat([pdf_df, pd.DataFrame({
            "PDF文件": ["合计"],
            "Item quantity": [total_expected],
            "提取出货数量": [total_actual],
            "状态": [total_status]
        })], ignore_index=True)

    st.dataframe(pdf_df, use_container_width=True)

    # —— 汇总所有 PDF 的 SKU —— #
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

    # —— 合并库存数据 —— #
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

    # 总对账
    total_sold = summary_df.loc["合计", "Sold Qty"]
    if total_expected > 0:
        if total_sold == total_expected:
            st.success(f"提取成功：共 {total_sold} 件，与 PDF 标注汇总一致")
        else:
            st.error(f"提取数量 {total_sold} 与 PDF 标注汇总 {total_expected} 不一致")

    # 可复制 New Stock
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
        "PDF标注数量": total_expected if total_expected else "",
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
