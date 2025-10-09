import streamlit as st
import pandas as pd
import pdfplumber
import fitz
import re
from collections import defaultdict
from io import BytesIO
from datetime import datetime
import os

st.set_page_config(page_title="NailVesta 库存系统", layout="centered")
st.title("ColorFour Inventory 系统")

# ---------- 通用：文本标准化 & 断行修复（沿用 new picking list 的做法） ----------
def _normalize(t: str) -> str:
    if not t:
        return ""
    t = t.replace("\u00ad", "").replace("\u200b", "").replace("\u00a0", " ")
    t = t.replace("–", "-").replace("—", "-")
    return t

def _fix_orphan_digit_before_size(txt: str) -> str:
    """
    修复跨行断在 size 前一位的 bundle：
      NPJ011NPX01\n5-M  ->  NPJ011NPX015-M
    """
    pat = re.compile(r'((?:[A-Z]{3}\d{3}){0,3}[A-Z]{3}\d{2})\s*[\r\n]+\s*(\d)\s*-\s*([SML])')
    prev, cur = None, txt
    while prev != cur:
        prev, cur = cur, pat.sub(lambda m: f"{m.group(1)}{m.group(2)}-{m.group(3)}", cur)
    return cur

SKU_BUNDLE = re.compile(r'((?:[A-Z]{3}\d{3}[\s\n]*){1,4}-[SML])', re.DOTALL)

# 数量：在 SKU 后 120 字符内找“数量 + 订单号(>=9位)”；找不到就当 1
QTY_NEAR   = re.compile(r'\b([1-9]\d{0,2})\b(?:\s+\d{9,})?')

def _expand_bundle_or_single(sku_with_size: str, qty: int, counter: dict):
    sku_with_size = re.sub(r'\s+', '', sku_with_size.strip())
    if "-" not in sku_with_size:
        counter[sku_with_size] += qty
        return
    code, size = sku_with_size.split("-", 1)
    code, size = code.strip(), size.strip()
    if len(code) % 6 == 0 and 6 <= len(code) <= 24:
        segs = [code[i:i+6] for i in range(0, len(code), 6)]
        if all(re.fullmatch(r"[A-Z]{3}\d{3}", s) for s in segs):
            for s in segs:
                counter[f"{s}-{size}"] += qty
            return
    counter[sku_with_size] += qty

def _extract_text_plumber_then_fitz(pf) -> str:
    """优先 pdfplumber，若文本太少就回退 fitz；最后拼成一块文本。"""
    # plumber
    all_text = []
    try:
        with pdfplumber.open(pf) as pdf:
            for p in pdf.pages:
                all_text.append(_normalize(p.extract_text() or ""))
    except Exception:
        pass
    text = "\n".join(all_text).strip()

    # 回退：fitz
    if len(text) < 30:  # 很短，基本抽不到
        try:
            pf.seek(0)
        except Exception:
            pass
        try:
            doc = fitz.open(stream=pf.read() if hasattr(pf, "read") else pf, filetype="pdf")
            text2 = []
            for page in doc:
                text2.append(_normalize(page.get_text()))
            text = "\n".join(text2).strip()
        except Exception:
            pass
    return text

def extract_skus_from_pdf(pf) -> tuple[dict, int]:
    """
    —— 关键函数（直接搬自 new picking list 的思路）——
    返回：(sku_counts_single, item_quantity_mark)
    """
    # 读取第一页的 Item quantity
    item_q = ""
    try:
        with pdfplumber.open(pf) as pdf:
            first = _normalize(pdf.pages[0].extract_text() or "")
            m = re.search(r'Item\s+quantity[:：]?\s*(\d+)', first, re.I)
            item_q = int(m.group(1)) if m else ""
    except Exception:
        pass

    # 全文文本（plumber→fitz）
    try:
        pf.seek(0)
    except Exception:
        pass
    full = _extract_text_plumber_then_fitz(pf)
    full = _fix_orphan_digit_before_size(full)

    sku_counts = defaultdict(int)

    # 识别 1–4 件 bundle（允许穿插换行）
    for m in SKU_BUNDLE.finditer(full):
        raw = re.sub(r'\s+', '', m.group(1))
        lookahead = full[m.end(): m.end() + 120]
        mq = QTY_NEAR.search(lookahead)
        qty = int(mq.group(1)) if mq else 1
        _expand_bundle_or_single(raw, qty, sku_counts)

    # 兜底：无 SKU 行里若有“数量+>=9位数字”，保留到 MISSING_，后续可手动补录
    for line in full.split("\n"):
        m2 = re.search(r'^\s*(\d{1,3})\s+\d{9,}\s*$', line.strip())
        if m2:
            sku_counts[f"MISSING_{len(sku_counts)}"] += int(m2.group(1))

    return sku_counts, item_q

# ================= UI 上传 =================
pdf_files = st.file_uploader("上传 Picking List PDF（可多选）", type=["pdf"], accept_multiple_files=True, key="pdf_uploader")
csv_file = st.file_uploader("上传库存表 CSV", type=["csv"])

selected_pdfs = []
if pdf_files:
    selected_names = st.multiselect(
        "选择要参与统计的 Picking List PDF",
        options=[f.name for f in pdf_files],
        default=[f.name for f in pdf_files]
    )
    selected_pdfs = [f for f in pdf_files if f.name in selected_names]

# —— 达人换货（保持原逻辑）——
if "show_exchange" not in st.session_state:
    st.session_state.show_exchange = False
if st.button("有达人换货吗？"):
    st.session_state.show_exchange = True
exchange_df = None
if st.session_state.show_exchange:
    st.info("请上传换货记录文件（CSV / Excel），将执行：原款 +1、换货 -1（每行各一件）")
    exchange_file = st.file_uploader("上传换货记录", type=["csv", "xlsx"])
    if exchange_file:
        exchange_df = pd.read_csv(exchange_file) if exchange_file.name.endswith(".csv") else pd.read_excel(exchange_file)
        st.success("换货表已上传")

# ================= 主流程 =================
if selected_pdfs and csv_file:
    st.success("文件上传成功，开始处理...")

    stock_df = pd.read_csv(csv_file)
    stock_df.columns = [c.strip() for c in stock_df.columns]
    stock_col = [c for c in stock_df.columns if re.match(r"\d{2}/\d{2}", c)]
    if not stock_col:
        st.error("未找到库存日期列（如 '06/03'）")
        st.stop()
    stock_date_col = stock_col[0]
    stock_skus = set(stock_df["SKU编码"].astype(str).str.strip())

    pdf_item_list = []
    pdf_sku_counts = {}

    for pf in selected_pdfs:
        # 关键：用 new picking list 同款解析
        # 需要多次读取同一个文件流时，先保存 buffer
        data = pf.read()
        sku_counts_single, item_q = extract_skus_from_pdf(BytesIO(data))
        pdf_sku_counts[pf.name] = sku_counts_single

        actual_total = sum(q for s, q in sku_counts_single.items() if not s.startswith("MISSING_"))
        status = "无标注" if item_q == "" else ("一致" if actual_total == item_q else f"不一致（差 {actual_total - item_q}）")

        pdf_item_list.append({
            "PDF文件": pf.name,
            "Item quantity": item_q,
            "提取出货数量": actual_total,
            "状态": status
        })

    # —— 对账表
    st.subheader("各 PDF 的 Item quantity 对账表")
    pdf_df = pd.DataFrame(pdf_item_list)
    total_expected = pdf_df["Item quantity"].replace("", 0).astype(int).sum() if not pdf_df.empty else 0
    total_actual = pdf_df["提取出货数量"].sum() if not pdf_df.empty else 0
    total_status = "—" if total_expected == 0 else ("一致" if total_actual == total_expected else f"不一致（差 {total_actual - total_expected}）")
    if not pdf_df.empty:
        pdf_df = pd.concat([pdf_df, pd.DataFrame({"PDF文件": ["合计"], "Item quantity": [total_expected], "提取出货数量": [total_actual], "状态": [total_status]})], ignore_index=True)
    st.dataframe(pdf_df, use_container_width=True)

    # —— 汇总所有 PDF 的 SKU（bundle 已拆开）
    sku_counts_all = defaultdict(int)
    missing_lines = []
    raw_missing = []
    for pf_name, counts in pdf_sku_counts.items():
        for sku, qty in counts.items():
            if sku.startswith("MISSING_"):
                missing_lines.append(qty)
                raw_missing.append(f"{pf_name} 中缺SKU的 {qty} 件")
            else:
                sku_counts_all[sku] += qty

    # 缺 SKU 补录（依然支持 bundle，自动拆）
    if missing_lines:
        st.warning("以下出货记录缺 SKU，请补录：")
        manual_entries = {}
        for i, raw in enumerate(raw_missing):
            manual_entries[i] = st.text_input(f"“{raw}”的 SKU 是：", key=f"miss_{i}")
        if st.button("确认补录"):
            for i, sku in manual_entries.items():
                if sku and sku != "":
                    _expand_bundle_or_single(sku.strip(), missing_lines[i], sku_counts_all)
            st.success("已将补录 SKU 添加进库存统计")

    # —— 换货处理（保持原逻辑）
    if exchange_df is not None:
        if "原款式" in exchange_df.columns and "换货款式" in exchange_df.columns:
            for _, row in exchange_df.iterrows():
                o = str(row["原款式"]).strip()
                n = str(row["换货款式"]).strip()
                if sku_counts_all.get(o):
                    qty = sku_counts_all.pop(o)
                    sku_counts_all[n] += qty
                stock_df.loc[stock_df["SKU编码"] == o, stock_date_col] += 1
                stock_df.loc[stock_df["SKU编码"] == n, stock_date_col] -= 1
            st.success("换货处理完成：已替换提取数量并调整库存（原款 +1 / 换货 -1）")
        else:
            st.warning("换货表中必须包含“原款式”和“换货款式”两列")

    # —— 合并库存数据（保持原逻辑）
    stock_df["Sold"] = stock_df["SKU编码"].map(sku_counts_all).fillna(0).astype(int)
    stock_df["New Stock"] = stock_df[stock_date_col] - stock_df["Sold"]
    summary_df = stock_df[["SKU编码", stock_date_col, "Sold", "New Stock"]].copy()
    summary_df.columns = ["SKU", "Old Stock", "Sold Qty", "New Stock"]
    summary_df.index += 1
    summary_df.loc["合计"] = ["—", summary_df["Old Stock"].sum(), summary_df["Sold Qty"].sum(), summary_df["New Stock"].sum()]

    st.subheader("库存更新结果")
    st.dataframe(summary_df, use_container_width=True)

    # —— 总对账
    total_sold = summary_df.loc["合计", "Sold Qty"]
    if total_expected and total_expected > 0:
        if total_sold == total_expected:
            st.success(f"提取成功：共 {total_sold} 件，与 PDF 标注汇总一致")
        else:
            st.error(f"提取数量 {total_sold} 与 PDF 标注汇总 {total_expected} 不一致")
    else:
        st.warning("未识别 PDF 中的 Item quantity")

    # —— 一键复制 New Stock
    st.subheader("一键复制 New Stock")
    new_stock_text = "\n".join(summary_df.iloc[:-1]["New Stock"].astype(str).tolist())
    st.code(new_stock_text, language="text")

    # —— 下载 Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary_df.to_excel(writer, index_label="序号")
    st.download_button("下载库存更新表 Excel", data=output.getvalue(), file_name="库存更新结果.xlsx")

    # —— 上传历史记录
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
