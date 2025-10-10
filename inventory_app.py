import streamlit as st
import pandas as pd
import pdfplumber
import re
from collections import defaultdict
from io import BytesIO
from datetime import datetime
import os

st.set_page_config(page_title="NailVesta 库存系统", layout="centered")
st.title("ColorFour Inventory 系统")

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

# =========================
# Bundle 拆分工具（1–4件）
# =========================
SEG_RE = re.compile(r"[A-Z]{3}\d{3}")

def expand_bundle_or_single(sku_with_size: str, qty: int, counter: dict):
    s = sku_with_size.strip().upper().replace("–", "-").replace("—", "-")
    if "-" not in s:
        counter[s] += qty
        return
    code, size = s.split("-", 1)
    code = re.sub(r"\s+", "", code)
    size = size.strip()

    if len(code) % 6 == 0 and 6 <= len(code) <= 24:
        segs = [code[i:i+6] for i in range(0, len(code), 6)]
        if all(SEG_RE.fullmatch(seg) for seg in segs):
            for seg in segs:
                counter[f"{seg}-{size}"] += qty
            return
    # 回退
    counter[f"{code}-{size}"] += qty

# =========================
# 词元化与“断裂缝合”
# =========================
SKU_FULL_RE = re.compile(r"^(?:[A-Z]{3}\d{3}){1,4}-[SML]$")
ITEMQ_RE = re.compile(r'Item\s+quantity[:：]?\s*(\d+)', re.I)
QTY_TOKEN = re.compile(r"^\d{1,3}$")
ORDER_TOKEN = re.compile(r"^\d{9,}$")

def _norm_text(t: str) -> str:
    if not t: return ""
    return (t.replace("\u00ad", "")
             .replace("\u200b", "")
             .replace("\u00a0", " ")
             .replace("–", "-")
             .replace("—", "-"))

def _extract_tokens(fileobj):
    tokens = []
    with pdfplumber.open(fileobj) as pdf:
        for page in pdf.pages:
            words = page.extract_words(
                x_tolerance=2, y_tolerance=2,
                keep_blank_chars=False, use_text_flow=True
            )
            for w in words:
                txt = _norm_text(w["text"])
                for tk in txt.split():
                    if tk:
                        tokens.append(tk)
    return tokens

def _collect_prev_alnum(tokens, start_idx, max_chars=30):
    """
    从 start_idx 往前收集连续的 [A-Z0-9]+ 词元，拼接成字符串（最多 max_chars）
    """
    s = []
    i = start_idx
    while i >= 0:
        tk = tokens[i].upper()
        if re.fullmatch(r"[A-Z0-9]+", tk):
            s.append(tk)
            if sum(len(x) for x in s) >= max_chars:
                break
            i -= 1
        else:
            break
    s.reverse()
    return "".join(s), i

def _try_stitch_broken_sku(tokens, idx):
    """
    处理拆成两行的情况：
    前面若干 [A-Z0-9]+ 词元拼在一起是 code_raw，但长度 %6 == 5，
    当前词元若为 'd-S/M/L'（或 'd', 下一词元 '-M'），则把 d 补到 code_raw 尾部，得到完整 code。
    返回 (sku_with_size 或 None)
    """
    cur = tokens[idx]
    m = re.fullmatch(r"^(\d)-([SML])$", cur.upper())
    size = None
    digit = None
    consumed = 1

    if m:
        digit = m.group(1)
        size = m.group(2)
    else:
        # 兼容 '5', '-M' 分成两个词元
        if re.fullmatch(r"^\d$", cur) and idx + 1 < len(tokens):
            nxt = tokens[idx + 1].upper()
            m2 = re.fullmatch(r"^-[SML]$", nxt)
            if m2:
                digit = cur
                size = m2.group(0)[1]
                consumed = 2

    if digit is None:
        return None, 0

    # 回看前面的连续 [A-Z0-9]+ 作为 code_raw
    code_raw, _ = _collect_prev_alnum(tokens, idx - 1, max_chars=30)
    code_raw = code_raw.upper()

    if not code_raw:
        return None, 0

    # 只处理“差 1 位数字就满 6 的倍数”的场景
    if len(code_raw) % 6 != 5:
        return None, 0

    code = code_raw + digit
    # 校验切段
    if len(code) % 6 == 0 and 6 <= len(code) <= 24:
        segs = [code[i:i+6] for i in range(0, len(code), 6)]
        if all(SEG_RE.fullmatch(seg) for seg in segs):
            return f"{code}-{size}", consumed

    return None, 0

def parse_pdf_with_tokens(pf):
    """
    返回 (sku_counts_single, item_quantity_mark)
    - 先解析第一页 Item quantity
    - 词元流扫描：
        1) 直接识别一体的 SKU（含 bundle）
        2) 识别断裂：前缀 code_raw + 当前 'd-S' 补足
    - 数量=紧随其后 12 个词元内的 1~3 位数字，且其后 6 个词元内存在 9 位以上长数字；否则按 1 兜底
    """
    item_q = ""
    try:
        with pdfplumber.open(pf) as pdf:
            first_text = _norm_text((pdf.pages[0].extract_text() or ""))
            m = ITEMQ_RE.search(first_text)
            if m:
                item_q = int(m.group(1))
    except Exception:
        pass

    try: pf.seek(0)
    except Exception: pass

    tokens = _extract_tokens(pf)
    n = len(tokens)
    sku_counts = defaultdict(int)

    i = 0
    while i < n:
        tk = _norm_text(tokens[i].upper())

        # 情况 A：整个 SKU 一体
        if SKU_FULL_RE.fullmatch(tk):
            raw_sku = tk
            # 找数量 + 订单号
            qty = None
            for j in range(i, min(i + 20, n)):
                if QTY_TOKEN.fullmatch(tokens[j]):
                    if any(ORDER_TOKEN.fullmatch(tokens[k]) for k in range(j + 1, min(j + 7, n))):
                        qty = int(tokens[j]); break
            if qty is None: qty = 1

            expand_bundle_or_single(raw_sku, qty, sku_counts)
            i += 1
            continue

        # 情况 B：断裂 '…NPX01' + '5-M' 或 '5' '-M'
        stitched, consumed = _try_stitch_broken_sku(tokens, i)
        if stitched:
            qty = None
            # 从当前 i 往后找数量
            end_scan = min(i + 20, n)
            for j in range(i + consumed, end_scan):
                if QTY_TOKEN.fullmatch(tokens[j]):
                    if any(ORDER_TOKEN.fullmatch(tokens[k]) for k in range(j + 1, min(j + 7, n))):
                        qty = int(tokens[j]); break
            if qty is None: qty = 1

            expand_bundle_or_single(stitched, qty, sku_counts)
            i += consumed
            continue

        i += 1

    return sku_counts, (item_q if item_q != "" else "")

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

# —— 主流程 —— #
if selected_pdfs and csv_file:
    st.success("文件上传成功，开始处理...")

    # 读取库存 CSV（保持原逻辑）
    stock_df = pd.read_csv(csv_file)
    stock_df.columns = [col.strip() for col in stock_df.columns]
    stock_col = [col for col in stock_df.columns if re.match(r"\d{2}/\d{2}", col)]
    if not stock_col:
        st.error("未找到库存日期列（如 '06/03'）")
        st.stop()
    stock_date_col = stock_col[0]
    stock_skus = set(stock_df["SKU编码"].astype(str).str.strip())

    # 解析每个 PDF（新解析器）
    pdf_item_list = []
    pdf_sku_counts = {}
    pdf_nm001_counts = {}
    pdf_hb_counts = {}

    for pf in selected_pdfs:
        sku_counts_single, qty_val = parse_pdf_with_tokens(pf)
        pdf_sku_counts[pf.name] = sku_counts_single
        nm001_qty_scan = 0
        hb_qty_scan = 0
        pdf_nm001_counts[pf.name] = nm001_qty_scan
        pdf_hb_counts[pf.name] = hb_qty_scan

        actual_total = sum(q for s, q in sku_counts_single.items() if not s.startswith("MISSING_"))

        if qty_val == "":
            status = "无标注"
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
                status = f"不一致（差 {diff}）"

        pdf_item_list.append({
            "PDF文件": pf.name,
            "Item quantity": qty_val if qty_val != "" else "",
            "提取出货数量": actual_total,
            "状态": status
        })

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
            total_status = f"不一致（差 {total_actual - total_expected}）"
    else:
        total_status = "—"

    if not pdf_df.empty:
        pdf_df = pd.concat([pdf_df, pd.DataFrame({
            "PDF文件": ["合计"],
            "Item quantity": [total_expected],
            "提取出货数量": [total_actual],
            "状态": [total_status]
        })], ignore_index=True)

    st.dataframe(pdf_df, use_container_width=True)

    # —— 合并所有 PDF 的 SKU 数据 —— 
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

    # 缺 SKU 补录（支持直接填 bundle）
    if missing_lines:
        st.warning("以下出货记录缺 SKU，请补录：")
        manual_entries = {}
        for i, raw in enumerate(raw_missing):
            manual_entries[i] = st.text_input(f"“{raw}”的 SKU 是：", key=f"miss_{i}")
        if st.button("确认补录"):
            for i, sku in manual_entries.items():
                if sku and sku != "":
                    expand_bundle_or_single(sku.strip(), missing_lines[i], sku_counts_all)
            st.success("已将补录 SKU 添加进库存统计")

    # —— 换货处理（原逻辑）—— 
    if exchange_df is not None:
        if "原款式" in exchange_df.columns and "换货款式" in exchange_df.columns:
            for _, row in exchange_df.iterrows():
                original_sku = str(row["原款式"]).strip()
                new_sku = str(row["换货款式"]).strip()

                if sku_counts_all.get(original_sku):
                    qty = sku_counts_all.pop(original_sku)
                    sku_counts_all[new_sku] += qty

                stock_df.loc[stock_df["SKU编码"] == original_sku, stock_date_col] += 1
                stock_df.loc[stock_df["SKU编码"] == new_sku, stock_date_col] -= 1

            st.success("换货处理完成：已替换提取数量并调整库存（原款 +1 / 换货 -1）")
        else:
            st.warning("换货表中必须包含“原款式”和“换货款式”两列")

    # —— 合并库存（原逻辑）——
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

    st.subheader("库存更新结果")
    st.dataframe(summary_df, use_container_width=True)

    # 总对账
    total_sold = summary_df.loc["合计", "Sold Qty"]
    if total_expected and total_expected > 0:
        if total_sold == total_expected:
            st.success(f"提取成功：共 {total_sold} 件，与 PDF 标注汇总一致")
        elif ("NM001" not in stock_skus) and (total_sold + nm001_total_scan == total_expected):
            st.success(f"提取成功：共 {total_sold} 件（差 {nm001_total_scan} 件，均为 NM001，库存无此 SKU），与 PDF 标注汇总一致")
        elif (total_sold + hb_total_scan == total_expected):
            st.success(f"提取成功：共 {total_sold} 件（差 {hb_total_scan} 件，均为 Holiday Bunny，未被正则识别），与 PDF 标注汇总一致")
        elif ("NM001" not in stock_skus) and (total_sold + nm001_total_scan + hb_total_scan == total_expected):
            st.success(f"提取成功：共 {total_sold} 件（差 {nm001_total_scan + hb_total_scan} 件，其中 NM001 {nm001_total_scan}、Holiday Bunny {hb_total_scan}），与 PDF 标注汇总一致")
        else:
            st.error(f"提取数量 {total_sold} 与 PDF 标注汇总 {total_expected} 不一致")
    else:
        st.warning("未识别 PDF 中的 Item quantity")

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
