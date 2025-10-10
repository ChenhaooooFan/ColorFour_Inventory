import streamlit as st
import pandas as pd
import pdfplumber
import re
from collections import defaultdict
from io import BytesIO
from datetime import datetime
import os
from math import isclose

st.set_page_config(page_title="NailVesta 库存系统", layout="centered")
st.title("ColorFour Inventory 系统")

# ----------------------
# 上传区
# ----------------------
pdf_files = st.file_uploader("上传 Picking List PDF（可多选）", type=["pdf"], accept_multiple_files=True)
csv_file = st.file_uploader("上传库存表 CSV", type=["csv"])

selected_pdfs = []
if pdf_files:
    selected_names = st.multiselect(
        "选择要参与统计的 Picking List PDF",
        options=[f.name for f in pdf_files],
        default=[f.name for f in pdf_files]
    )
    selected_pdfs = [f for f in pdf_files if f.name in selected_names]

# —— 换货按钮 —— #
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

# ----------------------
# 通用：1–4 件 bundle 拆分器（保持你原有规则）
# ----------------------
def expand_bundle_or_single(sku_with_size: str, qty: int, counter: dict):
    """
    输入形如:
      - 单品: 'NPX005-S'
      - 2件: 'NPJ011NPX005-S'
      - 3件: 'NPJ011NPX005NPF001-S'
      - 4件: 'NPJ011NPX005NPF001NOX003-S'
    仅当 '-' 前部分长度为 6 的倍数（每段 3字母+3数字）且总长 6–24 时拆分。否则按原样累计。
    """
    sku_with_size = sku_with_size.strip()
    if "-" not in sku_with_size:
        counter[sku_with_size] += qty
        return

    code, size = sku_with_size.split("-", 1)
    code = code.strip()
    size = size.strip()

    if len(code) % 6 == 0 and 6 <= len(code) <= 24:
        parts = [code[i:i+6] for i in range(0, len(code), 6)]
        if all(re.fullmatch(r"[A-Z]{3}\d{3}", p) for p in parts):
            for p in parts:
                counter[f"{p}-{size}"] += qty
            return

    counter[sku_with_size] += qty

# ----------------------
# 重点升级①：字符级行重建 + 断裂修复
# ----------------------
def extract_lines_by_chars(pdf_path_or_file, y_tol=2.0, min_chars_per_line=2):
    """
    用 page.chars 逐字符聚类，按 y 坐标把同一行拼出来，再按 x 排序组合成真实文本。
    返回值：所有页的“行文本”列表（保持阅读顺序）
    """
    lines = []
    with pdfplumber.open(pdf_path_or_file) as pdf:
        for page in pdf.pages:
            # 收集字符
            chars = page.chars or []
            # 按 y 聚类
            rows = defaultdict(list)
            # 用近似相等分组（行高允许 y_tol 误差）
            ys = []
            for ch in chars:
                y = ch.get("top", ch.get("y0", 0))
                # 找到已存在的近似 y 组
                bucket = None
                for v in ys:
                    if abs(v - y) <= y_tol:
                        bucket = v
                        break
                if bucket is None:
                    ys.append(y)
                    bucket = y
                rows[bucket].append(ch)

            # y 从上到下（小到大）；同一行内按 x 从左到右
            for y in sorted(rows.keys()):
                row_chars = rows[y]
                row_chars.sort(key=lambda c: c.get("x0", c.get("x", 0)))
                text = "".join(c["text"] for c in row_chars)
                text = text.strip()
                if len(text) >= min_chars_per_line:
                    lines.append(text)
    return lines

def stitch_dangling_digit(lines):
    """
    重点升级②：修复你这个“最后一位数字换到下一行”的情况。
    规则：
      上一行以 ...[A-Z]{3}\d{2} 结尾，下一行形如 '^\d-[SML]$'
      → 把下一行的这一位数字拼回上一行，得到 ...\d{3}-S/M/L
    """
    if not lines:
        return lines
    stitched = []
    i = 0
    while i < len(lines):
        cur = lines[i]
        if i + 1 < len(lines):
            nxt = lines[i + 1]
            # 上一行以 ...[A-Z]{3}\d{2} 结尾（比如  NPX01）
            # 下一行是 5-M / 4-S 等（^\d-[SML]$）
            if re.search(r"(?:[A-Z]{3}\d{3}){0,3}[A-Z]{3}\d{2}$", cur) and re.fullmatch(r"\d-[SML]", nxt):
                # 拼回缺失的一位数字，形成 ...\d{3}-S/M/L
                fixed = cur + nxt  # 直接拼接即可： 'NPJ011NPX01' + '5-M' → 'NPJ011NPX015-M'
                stitched.append(fixed)
                i += 2
                continue
        stitched.append(cur)
        i += 1
    return stitched

def build_reconstructed_text(pdf_file):
    """综合重建后的行，返回一个统一的大文本，供正则一次性匹配。"""
    lines = extract_lines_by_chars(pdf_file, y_tol=2.0)
    lines = stitch_dangling_digit(lines)
    # 粘合回文本（保留换行，后续正则按行或跨行都能匹配）
    return "\n".join(lines)

# ----------------------
# 主流程
# ----------------------
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
    stock_skus = set(stock_df["SKU编码"].astype(str).str.strip())

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
        if re.search(r'\d{9,}', line):
            nums = re.findall(r'\b(\d{1,3})\b', line)
            if nums:
                return int(nums[0])
        return 0

    # 逐个 PDF 处理
    for pf in selected_pdfs:
        # 读取封面 item quantity
        with pdfplumber.open(pf) as pdf:
            first_page_text = pdf.pages[0].extract_text() or ""
            item_match = re.search(r'Item quantity[:：]?\s*(\d+)', first_page_text)
            qty_val = int(item_match.group(1)) if item_match else ""

        # 重建 & 修复后的完整文本
        full_text = build_reconstructed_text(pf)

        # SKU+Qty 匹配：bundle(1-4) + 尺码 + 数量 + 9位以上条码
        # 举例： NPJ011NPX015-M  1  5771336...
        sku_qty_pattern = re.compile(r'((?:[A-Z]{3}\d{3}){1,4}-[SML])\s+(\d{1,3})\s+\d{9,}')

        sku_counts_single = defaultdict(int)

        # 行内扫描：优先用重建后的逐行文本，匹配不到的也能让跨行匹配兜底
        lines = full_text.splitlines()
        matched_any = False
        for ln in lines:
            for m in sku_qty_pattern.finditer(ln):
                raw_sku, q = m.group(1), int(m.group(2))
                expand_bundle_or_single(raw_sku, q, sku_counts_single)
                matched_any = True

        # 兜底：全量文本再扫一次（防止极少数跨行仍未修复的情况）
        if not matched_any:
            for m in sku_qty_pattern.finditer(full_text):
                raw_sku, q = m.group(1), int(m.group(2))
                expand_bundle_or_single(raw_sku, q, sku_counts_single)

        # NM001 / Holiday Bunny 扫描（仅用于说明）
        nm001_qty_scan = 0
        hb_qty_scan = 0
        for ln in lines:
            m_nm = re.search(r'\bNM001\b\s+(\d{1,3})\s+\d{9,}', ln)
            if m_nm:
                nm001_qty_scan += int(m_nm.group(1))
            hb_qty_scan += _scan_holiday_bunny_qty(ln)

        pdf_nm001_counts[pf.name] = nm001_qty_scan
        pdf_hb_counts[pf.name] = hb_qty_scan

        # 实际提取出货数量（不含 MISSING_）
        actual_total = sum(q for s, q in sku_counts_single.items() if not s.startswith("MISSING_"))

        # 对账状态
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
                status = f"不一致（差 {diff}）" if hb_qty_scan == 0 else f"不一致（差 {diff}；Holiday Bunny 扫描到 {hb_qty_scan} 件）"

        pdf_item_list.append({
            "PDF文件": pf.name,
            "Item quantity": qty_val,
            "提取出货数量": actual_total,
            "状态": status
        })

        pdf_sku_counts[pf.name] = sku_counts_single

    # —— 对账表 —— #
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

    # —— 换货处理 —— #
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

    # —— 合并库存 —— #
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

    # 展示
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
            if hb_total_scan > 0:
                st.error(f"提取数量 {total_sold} 与 PDF 标注汇总 {total_expected} 不一致；其中 Holiday Bunny 扫描到 {hb_total_scan} 件")
            else:
                st.error(f"提取数量 {total_sold} 与 PDF 标注汇总 {total_expected} 不一致")
    else:
        st.warning("未识别 PDF 中的 Item quantity")

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

    # 历史记录
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
