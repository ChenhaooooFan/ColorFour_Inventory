import streamlit as st
import pandas as pd
import re
import fitz  # PyMuPDF
from collections import defaultdict
from io import BytesIO
from datetime import datetime
import os

st.set_page_config(page_title="NailVesta 库存系统", layout="centered")
st.title("ColorFour Inventory 系统")

# -----------------------
# 1) 文件上传
# -----------------------
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


# -----------------------
# 2) 正则与工具（基于 PyMuPDF 的文本）
# -----------------------

# 允许 1–4 件 bundle（可跨行/空格）
SKU_BUNDLE = re.compile(r'((?:[A-Z]{3}\d{3}){1,4}-[SML])', re.DOTALL)
# SKU 后 1–3 位数量（在右侧若干字符内）
QTY_AFTER  = re.compile(r'\b([1-9]\d{0,2})\b')

def normalize_text(t: str) -> str:
    # 去软连字符/零宽空格/不间断空格 + 标准化破折号
    t = t.replace("\u00ad","").replace("\u200b","").replace("\u00a0"," ")
    t = t.replace("–","-").replace("—","-")
    return t

def fix_orphan_digit_before_size(txt: str) -> str:
    """
    修复形如：
        NPJ011NPX01\n5-M  → NPJ011NPX015-M
    的换行折断（最后 3 位被分成 2+1 并跨行）。
    这里只修复「最后一段是 2位数字 + 换行 + 1位数字 + -[SML]」的场景。
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

def scan_bundle_extra_qty_from_text(full_text: str) -> dict:
    """
    扫描整份 PDF 文本，统计 bundle 造成的“额外件数”，仅用于对账解释：
      2件装：每套多 1
      3件装：每套多 2
      4件装：每套多 3
    返回：
      {
        "extra": 额外件数合计,
        "by_parts": {2: 套数, 3: 套数, 4: 套数}
      }
    """
    text = normalize_text(full_text)
    text_fixed = fix_orphan_digit_before_size(text)

    extra = 0
    by_parts = {2: 0, 3: 0, 4: 0}

    for m in SKU_BUNDLE.finditer(text_fixed):
        raw = m.group(1)
        code = re.sub(r'\s+', '', raw.split('-')[0])  # 去空白，拿到前缀拼接段
        after = text_fixed[m.end(): m.end()+50]
        mq = QTY_AFTER.search(after)
        qty = int(mq.group(1)) if mq else 1

        parts = len(code) // 6
        if parts in (2, 3, 4):
            extra += (parts - 1) * qty
            by_parts[parts] += qty

    return {"extra": extra, "by_parts": by_parts}

def expand_bundle_or_single(sku_with_size: str, qty: int, counter: dict):
    """
    输入形如:
      - 单品: 'NPX005-S'
      - 2件: 'NPJ011NPX005-S'
      - 3件: 'NPJ011NPX005NPF001-S'
      - 4件: 'NPJ011NPX005NPF001NOX003-S'
    拆分规则：
      - 仅当 '-' 前部分长度为 6 的倍数，且在 [6, 24] 之间（每段 3字母+3数字）
      - 按每 6 位切片，生成 'XXXXXX-Size' 列表，分别累计相同 qty
    其他不合规字符串保持原样累计（宽容性）
    """
    sku_with_size = sku_with_size.strip()
    if "-" not in sku_with_size:
        counter[sku_with_size] += qty
        return

    code, size = sku_with_size.split("-", 1)
    code = code.strip()
    size = size.strip()

    if len(code) % 6 == 0 and 6 <= len(code) <= 24:
        segments = [code[i:i+6] for i in range(0, len(code), 6)]
        if all(re.fullmatch(r"[A-Z]{3}\d{3}", seg) for seg in segments):
            for seg in segments:
                counter[f"{seg}-{size}"] += qty
            return

    # 回退：不满足规则则按原样累计
    counter[sku_with_size] += qty


# -----------------------
# 3) 主流程（PyMuPDF 抽取）
# -----------------------
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

    # —— 每个 PDF：读取标注值、按原规则提取、专项扫描 NM001 / Holiday Bunny（仅用于对账说明）
    pdf_item_list = []
    pdf_sku_counts = {}
    pdf_nm001_counts = {}
    pdf_hb_counts = {}

    # （新增）bundle 解释之合计
    total_bundle_extra = 0
    bundle_sum_by_parts = {2: 0, 3: 0, 4: 0}

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
        # 读取 PDF（PyMuPDF）
        raw = pf.read()
        doc = fitz.open(stream=raw, filetype="pdf")

        # —— 整份文本：用于对账的 bundle 解释、Item quantity
        full_text = ""
        for page in doc:
            full_text += page.get_text("text") + "\n"
        full_text = normalize_text(full_text)
        full_text = fix_orphan_digit_before_size(full_text)

        # 1) Item quantity
        item_match = re.search(r'Item\s+quantity[:：]?\s*(\d+)', full_text, re.I)
        qty_val = int(item_match.group(1)) if item_match else ""

        # 1.5) （新增）bundle 文本级扫描（仅用于对账解释）
        bundle_scan = scan_bundle_extra_qty_from_text(full_text)
        bundle_extra = bundle_scan["extra"]
        bundle_by_parts = bundle_scan["by_parts"]
        total_bundle_extra += bundle_extra
        for k in (2, 3, 4):
            bundle_sum_by_parts[k] += bundle_by_parts[k]

        # 2) 提取 SKU（**行级**扫描：PyMuPDF 每页文本 splitlines）
        pattern = r'([A-Z]{3}\d{3}(?:[A-Z]{3}\d{3})?-[SML])\s+(\d+)\s+\d{9,}'
        sku_counts_single = defaultdict(int)

        # 再次遍历页，逐行匹配 SKU 与数量
        doc2 = fitz.open(stream=raw, filetype="pdf")
        for page in doc2:
            lines = (page.get_text("text") or "").splitlines()
            for line in lines:
                line = normalize_text(line)
                # 行内可能没有修复的断行场景，但我们主要靠上一轮全文修复做 bundle 解释；
                # 这里仍按常规一行 "SKU  数量  条码" 匹配
                m = re.search(pattern, line)
                if m:
                    raw_sku, q = m.group(1), int(m.group(2))
                    expand_bundle_or_single(raw_sku, q, sku_counts_single)
                else:
                    # 备用：只有“数量  条码”的行，先记录缺 SKU
                    m2 = re.search(r'^(\d{1,3})\s+\d{9,}$', line.strip())
                    if m2:
                        sku_counts_single[f"MISSING_{len(pdf_item_list)}"] += int(m2.group(1))

        pdf_sku_counts[pf.name] = sku_counts_single

        # 3a) NM001 扫描（仅用于对账说明，不参与库存扣减）
        nm001_qty_scan = 0
        # 3b) Holiday Bunny 扫描（仅用于对账说明，不参与库存扣减）
        hb_qty_scan = 0

        doc3 = fitz.open(stream=raw, filetype="pdf")
        for page in doc3:
            lines = (page.get_text("text") or "").splitlines()
            for line in lines:
                line = normalize_text(line)
                m_nm = re.search(r'\bNM001\b\s+(\d{1,3})\s+\d{9,}', line)
                if m_nm:
                    nm001_qty_scan += int(m_nm.group(1))
                hb_qty_scan += _scan_holiday_bunny_qty(line)

        pdf_nm001_counts[pf.name] = nm001_qty_scan
        pdf_hb_counts[pf.name] = hb_qty_scan

        # 4) 计算该 PDF 的提取出货数量（不含 MISSING_）
        actual_total = sum(q for s, q in sku_counts_single.items() if not s.startswith("MISSING_"))

        # 5) 状态判定（含 bundle 解释）
        if qty_val == "":
            status = "无标注"
        else:
            diff = actual_total - qty_val
            if diff == 0:
                status = "一致"
            elif actual_total + bundle_extra == qty_val:
                explain_parts = []
                if bundle_by_parts[2]:
                    explain_parts.append(f"2件装 ×{bundle_by_parts[2]}")
                if bundle_by_parts[3]:
                    explain_parts.append(f"3件装 ×{bundle_by_parts[3]}")
                if bundle_by_parts[4]:
                    explain_parts.append(f"4件装 ×{bundle_by_parts[4]}")
                bundle_explain = "、".join(explain_parts) if explain_parts else "bundle"
                status = f"一致（包含 {bundle_explain}；按 {bundle_extra:+d} 件差额修正）"
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

        pdf_item_list.append({
            "PDF文件": pf.name,
            "Item quantity": qty_val,
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

    # 总状态（含 bundle 解释）
    if total_expected > 0:
        if total_actual == total_expected:
            total_status = "一致"
        elif total_actual + total_bundle_extra == total_expected:
            parts_line = []
            if bundle_sum_by_parts[2]:
                parts_line.append(f"2件装 ×{bundle_sum_by_parts[2]}")
            if bundle_sum_by_parts[3]:
                parts_line.append(f"3件装 ×{bundle_sum_by_parts[3]}")
            if bundle_sum_by_parts[4]:
                parts_line.append(f"4件装 ×{bundle_sum_by_parts[4]}")
            parts_str = "、".join(parts_line) if parts_line else "bundle"
            total_status = f"一致（包含 {parts_str}；按 {total_bundle_extra:+d} 件差额修正）"
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

    if hb_total_scan > 0:
        st.info(f"提示：扫描到 Holiday Bunny 共 {hb_total_scan} 件。如果未自动识别，请在下面“缺 SKU 补录”输入其对应的 SKU 后确认。")

    # —— 合并所有 PDF 的 SKU 数据（保持原逻辑，计数由前文已拆分）——
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

    # 缺 SKU 补录（保持原逻辑 + 支持 Bundle 补录）
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

    # —— 换货处理：提取替换 + 库存调整（每行原款 +1、换货 -1） —— 
    if exchange_df is not None:
        if "原款式" in exchange_df.columns and "换货款式" in exchange_df.columns:
            for _, row in exchange_df.iterrows():
                original_sku = str(row["原款式"]).strip()
                new_sku = str(row["换货款式"]).strip()

                # 1) 替换提取数量（原款 → 换货）
                if sku_counts_all.get(original_sku):
                    qty = sku_counts_all.pop(original_sku)
                    sku_counts_all[new_sku] += qty

                # 2) 直接修改库存（对应日期列）：原款 +1、换货 -1
                stock_df.loc[stock_df["SKU编码"] == original_sku, stock_date_col] += 1
                stock_df.loc[stock_df["SKU编码"] == new_sku, stock_date_col] -= 1

            st.success("换货处理完成：已替换提取数量并调整库存（原款 +1 / 换货 -1）")
        else:
            st.warning("换货表中必须包含“原款式”和“换货款式”两列")

    # —— 合并库存数据（保持原逻辑）——
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

    # 总对账（沿用上方解释）
    total_sold = summary_df.loc["合计", "Sold Qty"]
    if total_expected and total_expected > 0:
        if total_sold == total_expected:
            st.success(f"提取成功：共 {total_sold} 件，与 PDF 标注汇总一致")
        elif total_sold + total_bundle_extra == total_expected:
            parts_line = []
            if bundle_sum_by_parts[2]:
                parts_line.append(f"2件装 ×{bundle_sum_by_parts[2]}")
            if bundle_sum_by_parts[3]:
                parts_line.append(f"3件装 ×{bundle_sum_by_parts[3]}")
            if bundle_sum_by_parts[4]:
                parts_line.append(f"4件装 ×{bundle_sum_by_parts[4]}")
            parts_str = "、".join(parts_line) if parts_line else "bundle"
            st.success(f"提取成功：共 {total_sold} 件（包含 {parts_str}；按 {total_bundle_extra:+d} 件差额修正），与 PDF 标注汇总一致")
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
