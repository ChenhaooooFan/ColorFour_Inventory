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

# —— Bundle 拆分工具函数（通吃 1–4 件）——
BUNDLE_SEG = re.compile(r"[A-Z]{3}\d{3}")

def _expand_bundle_or_single(sku_with_size: str, qty: int, counter: dict):
    """
    入参形如:
      - 单品: 'NPX005-S'
      - 2件: 'NPJ011NPX005-S'
      - 3件: 'NPJ011NPX005NPF001-S'
      - 4件: 'NPJ011NPX005NPF001NOX003-S'
    拆分规则：
      - '-' 前整段长度为 6 的倍数，且每段为 3字母+3数字
    """
    s = sku_with_size.strip().upper().replace("–", "-").replace("—", "-")
    if "-" not in s:
        counter[s] += qty
        return
    code, size = s.split("-", 1)
    code = re.sub(r"\s+", "", code)
    size = size.strip()

    if len(code) % 6 == 0 and 6 <= len(code) <= 24:
        segs = [code[i:i+6] for i in range(0, len(code), 6)]
        if all(BUNDLE_SEG.fullmatch(seg) for seg in segs):
            for seg in segs:
                counter[f"{seg}-{size}"] += qty
            return

    # 回退：不满足规则则按原样累计
    counter[f"{code}-{size}"] += qty

# —— 文本标准化 —— 
def _norm_text(t: str) -> str:
    if not t:
        return ""
    t = (t.replace("\u00ad", "")
           .replace("\u200b", "")
           .replace("\u00a0", " ")
           .replace("–", "-")
           .replace("—", "-"))
    return t

# —— 从 PDF 抽取 tokens（词元）按阅读顺序拼成列表 —— 
def _extract_tokens(fileobj) -> list[str]:
    tokens = []
    # pdfplumber 的 extract_words 能按文本流顺序产出词
    with pdfplumber.open(fileobj) as pdf:
        for page in pdf.pages:
            text0 = _norm_text(page.extract_text() or "")
            # 先存一下第一页里 Item quantity 用于对账
            words = page.extract_words(
                x_tolerance=2, y_tolerance=2,
                keep_blank_chars=False, use_text_flow=True
            )
            for w in words:
                # 按空白继续细分，得到更细的 token 粒度
                for tk in _norm_text(w["text"]).split():
                    if tk:
                        tokens.append(tk)
    return tokens

# —— 词元流扫描：识别 SKU（允许跨行跨词）+ 向前搜索数量 —— 
SKU_WIN = re.compile(r"((?:[A-Z]{3}\d{3}){1,4}-[SML])")
QTY_TOKEN = re.compile(r"^\d{1,3}$")
ORDER_TOKEN = re.compile(r"^\d{9,}$")
ITEMQ_RE = re.compile(r'Item\s+quantity[:：]?\s*(\d+)', re.I)

def parse_pdf_with_tokens(pf) -> tuple[dict, int]:
    """
    返回 (sku_counts_single, item_quantity_mark)
    - 在 token 流里用滑窗匹配 SKU（1–4 件 bundle）
    - SKU 命中后，向后 20 个 token 寻找：数量(<=3位) + 9位以上长数字（订单号）
      找不到就按 1 件
    """
    # 先抓 Item quantity
    item_q = ""
    try:
        with pdfplumber.open(pf) as pdf:
            first_text = _norm_text((pdf.pages[0].extract_text() or ""))
            m = ITEMQ_RE.search(first_text)
            if m:
                item_q = int(m.group(1))
    except Exception:
        pass

    try:
        pf.seek(0)
    except Exception:
        pass
    tokens = _extract_tokens(pf)

    sku_counts = defaultdict(int)

    n = len(tokens)
    i = 0
    while i < n:
        # 取一个窗口把若干 token 连起来，以覆盖被分词/换行的 SKU
        # 8~10 个 token 足以覆盖 4 段 SKU + "-S"
        end = min(i + 10, n)
        buf = "".join(tokens[i:end])
        buf = _norm_text(buf).upper()

        m = SKU_WIN.search(buf)
        if not m:
            i += 1
            continue

        raw_sku = m.group(1)  # 形如 NPJ011NPX015-M / NPX005-S

        # 向后找“数量 + 9位订单号”
        qty = None
        for j in range(i, min(i + 20, n)):
            if QTY_TOKEN.fullmatch(tokens[j]):
                # 之后 1~6 个 token 里是否有 9位以上数字
                if any(ORDER_TOKEN.fullmatch(tk) for tk in tokens[j+1:j+7]):
                    qty = int(tokens[j])
                    break
        if qty is None:
            qty = 1

        _expand_bundle_or_single(raw_sku, qty, sku_counts)

        # i 前进一点，避免重复识别同一窗口；但不要跨太多以免漏
        i += 3

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

    # —— 每个 PDF：解析（token 流方案，解决 bundle 断行） —— 
    pdf_item_list = []
    pdf_sku_counts = {}
    pdf_nm001_counts = {}
    pdf_hb_counts = {}

    def _scan_holiday_bunny_qty(line: str) -> int:
        return 0  # 此分支保留接口，具体文件里一般不会命中；保留不影响

    for pf in selected_pdfs:
        # 用 token 流解析
        sku_counts_single, qty_val = parse_pdf_with_tokens(pf)
        pdf_sku_counts[pf.name] = sku_counts_single

        # NM001 / HB 扫描（仅用于对账说明，可保留为 0）
        nm001_qty_scan = 0
        hb_qty_scan = 0
        pdf_nm001_counts[pf.name] = nm001_qty_scan
        pdf_hb_counts[pf.name] = hb_qty_scan

        # 计算该 PDF 的提取出货数量（不含 MISSING_）
        actual_total = sum(q for s, q in sku_counts_single.items() if not s.startswith("MISSING_"))

        # 状态判定
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
                if hb_qty_scan > 0:
                    status = f"不一致（差 {diff}；Holiday Bunny 扫描到 {hb_qty_scan} 件）"
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

    # 缺 SKU 补录（保持原逻辑 + 支持 Bundle 补录）
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
