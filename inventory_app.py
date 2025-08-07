import streamlit as st
import pandas as pd
import pdfplumber
import re
from collections import defaultdict
from io import BytesIO
from datetime import datetime
import os

# ---------------------- 基础设置 ----------------------
st.set_page_config(page_title="NailVesta 库存系统", layout="centered")
st.title("ColorFour Inventory 系统")

# session 状态：记忆缺SKU补录输入 & 历史记录缓存
if "manual_missing" not in st.session_state:
    st.session_state.manual_missing = {}
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------- 上传区 ----------------------
# 支持多选 PDF
pdf_files = st.file_uploader("上传 Picking List PDF（可多选）", type=["pdf"], accept_multiple_files=True)
csv_file = st.file_uploader("上传库存表 CSV", type=["csv"])

# 换货表上传（可选）
exchange_mode = st.radio("今天是否有达人换货？", ["否", "是"])
exchange_df = None
if exchange_mode == "是":
    exchange_file = st.file_uploader("上传换货记录（CSV 或 XLSX）", type=["csv", "xlsx"])
    if exchange_file:
        if exchange_file.name.endswith(".csv"):
            exchange_df = pd.read_csv(exchange_file)
        else:
            exchange_df = pd.read_excel(exchange_file)
        st.success("换货表已上传")

# 让用户从已上传 PDF 中选择参与统计的文件
selected_pdfs = []
if pdf_files:
    selected_names = st.multiselect(
        "选择要参与统计的 Picking List PDF",
        options=[f.name for f in pdf_files],
        default=[f.name for f in pdf_files]
    )
    selected_pdfs = [f for f in pdf_files if f.name in selected_names]

# ---------------------- 工具函数与正则 ----------------------
SKU_PATTERN = re.compile(r"\b([A-Z]{2,}[A-Z]?\d{3}(?:-[A-Z])?)\b", re.I)
LINE_ITEM_PATTERN = re.compile(r"([A-Z]{2,}[A-Z]?\d{3}(?:-[A-Z])?)\s+(\d{1,3})\s+\d{9,}", re.I)
LOOSE_QTY_PATTERN = re.compile(r"^(\d{1,3})\s+\d{9,}")

def norm_sku(s: str) -> str:
    return str(s).strip().upper().replace(" ", "")

def parse_item_quantity(text: str):
    if not text:
        return None
    patterns = [
        r"Item(?:s)?\s*quantity[:：]?\s*(\d+)",
        r"Total\s*Items[:：]?\s*(\d+)",
        r"Total\s*Quantity[:：]?\s*(\d+)"
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.I)
        if m:
            return int(m.group(1))
    return None

def safe_int(x):
    try:
        if pd.isna(x):
            return 0
        return int(str(x).replace(",", "").strip())
    except:
        return 0

# ---------------------- 主流程 ----------------------
if selected_pdfs and csv_file:
    st.success("文件上传成功，开始处理...")

    # 读取库存 CSV
    stock_df = pd.read_csv(csv_file)
    stock_df.columns = [str(col).strip() for col in stock_df.columns]

    # 定位 SKU 列
    if "SKU编码" in stock_df.columns:
        sku_col = "SKU编码"
    else:
        sku_col_candidates = [c for c in stock_df.columns if "SKU" in c.upper()]
        if sku_col_candidates:
            sku_col = sku_col_candidates[0]
            st.info(f"未找到“SKU编码”列，自动使用“{sku_col}”。")
        else:
            st.error("未找到 SKU 列（如“SKU编码”）。")
            st.stop()

    stock_df[sku_col] = stock_df[sku_col].astype(str).map(norm_sku)

    # 自动识别库存日期列（如 06/03）
    stock_date_cols = [c for c in stock_df.columns if re.fullmatch(r"\d{2}/\d{2}", str(c).strip())]
    if not stock_date_cols:
        st.error("未找到库存日期列（如 '06/03'）。")
        st.stop()
    stock_date_col = stock_date_cols[0]
    stock_df[stock_date_col] = stock_df[stock_date_col].map(safe_int)

    # 逐个 PDF 抽取：期望数量与出货明细
    expected_total_list = []  # 每个文件的 Item quantity
    sku_counts = defaultdict(int)
    missing_lines = []  # 只有数量但缺SKU
    raw_missing = []    # 展示给用户补录，标注来源文件名

    for pf in selected_pdfs:
        # 读第一页文本识别 Item quantity
        with pdfplumber.open(pf) as pdf:
            first_page_text = pdf.pages[0].extract_text() or ""
            exp = parse_item_quantity(first_page_text)
            expected_total_list.append({"文件": pf.name, "Item quantity": exp if exp is not None else ""})

        # 遍历每页，提取 SKU+数量；同时收集缺SKU的行
        with pdfplumber.open(pf) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                # 按行匹配
                for line in text.split("\n"):
                    line = line.strip()
                    m = LINE_ITEM_PATTERN.search(line)
                    if m:
                        sku, qty = norm_sku(m.group(1)), int(m.group(2))
                        sku_counts[sku] += qty
                    else:
                        m2 = LOOSE_QTY_PATTERN.search(line)
                        if m2:
                            qty = int(m2.group(1))
                            missing_lines.append(qty)
                            raw_missing.append(f"{pf.name}｜{line}")

                # 再尝试表格抽取（有些PDF是表格结构）
                try:
                    tables = page.extract_tables() or []
                    for tbl in tables:
                        for row in tbl:
                            if not row:
                                continue
                            row_str = " ".join([str(x) for x in row if pd.notna(x)])
                            m = LINE_ITEM_PATTERN.search(row_str)
                            if m:
                                sku, qty = norm_sku(m.group(1)), int(m.group(2))
                                sku_counts[sku] += qty
                except Exception:
                    pass

    # 缺 SKU 补录
    if missing_lines:
        st.warning("检测到缺少 SKU 的出货行，请补录：")
        for i, raw in enumerate(raw_missing):
            key = f"miss_{i}"
            if key not in st.session_state.manual_missing:
                st.session_state.manual_missing[key] = ""
            st.session_state.manual_missing[key] = st.text_input(
                f"“{raw}”的 SKU 是：",
                value=st.session_state.manual_missing[key],
                key=key
            )

        if st.button("确认补录/更新"):
            for i, sku in enumerate(list(st.session_state.manual_missing.values())):
                if str(sku).strip():
                    sku_counts[norm_sku(sku)] += missing_lines[i]
            st.success("补录已合并到库存统计。")

    # 处理换货：支持列 原款式、换货款式、数量(可选)
    if exchange_df is not None:
        required_cols = {"原款式", "换货款式"}
        if not required_cols.issubset(set(exchange_df.columns)):
            st.warning("换货表需要包含列：原款式、换货款式（可选：数量）")
        else:
            for _, row in exchange_df.iterrows():
                original_sku = norm_sku(row["原款式"])
                new_sku = norm_sku(row["换货款式"])
                qty_in_sheet = safe_int(row["数量"]) if "数量" in exchange_df.columns else None

                if qty_in_sheet and qty_in_sheet > 0:
                    replace_qty = min(qty_in_sheet, sku_counts.get(original_sku, 0))
                    if replace_qty > 0:
                        sku_counts[original_sku] -= replace_qty
                        sku_counts[new_sku] += replace_qty
                else:
                    if sku_counts.get(original_sku):
                        q = sku_counts.pop(original_sku)
                        sku_counts[new_sku] += q
            st.success("换货处理完成。")

    # 合并库存数据
    sold_series = pd.Series(sku_counts, name="Sold").astype(int)
    stock_df = stock_df.merge(
        sold_series.rename("Sold"),
        left_on=sku_col,
        right_index=True,
        how="left"
    )
    stock_df["Sold"] = stock_df["Sold"].fillna(0).astype(int)

    # 计算新库存，下限为 0
    stock_df["New Stock"] = (stock_df[stock_date_col] - stock_df["Sold"]).clip(lower=0)

    # 结果汇总表
    summary_df = stock_df[[sku_col, stock_date_col, "Sold", "New Stock"]].copy()
    summary_df.columns = ["SKU", "Old Stock", "Sold Qty", "New Stock"]
    summary_df.index += 1

    total_row = pd.DataFrame({
        "SKU": ["—"],
        "Old Stock": [summary_df["Old Stock"].sum()],
        "Sold Qty": [summary_df["Sold Qty"].sum()],
        "New Stock": [summary_df["New Stock"].sum()],
    }, index=["合计"])
    summary_df = pd.concat([summary_df, total_row], axis=0)

    # 展示库存更新结果
    st.subheader("库存更新结果")
    st.dataframe(summary_df, use_container_width=True)

    # 对账：逐文件 + 汇总
    st.subheader("PDF 对账（逐文件）")
    exp_df = pd.DataFrame(expected_total_list)
    st.dataframe(exp_df, use_container_width=True)

    total_expected = exp_df["Item quantity"].replace("", 0).astype(int).sum()
    total_sold = int(summary_df.loc["合计", "Sold Qty"])

    if total_expected > 0:
        if total_sold == total_expected:
            st.success(f"提取成功：共 {total_sold} 件，与 {len(selected_pdfs)} 个 PDF 标注汇总一致。")
        else:
            diff = total_sold - total_expected
            msg = "多提取" if diff > 0 else "少提取"
            st.error(f"提取数量 {total_sold} 与 PDF 标注汇总 {total_expected} 不一致（{msg} {abs(diff)} 件）。")
            # 诊断：未匹配到库存表的 SKU、缺SKU行提示
            unmatched_in_stock = [s for s in sku_counts.keys() if s not in set(stock_df[sku_col])]
            if unmatched_in_stock:
                st.warning(
                    f"下列SKU在库存表中未找到：{', '.join(unmatched_in_stock[:30])}"
                    + (" …" if len(unmatched_in_stock) > 30 else "")
                )
            if missing_lines:
                st.info("存在缺SKU的出货行，请确认补录是否完整。")
    else:
        st.warning("未识别到 PDF 中的 Item quantity（或全部为空）。")

    # 可复制 New Stock（排除合计行）
    st.subheader("一键复制 New Stock")
    new_stock_text = "\n".join(summary_df.iloc[:-1]["New Stock"].astype(str).tolist())
    st.code(new_stock_text, language="text")

    # 下载 Excel/CSV
    st.subheader("下载库存更新表")
    output_xlsx = BytesIO()
    with pd.ExcelWriter(output_xlsx, engine='openpyxl') as writer:
        summary_df.to_excel(writer, index_label="序号")
    st.download_button(
        label="下载 Excel",
        data=output_xlsx.getvalue(),
        file_name="库存更新结果.xlsx"
    )

    output_csv = summary_df.copy()
    csv_bytes = output_csv.to_csv(index_label="序号").encode("utf-8-sig")
    st.download_button(
        label="下载 CSV",
        data=csv_bytes,
        file_name="库存更新结果.csv"
    )

    # 上传记录保存（写入本地 CSV，附带多文件名）
    history_file = "upload_history.csv"
    new_record = {
        "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "PDF文件": "; ".join([f.name for f in selected_pdfs]),
        "库存文件": csv_file.name,
        "PDF标注数量(汇总)": total_expected if total_expected else "",
        "提取出货数量(汇总)": total_sold
    }

    if os.path.exists(history_file):
        try:
            history_df = pd.read_csv(history_file)
        except Exception:
            history_df = pd.DataFrame(columns=list(new_record.keys()))
        history_df = pd.concat([history_df, pd.DataFrame([new_record])], ignore_index=True)
    else:
        history_df = pd.DataFrame([new_record])

    history_df.to_csv(history_file, index=False)

    st.subheader("上传历史记录")
    st.dataframe(history_df, use_container_width=True)

else:
    st.info("请上传一个或多个 Picking List PDF 和库存 CSV 以开始处理。")
