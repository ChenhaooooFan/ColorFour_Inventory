import streamlit as st
import pandas as pd
import pdfplumber
import re
from collections import defaultdict
from io import BytesIO
from datetime import datetime
import os

st.set_page_config(page_title="NailVesta 库存系统", layout="centered")
st.title("📦 ColorFour Inventory 系统")

# 上传文件（PDF 支持多选）
pdf_files = st.file_uploader("📤 上传 Picking List PDF（可多选）", type=["pdf"], accept_multiple_files=True)
csv_file = st.file_uploader("📥 上传库存表 CSV", type=["csv"])

# 从多选 PDF 里再选择要参与统计的文件（默认全选）
selected_pdfs = []
if pdf_files:
    selected_names = st.multiselect(
        "✅ 选择要参与统计的 Picking List PDF",
        options=[f.name for f in pdf_files],
        default=[f.name for f in pdf_files]
    )
    selected_pdfs = [f for f in pdf_files if f.name in selected_names]

# 新增换货表上传功能
exchange_mode = st.radio("今天是否有达人换货？", ["否", "是"])
exchange_df = None
if exchange_mode == "是":
    exchange_file = st.file_uploader("📎 上传换货记录截图（支持 Excel 或 CSV）", type=["csv", "xlsx"])
    if exchange_file:
        if exchange_file.name.endswith(".csv"):
            exchange_df = pd.read_csv(exchange_file)
        else:
            exchange_df = pd.read_excel(exchange_file)
        st.success("✅ 换货表已上传")

if selected_pdfs and csv_file:
    st.success("✅ 文件上传成功，开始处理...")

    # 读取库存 CSV
    stock_df = pd.read_csv(csv_file)
    stock_df.columns = [col.strip() for col in stock_df.columns]
    stock_col = [col for col in stock_df.columns if re.match(r"\d{2}/\d{2}", col)]
    if not stock_col:
        st.error("❌ 未找到库存日期列（如 '06/03'）")
        st.stop()
    stock_date_col = stock_col[0]

    # 存放每个 PDF 的标注数量 & 实际提取数量
    pdf_item_list = []
    pdf_sku_counts = {}  # {pdf文件名: {sku: qty}}

    for pf in selected_pdfs:
        # 1. 获取 PDF 标注数量
        with pdfplumber.open(pf) as pdf:
            first_page_text = pdf.pages[0].extract_text()
            item_match = re.search(r'Item quantity[:：]?\s*(\d+)', first_page_text or "")
            qty_val = int(item_match.group(1)) if item_match else ""

        # 2. 提取该 PDF 的 SKU+数量
        sku_counts_single = defaultdict(int)
        with pdfplumber.open(pf) as pdf:
            for page in pdf.pages:
                lines = (page.extract_text() or "").split("\n")
                for line in lines:
                    match = re.search(r'([A-Z]{2,}\d{3}-[A-Z])\s+(\d+)\s+\d{9,}', line)
                    if match:
                        sku, qty = match.group(1), int(match.group(2))
                        sku_counts_single[sku] += qty
                    else:
                        match_loose = re.search(r'^(\d{1,3})\s+\d{9,}', line.strip())
                        if match_loose:
                            # 缺SKU的数量，这里先加到一个临时SKU里
                            sku_counts_single[f"MISSING_{len(pdf_item_list)}"] += int(match_loose.group(1))

        pdf_sku_counts[pf.name] = sku_counts_single

        # 3. 对账状态
        actual_total = sum(qty for sku, qty in sku_counts_single.items() if not sku.startswith("MISSING_"))
        if qty_val == "":
            status = "⚠️ 无标注"
        elif actual_total == qty_val:
            status = "✅ 一致"
        else:
            status = f"❌ 不一致（差 {actual_total - qty_val}）"

        pdf_item_list.append({
            "PDF文件": pf.name,
            "Item quantity": qty_val,
            "提取出货数量": actual_total,
            "状态": status
        })

    # 显示小表 + 合计行
    st.subheader("📄 各 PDF 的 Item quantity 对账表")
    pdf_df = pd.DataFrame(pdf_item_list)
    if not pdf_df.empty:
        total_expected = pdf_df["Item quantity"].replace("", 0).astype(int).sum()
        total_actual = pdf_df["提取出货数量"].sum()
        total_status = "✅ 一致" if total_expected == total_actual else f"❌ 不一致（差 {total_actual - total_expected}）"
        total_row = pd.DataFrame({
            "PDF文件": ["合计"],
            "Item quantity": [total_expected],
            "提取出货数量": [total_actual],
            "状态": [total_status]
        })
        pdf_df = pd.concat([pdf_df, total_row], ignore_index=True)
    st.dataframe(pdf_df, use_container_width=True)

    # —— 下面合并所有 PDF 的 SKU 数据（保持原逻辑）——
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

    # 缺 SKU 补录（保持原逻辑）
    if missing_lines:
        st.warning("⚠️ 以下出货记录缺 SKU，请补录：")
        manual_entries = {}
        for i, raw in enumerate(raw_missing):
            manual_entries[i] = st.text_input(f"❓“{raw}”的 SKU 是：", key=f"miss_{i}")
        if st.button("✅ 确认补录"):
            for i, sku in manual_entries.items():
                if sku and sku != "":
                    sku_counts_all[sku.strip()] += missing_lines[i]
            st.success("✅ 已将补录 SKU 添加进库存统计")

    # 处理换货（保持原逻辑）
    if exchange_df is not None:
        if "原款式" in exchange_df.columns and "换货款式" in exchange_df.columns:
            for _, row in exchange_df.iterrows():
                original_sku = str(row["原款式"]).strip()
                new_sku = str(row["换货款式"]).strip()
                if sku_counts_all.get(original_sku):
                    qty = sku_counts_all.pop(original_sku)
                    sku_counts_all[new_sku] += qty
            st.success("✅ 换货处理完成：已用换货款式替代原款式")
        else:
            st.warning("⚠️ 换货表中必须包含“原款式”和“换货款式”两列")

    # 合并库存数据
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

    # 展示表格
    st.subheader("📊 库存更新结果")
    st.dataframe(summary_df, use_container_width=True)

    # 总对账
    total_sold = summary_df.loc["合计", "Sold Qty"]
    if total_expected and total_expected > 0:
        if total_sold == total_expected:
            st.success(f"✅ 提取成功：共 {total_sold} 件，与 PDF 标注汇总一致")
        else:
            st.error(f"❌ 提取数量 {total_sold} 与 PDF 标注汇总 {total_expected} 不一致")
    else:
        st.warning("⚠️ 未识别 PDF 中的 Item quantity")

    # 可复制 New Stock
    st.subheader("📋 一键复制 New Stock")
    new_stock_text = "\n".join(summary_df.iloc[:-1]["New Stock"].astype(str).tolist())
    st.code(new_stock_text, language="text")

    # 下载 Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        summary_df.to_excel(writer, index_label="序号")
    st.download_button(
        label="📥 下载库存更新表 Excel",
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

    st.subheader("📝 上传历史记录")
    st.dataframe(history_df, use_container_width=True)

else:
    st.info("请上传一个或多个 Picking List PDF 和库存 CSV 以开始处理。")
