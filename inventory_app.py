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

# 上传文件（改：PDF 支持多选）
pdf_files = st.file_uploader("📤 上传 Picking List PDF（可多选）", type=["pdf"], accept_multiple_files=True)
csv_file = st.file_uploader("📥 上传库存表 CSV", type=["csv"])

# 新增：从多选 PDF 里再选择要参与统计的文件（默认全选）
selected_pdfs = []
if pdf_files:
    selected_names = st.multiselect(
        "✅ 选择要参与统计的 Picking List PDF",
        options=[f.name for f in pdf_files],
        default=[f.name for f in pdf_files]
    )
    selected_pdfs = [f for f in pdf_files if f.name in selected_names]

# 新增换货表上传功能（保持原逻辑不变）
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

# 触发处理条件由“单个pdf”改为“至少一个被选中的pdf”
if selected_pdfs and csv_file:
    st.success("✅ 文件上传成功，开始处理...")

    # 读取库存 CSV（保持原逻辑）
    stock_df = pd.read_csv(csv_file)
    stock_df.columns = [col.strip() for col in stock_df.columns]
    stock_col = [col for col in stock_df.columns if re.match(r"\d{2}/\d{2}", col)]
    if not stock_col:
        st.error("❌ 未找到库存日期列（如 '06/03'）")
        st.stop()
    stock_date_col = stock_col[0]

    # 识别 PDF 中 Item quantity（改：逐个 PDF 识别并汇总）
    pdf_item_list = []
    expected_total = None
    total_expected_sum = 0
    found_any_expected = False
    for pf in selected_pdfs:
        with pdfplumber.open(pf) as pdf:
            first_page_text = pdf.pages[0].extract_text()
            item_match = re.search(r'Item quantity[:：]?\s*(\d+)', first_page_text or "")
            qty_val = int(item_match.group(1)) if item_match else ""
            pdf_item_list.append({"PDF文件": pf.name, "Item quantity": qty_val})
            if item_match:
                total_expected_sum += int(item_match.group(1))
                found_any_expected = True
    if found_any_expected:
        expected_total = total_expected_sum  # 汇总后的期望数量

    # 新增：显示每个 PDF 的单独 Item quantity 小表
    st.subheader("📄 各 PDF 的 Item quantity")
    st.dataframe(pd.DataFrame(pdf_item_list), use_container_width=True)

    # 提取 SKU + 数量 & 未识别行（改：遍历多个 PDF，累加结果；提取规则不变）
    sku_counts = defaultdict(int)
    missing_lines = []
    raw_missing = []

    for pf in selected_pdfs:
        with pdfplumber.open(pf) as pdf:
            for page in pdf.pages:
                lines = (page.extract_text() or "").split("\n")
                for line in lines:
                    match = re.search(r'([A-Z]{2,}\d{3}-[A-Z])\s+(\d+)\s+\d{9,}', line)
                    if match:
                        sku, qty = match.group(1), int(match.group(2))
                        sku_counts[sku] += qty
                    else:
                        match_loose = re.search(r'^(\d{1,3})\s+\d{9,}', line.strip())
                        if match_loose:
                            qty = int(match_loose.group(1))
                            missing_lines.append(qty)
                            raw_missing.append(line.strip())

    # 缺 SKU 补录（保持原逻辑）
    if missing_lines:
        st.warning("⚠️ 以下出货记录缺 SKU，请补录：")
        manual_entries = {}
        for i, raw in enumerate(raw_missing):
            manual_entries[i] = st.text_input(f"❓“{raw}”的 SKU 是：", key=f"miss_{i}")

        if st.button("✅ 确认补录"):
            for i, sku in manual_entries.items():
                if sku and sku != "":
                    sku_counts[sku.strip()] += missing_lines[i]
            st.success("✅ 已将补录 SKU 添加进库存统计")

    # 处理换货：替换 SKU（保持原逻辑）
    if exchange_df is not None:
        if "原款式" in exchange_df.columns and "换货款式" in exchange_df.columns:
            for _, row in exchange_df.iterrows():
                original_sku = str(row["原款式"]).strip()
                new_sku = str(row["换货款式"]).strip()
                if sku_counts.get(original_sku):
                    qty = sku_counts.pop(original_sku)
                    sku_counts[new_sku] += qty
            st.success("✅ 换货处理完成：已用换货款式替代原款式")
        else:
            st.warning("⚠️ 换货表中必须包含“原款式”和“换货款式”两列")

    # 合并库存数据（保持原逻辑）
    stock_df["Sold"] = stock_df["SKU编码"].map(sku_counts).fillna(0).astype(int)
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

    # 展示表格（保持原逻辑）
    st.subheader("📊 库存更新结果")
    st.dataframe(summary_df, use_container_width=True)

    total_sold = summary_df.loc["合计", "Sold Qty"]
    # 对账（改：与“多个 PDF 的期望数量汇总”对比；其余逻辑保持）
    if expected_total is not None:
        if total_sold == expected_total:
            st.success(f"✅ 提取成功：共 {total_sold} 件，与所选 PDF 的 Item quantity 汇总一致")
        else:
            st.error(f"❌ 提取数量 {total_sold} 与 PDF 标注汇总 {expected_total} 不一致")
    else:
        st.warning("⚠️ 未识别 PDF 中的 Item quantity")

    # 可复制 New Stock（保持原逻辑）
    st.subheader("📋 一键复制 New Stock")
    new_stock_text = "\n".join(summary_df.iloc[:-1]["New Stock"].astype(str).tolist())
    st.code(new_stock_text, language="text")

    # 下载 Excel（保持原逻辑）
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        summary_df.to_excel(writer, index_label="序号")
    st.download_button(
        label="📥 下载库存更新表 Excel",
        data=output.getvalue(),
        file_name="库存更新结果.xlsx"
    )

    # 上传记录保存（改：记录多个文件名；其余保持）
    history_file = "upload_history.csv"
    new_record = {
        "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "PDF文件": "; ".join([f.name for f in selected_pdfs]),
        "库存文件": csv_file.name,
        "PDF标注数量": expected_total if expected_total else "",
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
