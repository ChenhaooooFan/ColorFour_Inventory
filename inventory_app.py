
import streamlit as st
import pandas as pd
import pdfplumber
import re
from collections import defaultdict
from io import BytesIO
from datetime import datetime
import os

st.set_page_config(page_title="ColorFour 库存系统", layout="centered")
st.title("📦 Nailvesta Inventory 系统")

# 上传文件
pdf_file = st.file_uploader("📤 上传 Picking List PDF", type=["pdf"])
csv_file = st.file_uploader("📥 上传库存表 CSV", type=["csv"])

if pdf_file and csv_file:
    st.success("✅ 文件上传成功，开始处理...")

    # 读取库存
    stock_df = pd.read_csv(csv_file)
    stock_df.columns = [col.strip() for col in stock_df.columns]
    stock_col = [col for col in stock_df.columns if re.match(r"\d{2}/\d{2}", col)]
    if not stock_col:
        st.error("❌ 未找到库存日期列（如 '06/03'）")
        st.stop()
    stock_date_col = stock_col[0]

    # 提取 PDF 中的 Item quantity
    with pdfplumber.open(pdf_file) as pdf:
        first_page_text = pdf.pages[0].extract_text()
        item_match = re.search(r'Item quantity[:：]?\s*(\d+)', first_page_text)
        expected_total = int(item_match.group(1)) if item_match else None

    # 提取 SKU 和数量
    sku_counts = defaultdict(int)
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            lines = page.extract_text().split("\n")
            for line in lines:
                match = re.search(r'([A-Z]{2,}\d{3}-[A-Z])\s+(\d+)\s+\d{9,}', line)
                if match:
                    sku, qty = match.group(1), int(match.group(2))
                    sku_counts[sku] += qty

    # 合并库存
    stock_df["Sold"] = stock_df["SKU编码"].map(sku_counts).fillna(0).astype(int)
    stock_df["New Stock"] = stock_df[stock_date_col] - stock_df["Sold"]
    summary_df = stock_df[["SKU编码", stock_date_col, "Sold", "New Stock"]].copy()
    summary_df.columns = ["SKU", "Old Stock", "Sold Qty", "New Stock"]
    summary_df.index += 1

    # 添加合计
    summary_df.loc["合计"] = [
        "—",
        summary_df["Old Stock"].sum(),
        summary_df["Sold Qty"].sum(),
        summary_df["New Stock"].sum()
    ]

    # 展示表格
    st.subheader("📊 库存更新结果")
    st.dataframe(summary_df, use_container_width=True)

    # 校验数量
    total_sold = summary_df.loc["合计", "Sold Qty"]
    if expected_total:
        if total_sold == expected_total:
            st.success(f"✅ 提取成功：共 {total_sold} 件，与 PDF 标注一致")
        else:
            st.error(f"❌ 提取数量 {total_sold} 与 PDF 标注 {expected_total} 不一致")
    else:
        st.warning("⚠️ 未识别 PDF 中的 Item quantity")

    # 可复制的 New Stock
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

    # 保存上传记录
    history_file = "upload_history.csv"
    new_record = {
        "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "PDF文件": pdf_file.name,
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
    st.info("请上传 Picking List PDF 和库存 CSV 以开始处理。")
