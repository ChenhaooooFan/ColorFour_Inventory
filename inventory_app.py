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

    # —— 每个 PDF：读取标注值、按原规则提取、另外专项扫描 NM001 / Holiday Bunny（仅用于对账说明）——
    pdf_item_list = []
    pdf_sku_counts = {}
    pdf_nm001_counts = {}
    pdf_hb_counts = {}

    def _scan_holiday_bunny_qty(line: str) -> int:
        """
        尽量从包含 Holiday Bunny 的行里抓到其数量：
        规则：
        - 同行若存在一个 <1000 的数字（疑似数量）且同样行里还出现一个 >=9位的数字（订单号/条码），
          则把 <1000 的数字视作件数（取首个匹配）。
        - 若存在形如 'Holiday Bunny ... <qty>  <longdigits>' 则优先按该模式抓取 qty。
        """
        if not re.search(r'holiday\s*bunny', line, flags=re.I):
            return 0
        # 优先：名称后到行尾的“数量 + 9位以上数字”
        m = re.search(r'holiday\s*bunny.*?(\d{1,3})\s+\d{9,}', line, flags=re.I)
        if m:
            return int(m.group(1))
        # 备选：同一行若包含 9位以上数字，再找一个 1-3位的数字作为数量
        has_long_digits = re.search(r'\d{9,}', line) is not None
        if has_long_digits:
            nums = re.findall(r'\b(\d{1,3})\b', line)
            if nums:
                # 取第一个 1-3位数字作为数量（常见格式数量靠前）
                return int(nums[0])
        return 0

    for pf in selected_pdfs:
        # 1) PDF 标注 Item quantity（保持原识别）
        with pdfplumber.open(pf) as pdf:
            first_page_text = pdf.pages[0].extract_text()
            item_match = re.search(r'Item quantity[:：]?\s*(\d+)', first_page_text or "")
            qty_val = int(item_match.group(1)) if item_match else ""

        # 2) 原规则提取（不改）
        sku_counts_single = defaultdict(int)
        with pdfplumber.open(pf) as pdf:
            for page in pdf.pages:
                lines = (page.extract_text() or "").split("\n")
                for line in lines:
                    m = re.search(r'([A-Z]{2,}\d{3}-[A-Z])\s+(\d+)\s+\d{9,}', line)
                    if m:
                        sku, qty = m.group(1), int(m.group(2))
                        sku_counts_single[sku] += qty
                    else:
                        # 无 SKU 的行，先按你原逻辑放到 MISSING_，稍后手动补录
                        m2 = re.search(r'^(\d{1,3})\s+\d{9,}', line.strip())
                        if m2:
                            sku_counts_single[f"MISSING_{len(pdf_item_list)}"] += int(m2.group(1))

        pdf_sku_counts[pf.name] = sku_counts_single

        # 3a) NM001 扫描（仅用于对账说明，不参与库存扣减）
        nm001_qty_scan = 0
        # 3b) Holiday Bunny 扫描（仅用于对账说明，不参与库存扣减）
        hb_qty_scan = 0

        with pdfplumber.open(pf) as pdf:
            for page in pdf.pages:
                lines = (page.extract_text() or "").split("\n")
                for line in lines:
                    # NM001
                    m_nm = re.search(r'\bNM001\b\s+(\d{1,3})\s+\d{9,}', line)
                    if m_nm:
                        nm001_qty_scan += int(m_nm.group(1))
                    # Holiday Bunny（大小写不敏感）
                    hb_qty_scan += _scan_holiday_bunny_qty(line)

        pdf_nm001_counts[pf.name] = nm001_qty_scan
        pdf_hb_counts[pf.name] = hb_qty_scan

        # 4) 计算该 PDF 的提取出货数量（不含 MISSING_）
        actual_total = sum(q for s, q in sku_counts_single.items() if not s.startswith("MISSING_"))

        # 5) 状态判定（考虑 NM001 / Holiday Bunny 扫描数量的解释）
        if qty_val == "":
            status = "无标注"
        else:
            diff = actual_total - qty_val
            # “严格一致”
            if diff == 0:
                status = "一致"
            # NM001 在库存中不存在且能解释全部差额
            elif ("NM001" not in stock_skus) and (actual_total + nm001_qty_scan == qty_val):
                status = f"一致（差 {nm001_qty_scan} 件，均为 NM001，库存无此 SKU）"
            # Holiday Bunny 也能解释差额（它通常未被正则识别为 SKU）
            elif (actual_total + hb_qty_scan == qty_val):
                status = f"一致（差 {hb_qty_scan} 件，均为 Holiday Bunny，未被正则识别）"
            # NM001 + Holiday Bunny 合计能解释差额
            elif ("NM001" not in stock_skus) and (actual_total + nm001_qty_scan + hb_qty_scan == qty_val):
                status = f"一致（差 {nm001_qty_scan + hb_qty_scan} 件，其中 NM001 {nm001_qty_scan}、Holiday Bunny {hb_qty_scan}）"
            else:
                # 仍不一致则指出具体差额，并在存在 Holiday Bunny 时提示它的扫描数量
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

    # 如果扫描到了 Holiday Bunny，但它没有被计入 SKU 统计，提醒可用“缺 SKU 补录”来录入
    if hb_total_scan > 0:
        st.info(f"提示：扫描到 Holiday Bunny 共 {hb_total_scan} 件。如果未自动识别，请在下面“缺 SKU 补录”输入其对应的 SKU 后确认。")

    # —— 合并所有 PDF 的 SKU 数据（保持原逻辑）——
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
        st.warning("以下出货记录缺 SKU，请补录：")
        manual_entries = {}
        for i, raw in enumerate(raw_missing):
            manual_entries[i] = st.text_input(f"“{raw}”的 SKU 是：", key=f"miss_{i}")
        if st.button("确认补录"):
            for i, sku in manual_entries.items():
                if sku and sku != "":
                    sku_counts_all[sku.strip()] += missing_lines[i]
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

    # 总对账（复用 NM001 / Holiday Bunny 解释）
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
