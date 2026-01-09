import streamlit as st
import pandas as pd
import re
import fitz  # PyMuPDF
from collections import defaultdict
from io import BytesIO
from datetime import datetime
import os

# ---------------- UI ----------------
st.set_page_config(page_title="NailVesta 库存系统", layout="centered")
st.title("ColorFour Inventory 系统（fitz 解析 + bundle 修复 + Mystery 抵扣期望 + Shopify Picking List 合并扣库存）")

# 上传文件（PDF 支持多选）
pdf_files = st.file_uploader("上传 Picking List PDF（可多选）", type=["pdf"], accept_multiple_files=True)
shopify_pdf_files = st.file_uploader("上传 Shopify Picking List PDF（可多选，会一起扣库存）", type=["pdf"], accept_multiple_files=True, key="shopify_pdf")
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

selected_shopify_pdfs = []
if shopify_pdf_files:
    selected_shopify_names = st.multiselect(
        "选择要参与统计的 Shopify Picking List PDF",
        options=[f.name for f in shopify_pdf_files],
        default=[f.name for f in shopify_pdf_files]
    )
    selected_shopify_pdfs = [f for f in shopify_pdf_files if f.name in selected_shopify_names]

# —— 按钮触发：达人换货 ——（仅保留“达人换货统计表（逐行一件）”）
if "show_exchange" not in st.session_state:
    st.session_state.show_exchange = False
if st.button("有达人换货吗？"):
    st.session_state.show_exchange = True

creator_swap_df = None  # 仅使用新格式（每行=1件）

if st.session_state.show_exchange:
    st.info("上传【达人换货统计表】（CSV/XLSX）：每行代表发货了 1 件，包含列 “原款式 SKU” 与 “发货款式SKU”。系统将逐行执行：提取原SKU -1、发货SKU +1；库存原SKU +1、发货SKU -1，并生成对账表。")
    creator_swap_file = st.file_uploader("上传达人换货统计表（每行=1件）", type=["csv", "xlsx"], key="creator_swap")
    if creator_swap_file:
        if creator_swap_file.name.endswith(".csv"):
            creator_swap_df = pd.read_csv(creator_swap_file)
        else:
            creator_swap_df = pd.read_excel(creator_swap_file)
        st.success("达人换货统计表（逐行一件）已上传")

# ---------- 规则与小工具 ----------
# 1) 含尺码的组合（bundle）：允许 1–4 件；部件可为标准 6 位或 NM001；结尾必须 -S/M/L
SKU_BUNDLE_WITH_SIZE = re.compile(r'((?:[A-Z]{3}\d{3}|NM001){1,4}-(?:S|M|L))', re.DOTALL)

# 2) 仅允许 NM001 无尺码（你的单独栏场景）
SKU_SOLO_NM_ONLY = re.compile(r'\bNM001\b')

# 数量：1–3 位整数，且后面不能紧跟字母（避免把 “3D” 的 3 当数量）
QTY_AFTER_SHORT = re.compile(r'\b([1-9]\d{0,2})(?![A-Za-z])\b')

# Shopify 常见写法：Qty: 2 / Quantity 2
QTY_LABEL_RE = re.compile(r'(?:Qty|Quantity)\s*[:：]?\s*([1-9]\d{0,2})', re.I)

# “Item quantity”
ITEM_QTY_RE = re.compile(r"Item\s+quantity[:：]?\s*(\d+)", re.I)

# Shopify Slip/Invoice 常见 “1 of 1”
SHOPIFY_SKU_RE = re.compile(r"\b((?:[A-Z]{3}\d{3}|NM001)-(?:S|M|L))\b")
SHOPIFY_OF_RE = re.compile(r"\b(\d+)\s+of\s+(\d+)\b", re.I)

def normalize_text(t: str) -> str:
    t = t.replace("\u00ad","").replace("\u200b","").replace("\u00a0"," ")
    t = t.replace("–","-").replace("—","-")
    return t

def fix_orphan_digit_before_size(txt: str) -> str:
    """
    修复形如：NPJ011NPX01\n5-M → NPJ011NPX015-M 的换行折断
    同时兼容 NM001 出现在组合中
    """
    pattern = re.compile(
        r'(?P<prefix>(?:[A-Z]{3}\d{3}|NM001){0,3}(?:[A-Z]{3}\d{2}|NM001))\s*[\r\n]+\s*(?P<d>\d)\s*-\s*(?P<size>[SML])'
    )
    def _join(m):
        prefix = m.group('prefix')
        d = m.group('d')
        size = m.group('size')
        if prefix.endswith('NM001'):
            return f"{prefix}-{size}"
        return f"{prefix}{d}-{size}"
    prev = None
    cur = txt
    while prev != cur:
        prev = cur
        cur = pattern.sub(_join, cur)
    return cur

def parse_code_parts(code: str):
    """将主体按 NM001 或 6位块切分，限制 1–4 件。全部成功返回列表，否则 None。"""
    parts, i, n = [], 0, len(code)
    while i < n:
        if code.startswith('NM001', i):
            parts.append('NM001'); i += 5; continue
        seg = code[i:i+6]
        if re.fullmatch(r'[A-Z]{3}\d{3}', seg):
            parts.append(seg); i += 6; continue
        return None
    return parts if 1 <= len(parts) <= 4 else None

def expand_bundle(counter: dict, sku_with_size: str, qty: int):
    """
    拆组合：
    返回 extra_units=(件数-1)*qty；mystery_units=NM001 次数 * qty
    """
    s = re.sub(r'\s+', '', sku_with_size)
    if '-' not in s:
        counter[s] += qty
        return 0, (qty if s == 'NM001' else 0)
    code, size = s.split('-', 1)
    parts = parse_code_parts(code)
    if parts:
        mystery_units = 0
        for p in parts:
            key = f"{p}-{size}"
            counter[key] += qty
            if p == 'NM001':
                mystery_units += qty
        extra = (len(parts) - 1) * qty
        return extra, mystery_units
    counter[s] += qty
    return 0, (qty if code == 'NM001' else 0)

def _read_pdf_text(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages_text = [p.get_text("text") for p in doc]
    text_raw = "\n".join(pages_text)
    text = normalize_text(text_raw)
    text = fix_orphan_digit_before_size(text)
    return text

def parse_pdf_with_fitz(file_bytes: bytes):
    """
    原 Picking List 解析：expected(Item quantity) + bundle + Mystery 抵扣
    """
    text_fixed = _read_pdf_text(file_bytes)

    m_total = ITEM_QTY_RE.search(text_fixed)
    expected_total_raw = int(m_total.group(1)) if m_total else 0

    sku_counts = defaultdict(int)
    bundle_extra_units = 0
    mystery_units = 0

    for m in SKU_BUNDLE_WITH_SIZE.finditer(text_fixed):
        token = re.sub(r'\s+', '', m.group(1))
        after = text_fixed[m.end(): m.end()+80]

        mq = QTY_LABEL_RE.search(after)
        if mq:
            qty = int(mq.group(1))
        else:
            mq2 = QTY_AFTER_SHORT.search(after)
            qty = int(mq2.group(1)) if mq2 else 1

        extra, myst = expand_bundle(sku_counts, token, qty)
        bundle_extra_units += extra
        mystery_units += myst

    for m in SKU_SOLO_NM_ONLY.finditer(text_fixed):
        next_chunk = text_fixed[m.end(): m.end()+3]
        if '-' in next_chunk:
            continue
        after = text_fixed[m.end(): m.end()+120]

        mq = QTY_LABEL_RE.search(after)
        if mq:
            qty = int(mq.group(1))
        else:
            mq2 = QTY_AFTER_SHORT.search(after)
            qty = int(mq2.group(1)) if mq2 else 1

        sku_counts['NM001'] += qty
        mystery_units += qty

    return expected_total_raw, sku_counts, bundle_extra_units, mystery_units

def parse_shopify_picking_pdf(file_bytes: bytes):
    """
    Shopify Picking List / Slip 模板：
    - SKU：ABC123-S/M/L 或 NM001-S/M/L
    - 数量：常见 “1 of 1 / 2 of 2”
    返回：sku_counts, total_units, order_ids
    """
    text = _read_pdf_text(file_bytes)

    order_ids = set(re.findall(r"\bOrder\s*#\s*([A-Za-z0-9_-]+)\b", text, flags=re.I))
    sku_counts = defaultdict(int)

    for m in SHOPIFY_SKU_RE.finditer(text):
        sku = m.group(1).strip()
        after = text[m.end(): m.end() + 80]
        mm = SHOPIFY_OF_RE.search(after)
        qty = int(mm.group(1)) if mm else 1
        sku_counts[sku] += qty

    total_units = int(sum(sku_counts.values()))
    return sku_counts, total_units, order_ids

# ---------- 主流程 ----------
if (selected_pdfs or selected_shopify_pdfs) and csv_file:
    st.success("文件上传成功，开始处理...")

    # 读取库存 CSV
    stock_df = pd.read_csv(csv_file)
    stock_df.columns = [col.strip() for col in stock_df.columns]

    if "SKU编码" not in stock_df.columns:
        st.error("库存表缺少列：SKU编码")
        st.stop()

    # ✅ 保持库存 CSV 原始顺序
    stock_df["_row_order"] = range(len(stock_df))

    # ✅ 禁止重复 SKU（避免重复扣减）
    dup = stock_df["SKU编码"].duplicated(keep=False)
    if dup.any():
        dup_list = (
            stock_df.loc[dup, "SKU编码"]
            .astype(str)
            .value_counts()
            .head(50)
            .to_dict()
        )
        st.error("检测到库存表存在重复 SKU编码，已停止运行（避免重复扣减库存）。请先修正库存表。")
        st.caption("前 50 个重复 SKU（SKU: 重复次数）：")
        st.json(dup_list)
        st.stop()

    stock_col = [col for col in stock_df.columns if re.match(r"\d{2}/\d{2}", col)]
    if not stock_col:
        st.error("未找到库存日期列（如 '06/03'）")
        st.stop()
    stock_date_col = stock_col[0]

    # =========================
    # A) 原 Picking List PDF：解析 + expected 对账（保留原逻辑）
    # =========================
    pdf_item_list = []
    pdf_sku_counts = {}
    per_pdf_expected = []
    per_pdf_extra = []
    per_pdf_actual = []
    per_pdf_mystery = []

    for pf in selected_pdfs:
        file_bytes = pf.read()
        expected_total_raw, sku_counts_single, bundle_extra_units, mystery_units = parse_pdf_with_fitz(file_bytes)
        pdf_sku_counts[pf.name] = sku_counts_single

        actual_total = sum(sku_counts_single.values())
        expected_bundle = expected_total_raw + bundle_extra_units
        expected_final = expected_bundle - mystery_units

        if expected_total_raw == 0:
            status = "未识别到 Item quantity"
        elif actual_total == expected_total_raw:
            status = "一致"
        elif actual_total == expected_bundle:
            status = f"与PDF标注不一致，但考虑 bundle 后相符（差 {actual_total - expected_total_raw}）"
        elif actual_total == expected_final:
            status = (
                f"与PDF标注不一致，但考虑 bundle 与 Mystery 抵扣后相符"
                f"（bundle +{bundle_extra_units}，Mystery −{mystery_units}，最终期望 {expected_final}）"
            )
        else:
            diff = actual_total - expected_total_raw
            status = (
                f"不一致（差 {diff}；bundle 影响 +{bundle_extra_units}；"
                f"Mystery 抵扣 −{mystery_units}；按理应为 {expected_final}）"
            )

        pdf_item_list.append({
            "PDF文件": pf.name,
            "Item quantity（原始）": expected_total_raw,
            "bundle 额外件数(+)": bundle_extra_units,
            "Mystery(NM001) 件数(−)": mystery_units,
            "调整后应为（bundle − Mystery）": expected_final,
            "提取出货数量": actual_total,
            "状态": status
        })

        per_pdf_expected.append(expected_total_raw)
        per_pdf_extra.append(bundle_extra_units)
        per_pdf_actual.append(actual_total)
        per_pdf_mystery.append(mystery_units)

    total_expected_raw = sum(per_pdf_expected) if per_pdf_expected else 0
    total_bundle_extra = sum(per_pdf_extra) if per_pdf_extra else 0
    total_mystery = sum(per_pdf_mystery) if per_pdf_mystery else 0
    total_expected_bundle = total_expected_raw + total_bundle_extra
    total_expected_final = total_expected_bundle - total_mystery
    total_actual_picking = sum(per_pdf_actual) if per_pdf_actual else 0

    if selected_pdfs:
        st.subheader("各 Picking List PDF 的 Item quantity 对账表（bundle − Mystery 口径）")
        pdf_df = pd.DataFrame(pdf_item_list)

        if not pdf_df.empty:
            if total_actual_picking == total_expected_raw:
                total_status = "一致"
            elif total_actual_picking == total_expected_bundle:
                total_status = f"与PDF标注不一致，但考虑 bundle 后相符（差 {total_actual_picking - total_expected_raw}）"
            elif total_actual_picking == total_expected_final:
                total_status = (
                    f"与PDF标注不一致，但考虑 bundle 与 Mystery 抵扣后相符"
                    f"（bundle +{total_bundle_extra}，Mystery −{total_mystery}，最终期望 {total_expected_final}）"
                )
            else:
                diff = total_actual_picking - total_expected_raw
                total_status = (
                    f"不一致（差 {diff}；bundle 影响 +{total_bundle_extra}；"
                    f"Mystery 抵扣 −{total_mystery}；按理应为 {total_expected_final}）"
                )

            total_row = {
                "PDF文件": "合计",
                "Item quantity（原始）": total_expected_raw,
                "bundle 额外件数(+)": total_bundle_extra,
                "Mystery(NM001) 件数(−)": total_mystery,
                "调整后应为（bundle − Mystery）": total_expected_final,
                "提取出货数量": total_actual_picking,
                "状态": total_status
            }
            pdf_df = pd.concat([pdf_df, pd.DataFrame([total_row])], ignore_index=True)

        st.dataframe(pdf_df, use_container_width=True)

    # 汇总 Picking List 的 SKU
    sku_counts_all = defaultdict(int)
    for counts in pdf_sku_counts.values():
        for sku, qty in counts.items():
            sku_counts_all[sku] += qty

    # =========================
    # B) Shopify Picking List：新增解析 + 合并扣库存
    # =========================
    shopify_audit_rows = []
    shopify_sku_counts_all = defaultdict(int)
    shopify_total_units = 0

    for pf in selected_shopify_pdfs:
        file_bytes = pf.read()
        s_counts, s_units, s_orders = parse_shopify_picking_pdf(file_bytes)

        for sku, qty in s_counts.items():
            shopify_sku_counts_all[sku] += qty

        shopify_total_units += s_units
        shopify_audit_rows.append({
            "Shopify PDF文件": pf.name,
            "识别订单数（Order #）": len(s_orders),
            "提取SKU种类数": len(s_counts),
            "提取件数": s_units
        })

    if selected_shopify_pdfs:
        st.subheader("Shopify Picking List 提取统计（将一起扣库存，不参与 Item quantity 对账）")
        shopify_df = pd.DataFrame(shopify_audit_rows)
        if not shopify_df.empty:
            total_row = {
                "Shopify PDF文件": "合计",
                "识别订单数（Order #）": int(shopify_df["识别订单数（Order #）"].sum()),
                "提取SKU种类数": int(shopify_df["提取SKU种类数"].sum()),
                "提取件数": int(shopify_df["提取件数"].sum()),
            }
            shopify_df = pd.concat([shopify_df, pd.DataFrame([total_row])], ignore_index=True)
        st.dataframe(shopify_df, use_container_width=True)

    # 合并到最终扣库存口径
    for sku, qty in shopify_sku_counts_all.items():
        sku_counts_all[sku] += qty

    # —— 达人换货统计表（逐行一件）：原款 -1、发货 +1；库存原款 +1、发货 -1 ——【带对账表】
    if creator_swap_df is not None:
        rename_map = {}
        cols = [c.strip() for c in creator_swap_df.columns]
        creator_swap_df.columns = cols
        if "原款式 SKU" in cols:
            rename_map["原款式 SKU"] = "原SKU"
        if "原款式SKU" in cols:
            rename_map["原款式SKU"] = "原SKU"
        if "发货款式SKU" in cols:
            rename_map["发货款式SKU"] = "新SKU"
        if "发货款式 SKU" in cols:
            rename_map["发货款式 SKU"] = "新SKU"
        creator_swap_df = creator_swap_df.rename(columns=rename_map)

        if {"原SKU", "新SKU"}.issubset(creator_swap_df.columns):
            sold_before = dict(sku_counts_all)

            applied_rows = 0
            missing_in_sold = 0
            log_rows = []

            for _, row in creator_swap_df.iterrows():
                original_sku = str(row["原SKU"]).strip()
                new_sku = str(row["新SKU"]).strip()

                found_in_sold = False
                if original_sku:
                    if sku_counts_all.get(original_sku, 0) > 0:
                        sku_counts_all[original_sku] -= 1
                        if sku_counts_all[original_sku] == 0:
                            del sku_counts_all[original_sku]
                        found_in_sold = True
                    else:
                        missing_in_sold += 1
                if new_sku:
                    sku_counts_all[new_sku] += 1

                if original_sku:
                    stock_df.loc[stock_df["SKU编码"] == original_sku, stock_date_col] += 1
                if new_sku:
                    stock_df.loc[stock_df["SKU编码"] == new_sku, stock_date_col] -= 1

                applied_rows += 1
                log_rows.append({
                    "原SKU": original_sku,
                    "新SKU": new_sku,
                    "原SKU是否在当日提取中找到": "是" if found_in_sold else "否",
                })

            swap_log_df = pd.DataFrame(log_rows)
            dec_counts = swap_log_df["原SKU"].value_counts().rename("原SKU减少次数") if not swap_log_df.empty else pd.Series(dtype=int)
            inc_counts = swap_log_df["新SKU"].value_counts().rename("新SKU增加次数") if not swap_log_df.empty else pd.Series(dtype=int)

            delta_sold = pd.concat([
                -dec_counts.rename("Delta"),
                inc_counts.rename("Delta")
            ], axis=0).groupby(level=0).sum().sort_index()

            idx = sorted(delta_sold.index.tolist())
            recon_df = pd.DataFrame({
                "Before Sold": [sold_before.get(k, 0) for k in idx],
                "Delta from Swap": [delta_sold.get(k, 0) for k in idx],
                "After Sold": [sku_counts_all.get(k, 0) for k in idx],
            }, index=idx)
            recon_df["OK?"] = recon_df["After Sold"] == (recon_df["Before Sold"] + recon_df["Delta from Swap"])

            stock_delta = pd.concat([
                dec_counts.rename("Stock Delta(原+1)"),
                (-inc_counts).rename("Stock Delta(新-1)")
            ], axis=0).groupby(level=0).sum().sort_values(ascending=False)
            stock_delta_df = stock_delta.to_frame(name="预期库存变动量（+原 / −新）")

            msg_tail = "" if missing_in_sold == 0 else f"；其中 {missing_in_sold} 行原SKU未在当日提取中找到（Sold 无法逐件 −1，仅做库存与新SKU +1）"
            st.success(f"达人换货处理完成（逐行一件）：共应用 {applied_rows} 行{msg_tail}")

            st.subheader("达人换货对账表")
            col1, col2 = st.columns(2)
            with col1:
                st.caption("Sold 变动对账（理论 Delta vs 应用前/后）")
                st.dataframe(recon_df, use_container_width=True)
            with col2:
                st.caption("库存预期变动（按达人换货累计）")
                st.dataframe(stock_delta_df, use_container_width=True)

            st.caption("达人换货明细（前100行）")
            st.dataframe(swap_log_df.head(100), use_container_width=True)

            total_dec = int(dec_counts.sum()) if not dec_counts.empty else 0
            total_inc = int(inc_counts.sum()) if not inc_counts.empty else 0
            st.info(f"理论上 Sold 变动：原SKU 合计 −{total_dec}，新SKU 合计 +{total_inc}。二者应相等（均为已应用行数 {applied_rows}）。")

        else:
            st.warning("达人换货统计表需要包含列：“原款式 SKU/原款式SKU” 与 “发货款式SKU/发货款式 SKU”")

    # —— 合并库存数据（最终扣库存：Picking + Shopify + 换货修正） ——
    stock_df["Sold"] = stock_df["SKU编码"].map(sku_counts_all).fillna(0).astype(int)
    stock_df["New Stock"] = stock_df[stock_date_col] - stock_df["Sold"]

    # ✅ 最终显示按库存 CSV 原始顺序
    stock_df = stock_df.sort_values("_row_order", kind="stable").reset_index(drop=True)

    summary_df = stock_df[["SKU编码", stock_date_col, "Sold", "New Stock"]].copy()
    summary_df.columns = ["SKU", "Old Stock", "Sold Qty", "New Stock"]
    summary_df.index += 1
    summary_df.loc["合计"] = [
        "—",
        summary_df["Old Stock"].sum(),
        summary_df["Sold Qty"].sum(),
        summary_df["New Stock"].sum()
    ]

    st.subheader("库存更新结果（Picking List + Shopify 一起扣库存）")
    st.dataframe(summary_df, use_container_width=True)

    # 对账提示：原 Picking List expected 对账不包含 Shopify
    total_sold_all = int(summary_df.loc["合计", "Sold Qty"])
    if selected_pdfs and total_expected_raw > 0:
        if total_actual_picking == total_expected_raw:
            st.success(f"Picking List 提取成功：共 {total_actual_picking} 件，与 PDF 标注汇总一致")
        elif total_actual_picking == total_expected_bundle:
            st.success(
                f"Picking List 提取成功：共 {total_actual_picking} 件。与 PDF 原始汇总不一致，但考虑 bundle 后相符（差 {total_actual_picking - total_expected_raw}）。"
            )
        elif total_actual_picking == total_expected_final:
            st.success(
                f"Picking List 提取成功：共 {total_actual_picking} 件。与 PDF 原始汇总不一致，但考虑 bundle 与 Mystery 抵扣后相符"
                f"（bundle +{total_bundle_extra}，Mystery −{total_mystery}）。"
            )
        else:
            st.error(
                f"Picking List 提取数量 {total_actual_picking} 与 PDF 标注汇总不一致；"
                f"原始: {total_expected_raw}，bundle 调整后: {total_expected_bundle}，"
                f"bundle−Mystery 调整后: {total_expected_final}（bundle +{total_bundle_extra}，Mystery −{total_mystery}）。"
            )

    if selected_shopify_pdfs:
        st.info(f"本次 Shopify Picking List 额外扣库存：{shopify_total_units} 件（不参与 Item quantity 对账）")
    st.info(f"本次最终扣库存总 Sold（Picking + Shopify + 换货修正）：{total_sold_all} 件")

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

    # 上传历史记录
    history_file = "upload_history.csv"
    new_record = {
        "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Picking List PDF文件": "; ".join([f.name for f in selected_pdfs]) if selected_pdfs else "",
        "Shopify PDF文件": "; ".join([f.name for f in selected_shopify_pdfs]) if selected_shopify_pdfs else "",
        "库存文件": csv_file.name,
        "PDF标注数量（原始）": total_expected_raw if total_expected_raw else "",
        "bundle 额外件数(+)": total_bundle_extra if total_bundle_extra else "",
        "Mystery（NM001）件数(−)": total_mystery if total_mystery else "",
        "PDF标注数量（最终，bundle−Mystery）": total_expected_final if total_expected_final else "",
        "Picking List 提取出货数量": total_actual_picking,
        "Shopify 提取出货数量": shopify_total_units,
        "最终扣库存 Sold": total_sold_all
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
    st.info("请上传 Picking List PDF（可选）/ Shopify Picking List PDF（可选） 和库存 CSV 以开始处理。")
