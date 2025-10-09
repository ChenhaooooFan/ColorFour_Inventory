from datetime import datetime
import os

st.set_page_config(page_title="NailVesta 库存系统💗", layout="centered")
st.set_page_config(page_title="NailVesta 库存系统", layout="centered")
st.title("ColorFour Inventory 系统")

# 上传文件（PDF 支持多选）
@@ -42,27 +42,38 @@
exchange_df = pd.read_excel(exchange_file)
st.success("换货表已上传")

# —— Bundle 拆分工具函数（新增，最小改动）——
# —— Bundle 拆分工具函数（升级为通吃 1–4 件）——
def expand_bundle_or_single(sku_with_size: str, qty: int, counter: dict):
"""
    输入形如 'NPJ011NPX005-S' 或 'NPX005-S'。
    - 若为 Bundle：拆为 ['NPJ011-S', 'NPX005-S']，分别累计 qty
    - 若为单品：直接累计 qty
    注意：单个 SKU 前缀长度固定为 6（3字母+3数字）
    输入形如:
      - 单品: 'NPX005-S'
      - 2件: 'NPJ011NPX005-S'
      - 3件: 'NPJ011NPX005NPF001-S'
      - 4件: 'NPJ011NPX005NPF001NOX003-S'
    拆分规则：
      - 仅当 '-' 前部分长度为 6 的倍数，且在 [6, 24] 之间（每段 3字母+3数字）
      - 按每 6 位切片，生成 'XXXXXX-Size' 列表，分别累计相同 qty
    其他不合规字符串保持原样累计（与原逻辑一致，保证宽容性）
   """
sku_with_size = sku_with_size.strip()
if "-" not in sku_with_size:
        # 不合规编码，直接丢入（遵循原有宽松容错；但本工具主要服务规范 SKU）
counter[sku_with_size] += qty
return

code, size = sku_with_size.split("-", 1)
    if len(code) == 12:  # 两个 SKU 拼接
        sku1 = code[:6] + "-" + size
        sku2 = code[6:] + "-" + size
        counter[sku1] += qty
        counter[sku2] += qty
    else:
        counter[sku_with_size] += qty
    code = code.strip()
    size = size.strip()

    if len(code) % 6 == 0 and 6 <= len(code) <= 24:
        # 校验每段是否都是 3字母+3数字
        segments = [code[i:i+6] for i in range(0, len(code), 6)]
        if all(re.fullmatch(r"[A-Z]{3}\d{3}", seg) for seg in segments):
            for seg in segments:
                counter[f"{seg}-{size}"] += qty
            return

    # 回退：不满足规则时，按原样累计
    counter[sku_with_size] += qty

# —— 主流程 —— #
if selected_pdfs and csv_file:
@@ -104,10 +115,10 @@ def _scan_holiday_bunny_qty(line: str) -> int:
item_match = re.search(r'Item quantity[:：]?\s*(\d+)', first_page_text or "")
qty_val = int(item_match.group(1)) if item_match else ""

        # 2) 提取 SKU（最小改动：支持 Bundle）
        # 2) 提取 SKU（仅此处升级：支持 1–4 件 Bundle）
# 原来：([A-Z]{2,}\d{3}-[A-Z])\s+(\d+)\s+\d{9,}
        # 现在：单品或 Bundle（两段 6 位前缀可选） + 尺码（不限制为 SML，保持原版的 [A-Z] 宽松匹配）
        pattern = r'([A-Z]{3}\d{3}(?:[A-Z]{3}\d{3})?-[A-Z])\s+(\d+)\s+\d{9,}'
        # 升级：((?:[A-Z]{3}\d{3}){1,4}-[A-Z])\s+(\d+)\s+\d{9,}
        pattern = r'((?:[A-Z]{3}\d{3}){1,4}-[A-Z])\s+(\d+)\s+\d{9,}'
sku_counts_single = defaultdict(int)
with pdfplumber.open(pf) as pdf:
for page in pdf.pages:
@@ -116,7 +127,7 @@ def _scan_holiday_bunny_qty(line: str) -> int:
m = re.search(pattern, line)
if m:
raw_sku, qty = m.group(1), int(m.group(2))
                        # —— 仅此处变更：对 Bundle 做拆分入库 —— #
                        # —— 升级拆分：1–4 件通吃 —— #
expand_bundle_or_single(raw_sku, qty, sku_counts_single)
else:
# 无 SKU 的行，先按你原逻辑放到 MISSING_，稍后手动补录
@@ -228,7 +239,6 @@ def _scan_holiday_bunny_qty(line: str) -> int:
if st.button("确认补录"):
for i, sku in manual_entries.items():
if sku and sku != "":
                    # —— 新增：支持在补录里直接填写 Bundle（自动拆分入库）——
expand_bundle_or_single(sku.strip(), missing_lines[i], sku_counts_all)
st.success("已将补录 SKU 添加进库存统计")
