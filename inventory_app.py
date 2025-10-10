# ---------- Bundle 文本级扫描工具（新增） ----------
import re

# 1–4 件 bundle（允许跨行/空格）
SKU_BUNDLE = re.compile(r'((?:[A-Z]{3}\d{3}){1,4}-[SML])', re.DOTALL)
# SKU 后的 1–3位数量
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
    的换行折断。最后一个“3位数字”被切成“前2位在上一行 + 最后一位在下一行，再接 -SIZE”。
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
    扫描全文，统计 bundle 造成的“额外件数”，用于对账解释。
    返回：
      {
        "extra": 额外件数合计（比如 2件装每套多1、3件装每套多2…总和）,
        "by_parts": {2: 套数, 3: 套数, 4: 套数}
      }
    """
    text = normalize_text(full_text)
    text_fixed = fix_orphan_digit_before_size(text)
    extra = 0
    by_parts = {2: 0, 3: 0, 4: 0}

    for m in SKU_BUNDLE.finditer(text_fixed):
        raw = m.group(1)
        # 数量在右侧 50 字符里找第一个 1-3 位数字
        after = text_fixed[m.end(): m.end()+50]
        mq = QTY_AFTER.search(after)
        qty = int(mq.group(1)) if mq else 1

        code = re.sub(r'\s+', '', raw.split("-")[0])
        parts = len(code) // 6
        if parts in (2, 3, 4):
            extra += (parts - 1) * qty
            by_parts[parts] += qty

    return {"extra": extra, "by_parts": by_parts}
