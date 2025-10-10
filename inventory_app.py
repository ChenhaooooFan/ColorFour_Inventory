# =========================
# Bundle 拆分工具（1–4件）
# =========================
SEG_RE = re.compile(r"[A-Z]{3}\d{3}")

def expand_bundle_or_single(sku_with_size: str, qty: int, counter: dict):
    s = sku_with_size.strip().upper().replace("–", "-").replace("—", "-")
    if "-" not in s:
        counter[s] += qty
        return
    code, size = s.split("-", 1)
    code = re.sub(r"\s+", "", code)
    size = size.strip()

    if len(code) % 6 == 0 and 6 <= len(code) <= 24:
        segs = [code[i:i+6] for i in range(0, len(code), 6)]
        if all(SEG_RE.fullmatch(seg) for seg in segs):
            for seg in segs:
                counter[f"{seg}-{size}"] += qty
            return
    # 回退
    counter[f"{code}-{size}"] += qty


# =========================
# 词元化与“断裂缝合”
# =========================
SKU_FULL_RE = re.compile(r"^(?:[A-Z]{3}\d{3}){1,4}-[SML]$")
ITEMQ_RE = re.compile(r'Item\s+quantity[:：]?\s*(\d+)', re.I)
QTY_TOKEN = re.compile(r"^\d{1,3}$")
ORDER_TOKEN = re.compile(r"^\d{9,}$")

# 兜底（原文）匹配：正常与断裂
NORMAL_LINE_RE = re.compile(
    r"((?:[A-Z]{3}\d{3}){1,4})\s*-\s*([SML])\s+(\d{1,3})\s+\d{9,}",
    re.I
)
BROKEN_LINE_RE = re.compile(
    # 末段少 1 位数字，在下一行/下一词元出现
    r"((?:[A-Z]{3}\d{3}){0,3}[A-Z]{3}\d{2})\s*([0-9])\s*-\s*([SML])\s+(\d{1,3})\s+\d{9,}",
    re.I
)

def _norm_text(t: str) -> str:
    if not t: return ""
    return (t.replace("\u00ad", "")
             .replace("\u200b", "")
             .replace("\u00a0", " ")
             .replace("–", "-")
             .replace("—", "-"))

def _extract_tokens(fileobj):
    tokens = []
    with pdfplumber.open(fileobj) as pdf:
        for page in pdf.pages:
            words = page.extract_words(
                x_tolerance=2, y_tolerance=2,
                keep_blank_chars=False, use_text_flow=True
            )
            for w in words:
                txt = _norm_text(w["text"])
                for tk in txt.split():
                    if tk:
                        tokens.append(tk)
    return tokens

def _collect_prev_alnum(tokens, start_idx, max_chars=30):
    s = []
    i = start_idx
    while i >= 0:
        tk = tokens[i].upper()
        if re.fullmatch(r"[A-Z0-9]+", tk):
            s.append(tk)
            if sum(len(x) for x in s) >= max_chars:
                break
            i -= 1
        else:
            break
    s.reverse()
    return "".join(s), i

def _try_stitch_broken_sku(tokens, idx):
    cur = tokens[idx]
    m = re.fullmatch(r"^(\d)-([SML])$", cur.upper())
    size = None
    digit = None
    consumed = 1

    if m:
        digit = m.group(1); size = m.group(2)
    else:
        if re.fullmatch(r"^\d$", cur) and idx + 1 < len(tokens):
            nxt = tokens[idx + 1].upper()
            m2 = re.fullmatch(r"^-[SML]$", nxt)
            if m2:
                digit = cur; size = m2.group(0)[1]; consumed = 2

    if digit is None:
        return None, 0

    code_raw, _ = _collect_prev_alnum(tokens, idx - 1, max_chars=30)
    code_raw = code_raw.upper()
    if not code_raw or len(code_raw) % 6 != 5:
        return None, 0

    code = code_raw + digit
    if len(code) % 6 == 0 and 6 <= len(code) <= 24:
        segs = [code[i:i+6] for i in range(0, len(code), 6)]
        if all(SEG_RE.fullmatch(seg) for seg in segs):
            return f"{code}-{size}", consumed
    return None, 0


def parse_pdf_with_tokens(pf):
    """
    先 token 扫描（含缝合 '5-M'），再对整页原文做兜底匹配（正常 + 断裂）。
    兜底只补充 token 阶段未识别到的条目，避免重复。
    """
    item_q = ""
    try:
        with pdfplumber.open(pf) as pdf:
            first_text = _norm_text((pdf.pages[0].extract_text() or ""))
            m = ITEMQ_RE.search(first_text)
            if m:
                item_q = int(m.group(1))
    except Exception:
        pass

    try: pf.seek(0)
    except Exception: pass

    tokens = _extract_tokens(pf)
    n = len(tokens)
    sku_counts = defaultdict(int)

    # ---------- 第一阶段：token 扫描 + 缝合 ----------
    i = 0
    while i < n:
        tk = _norm_text(tokens[i].upper())

        if SKU_FULL_RE.fullmatch(tk):
            raw_sku = tk
            qty = None
            for j in range(i, min(i + 20, n)):
                if QTY_TOKEN.fullmatch(tokens[j]):
                    if any(ORDER_TOKEN.fullmatch(tokens[k]) for k in range(j + 1, min(j + 7, n))):
                        qty = int(tokens[j]); break
            if qty is None: qty = 1
            expand_bundle_or_single(raw_sku, qty, sku_counts)
            i += 1
            continue

        stitched, consumed = _try_stitch_broken_sku(tokens, i)
        if stitched:
            qty = None
            end_scan = min(i + 20, n)
            for j in range(i + consumed, end_scan):
                if QTY_TOKEN.fullmatch(tokens[j]):
                    if any(ORDER_TOKEN.fullmatch(tokens[k]) for k in range(j + 1, min(j + 7, n))):
                        qty = int(tokens[j]); break
            if qty is None: qty = 1
            expand_bundle_or_single(stitched, qty, sku_counts)
            i += consumed
            continue

        i += 1

    # ---------- 第二阶段：原文兜底（正常 + 断裂） ----------
    try: pf.seek(0)
    except Exception: pass

    try:
        with pdfplumber.open(pf) as pdf:
            for page in pdf.pages:
                raw = _norm_text(page.extract_text() or "")
                blob = re.sub(r"[ \t]+", " ", raw)  # 规整空格

                # 正常形态
                for m in NORMAL_LINE_RE.finditer(blob):
                    code = m.group(1).upper()
                    size = m.group(2).upper()
                    qty  = int(m.group(3))
                    sku  = f"{code}-{size}"
                    if sku not in sku_counts:  # 避免重复
                        expand_bundle_or_single(sku, qty, sku_counts)

                # 断裂形态（……NPX01   5-M  1  5771...）
                for m in BROKEN_LINE_RE.finditer(blob):
                    code_head = m.group(1).upper()
                    digit     = m.group(2)
                    size      = m.group(3).upper()
                    qty       = int(m.group(4))
                    full_code = code_head + digit
                    sku       = f"{full_code}-{size}"
                    if sku not in sku_counts:
                        expand_bundle_or_single(sku, qty, sku_counts)
    except Exception:
        pass

    return sku_counts, (item_q if item_q != "" else "")
