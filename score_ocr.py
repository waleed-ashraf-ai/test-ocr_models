"""Score OCR outputs in results/<page>/ against ground truth (character & word accuracy)."""
import re, sys
from pathlib import Path

GT = {
 "page_4": "وظيفة يوم العطلة كان «عاصم» و«ياسر» رفيقين بصف واحد، وصدقين جيدين، وكانوا متحمسين جدا لعطلة الصيف. أعطتهما معلمة الصف فروضا منزلية خلال العطلة، وقالت لهما: «استمتعا بالعطلة، لكن لا تنسيا أن تنهيا فروضكما المنزلية في وقتها».",
 "page_5": "في المنزل سأل «عاصم» والدته: «أمي هلا ساعدتني لو سمحت بالفروض المنزلية فى العطلة ؟ أريد أن أنهيها الآن وأستمتع بالعطلة». وافقت والدته وساعدته في إنهاء الفروض المنزلية.",
}

def norm(t):
    t = re.sub(r"<img>.*?</img>", " ", t, flags=re.S)
    t = re.sub(r"^.*\d+\s*\|\s*وظيفة يوم العطلة.*$", " ", t, flags=re.M)  # page footer
    t = re.sub(r"[ً-ْٰـ]", "", t)                      # harakat, tatweel
    t = re.sub(r"[أإآٱ]", "ا", t).replace("ى", "ي")
    t = re.sub(r"[^ء-ي\s]", " ", t)                                # punctuation/latin/markdown
    return " ".join(t.split())

def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]

sys.stdout.reconfigure(encoding="utf-8")
for page, gt in GT.items():
    g = norm(gt); rows = []
    for f in sorted(Path("results", page).glob("*.txt")):
        raw = f.read_text(encoding="utf-8")
        if raw.startswith("[ERROR]"):
            rows.append((f.stem, None, None)); continue
        h = norm(raw)
        cacc = max(0, 1 - lev(h, g) / len(g)); wacc = max(0, 1 - lev(h.split(), g.split()) / len(g.split()))
        rows.append((f.stem, cacc, wacc))
    rows.sort(key=lambda r: -1 if r[1] is None else r[1], reverse=True)
    print(f"\n{page}\n{'model':<20}{'char acc':>10}{'word acc':>10}")
    for m, c, w in rows:
        print(f"{m:<20}" + ("     error" * 2 if c is None else f"{c:>9.1%}{w:>10.1%}"))
