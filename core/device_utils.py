"""
Device utilities
----------------
פותר את הערך של DEVICE ל-device אמיתי של torch, עם הודעת פידבק ברורה
למשתמש אם ביקש GPU ואין.

כללים:
  "auto"         -> cuda אם זמין, אחרת cpu.
  "gpu"          -> cuda אם זמין, אחרת cpu + הודעת אזהרה ברורה.
  "cpu"          -> cpu תמיד.
  "cuda", "cuda:N" -> cuda אם זמין, אחרת cpu + הודעת אזהרה ברורה.
"""

import torch


def resolve_device(requested: str) -> str:
    """
    מקבל מחרוזת מהמשתמש ומחזיר מחרוזת device של torch.

    ההחלטה נעשית פעם אחת כאן ומועברת לכל הניסויים. הפונקציה מדפיסה הודעה
    ברורה למשתמש על מה שנבחר בפועל.
    """
    req = (requested or "auto").lower()
    cuda_available = torch.cuda.is_available()

    if req == "cpu":
        _log("Device: CPU (מפורש).")
        return "cpu"

    if req == "auto":
        if cuda_available:
            name = torch.cuda.get_device_name(0)
            _log(f"Device: cuda:0 (auto) - '{name}'.")
            return "cuda:0"
        _log("Device: CPU (auto - לא נמצא GPU זמין).")
        return "cpu"

    if req == "gpu" or req.startswith("cuda"):
        if cuda_available:
            # אם ציין cuda:N, ננסה לכבד; אחרת cuda:0
            if req.startswith("cuda:"):
                try:
                    idx = int(req.split(":", 1)[1])
                    if idx < torch.cuda.device_count():
                        name = torch.cuda.get_device_name(idx)
                        _log(f"Device: {req} - '{name}'.")
                        return req
                    _log(f"אזהרה: בוקש {req} אבל יש רק "
                         f"{torch.cuda.device_count()} GPUs. משתמש ב-cuda:0.")
                    return "cuda:0"
                except ValueError:
                    pass
            name = torch.cuda.get_device_name(0)
            _log(f"Device: cuda:0 - '{name}'.")
            return "cuda:0"
        # ביקש GPU ואין - הודעה ברורה
        _log(
            "אזהרה: ביקשת GPU אך אין GPU זמין במערכת. "
            "המערכת תעבוד על CPU. אם זה מפתיע, בדוק שהתקנת CUDA "
            "ו-torch build עם cuda תואמים."
        )
        return "cpu"

    # ערך לא מוכר - ניפול ל-auto
    _log(f"אזהרה: ערך DEVICE='{requested}' לא מוכר. נעבור ל-auto.")
    return resolve_device("auto")


def _log(msg: str) -> None:
    print(f"[device] {msg}")
