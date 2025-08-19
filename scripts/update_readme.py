from datetime import datetime, timezone
from pathlib import Path
import re

README = Path("README.md")
START = "<!--START_SECTION:status-->"
END   = "<!--END_SECTION:status-->"

now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
block = f"{START}\n_Last run: {now}_\n{END}\n"

if not README.exists():
    # Bootstrap a README so the first run creates a commit
    README.write_text(f"# Project\n\n{block}", encoding="utf-8")
    print("README created")
else:
    text = README.read_text(encoding="utf-8")
    pattern = re.compile(rf"{re.escape(START)}.*?{re.escape(END)}", re.S)

    if pattern.search(text):
        updated = pattern.sub(block, text)
    else:
        # Append the status block if markers are missing
        updated = text.rstrip() + "\n\n" + block

    if updated != text:
        README.write_text(updated, encoding="utf-8")
        print("README updated")
    else:
        print("No changes")
