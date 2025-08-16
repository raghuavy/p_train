from datetime import datetime, timezone
from pathlib import Path
import re

README = Path("README.md")
start = "<!--START_SECTION:status-->"
end   = "<!--END_SECTION:status-->"

now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
new_block = f"{start}\n_Last run: {now}_\n{end}\n"

text = README.read_text(encoding="utf-8")

pattern = re.compile(rf"{re.escape(start)}.*?{re.escape(end)}", re.S)
updated = pattern.sub(new_block, text)

if updated != text:
    README.write_text(updated, encoding="utf-8")
    print("README updated")
else:
    print("No changes")
