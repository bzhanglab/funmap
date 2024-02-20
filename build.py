import markdown2
import glob
from datetime import datetime



template = open("template.html", "r").read()

md_files = glob.glob("*.md")


for md_file in md_files:
    text = open(md_file, "r").read()
    md = markdown2.markdown(text, extras=["fenced-code-blocks", "tables"])
    open(md_file.replace(".md", ".html"), "w").write(
        template.replace("{{MARKDOWN}}", md)
    )


with open("index.html", "r") as f:
    index = f.read()
    start_block = "<!--DATEBLOCK START-->"
    end_block = "<!--DATEBLOCK END-->"
    start_index = index.find(start_block) + len(start_block)
    end_index = index.find(end_block)
    index = index[:start_index] + f"<p>Last updated: <time datetime=\"{datetime.today().strftime('%Y-%m-%d')}\">{datetime.today().strftime('%B %d, %Y')}</time></p>" + index[end_index:]

with open("index.html", "w") as f:
    f.write(index)