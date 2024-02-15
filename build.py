import markdown2
import glob

template = open("template.html", "r").read()

md_files = glob.glob("*.md")


for md_file in md_files:
    text = open(md_file, "r").read()
    md = markdown2.markdown(text, extras=["fenced-code-blocks", "tables"])
    open(md_file.replace(".md", ".html"), "w").write(
        template.replace("{{MARKDOWN}}", md)
    )
