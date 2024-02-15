# Extract all svgs from html files and saves to a directory called svg. creates the folder if it does not exist.

import os
import re
import sys
import shutil
from bs4 import BeautifulSoup
from typing import List
from pathlib import Path

def extract_svgs(html_dir: str, output_dir: str):
    html_files = [f for f in os.listdir(html_dir) if f.endswith('.html')]
    for html_file in html_files:
        with open(os.path.join(html_dir, html_file), 'r') as file:
            soup = BeautifulSoup(file, 'html.parser')
            svgs = soup.find_all('svg')
            for i, svg in enumerate(svgs):
                svg_file = f"C{i + 1}.svg"
                with open(os.path.join(output_dir, svg_file), 'w') as svg_file:
                    svg_file.write(str(svg))

if __name__ == "__main__":
    extract_svgs(sys.argv[1], sys.argv[2])
