
# Compress file to min.js

build: compress
      @echo "Building markdown files"
      @python3 build.py

compress:
      @if [ ! -d assets/js/min ]; then \
            mkdir assets/js/min; \
      fi
      @echo "Compressing JS files"
      @for file in `find assets/js -name "*.js" ! -name "*.min.js" ! -name "echart*"`; do \
            echo "Compressing $file"; \
            uglifyjs --rename $file -o `echo $file | sed 's/\.js/\.min\.js/'`; \
            mv `echo $file | sed 's/\.js/\.min\.js/'` assets/js/min; \
      done
