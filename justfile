check:
      @echo "Running biome checks"
      @biome check assets/js/clique.js
      @biome check assets/js/dag.js
      @biome check assets/js/funmap.js
      @biome check assets/js/dense.js

js-ci:
      @echo "Running biome CI"
      @biome ci assets/js/clique.js
      @biome ci assets/js/dag.js
      @biome ci assets/js/funmap.js
      @biome ci assets/js/dense.js

build: js-ci compress
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
