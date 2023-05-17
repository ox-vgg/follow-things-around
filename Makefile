## Author: David Miguel Susano Pinto <carandraug+dev@gmail.com>
##
## Copying and distribution of this file, with or without modification,
## are permitted in any medium without royalty provided the copyright
## notice and this notice are preserved.  This file is offered as-is,
## without any warranty.

JQ ?= jq
JUPYTEXT ?= jupytext


## Ideally, we'd have instructions in the .py files so that jupytext
## includes then in the .ipynb files.  But seems that jupytext only
## has support to filter/include stuff in the other direction (when
## generating .py from .ipynb).  So we hack it this way.
##
## This is approach is particularly problematic because the
## "collapsed_sections" field but that's ok for now.

define NOTEBOOK_METADATA
{ \
  "colab": { \
    "provenance": [], \
    "collapsed_sections": [ \
      "7c1gbeVrFHUu" \
    ], \
    "toc_visible": true \
  }, \
  "kernelspec": { \
    "name": "python3", \
    "display_name": "Python 3" \
  }, \
  "language_info": { \
    "name": "python" \
  }, \
  "accelerator": "GPU", \
  "gpuClass": "standard" \
}
endef


.PHONY: notebooks


notebooks: \
  tracking.ipynb

# In addition to fix the notebook metadata, jupytext also adds the
# lines_to_next_cell in some cell metadata (not sure why).  We remove
# them.

%.ipynb: %.py
	$(JUPYTEXT) --to ipynb --output - $< \
	    | $(JQ) '.metadata |= $(NOTEBOOK_METADATA)' \
	    | $(JQ) 'del(.cells[].metadata.lines_to_next_cell)' \
	    > $@
