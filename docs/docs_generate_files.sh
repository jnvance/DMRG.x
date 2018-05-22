#!/bin/bash
# Note: To be called from root directory
sed $'/[@][@][@]/{r README.md\nd}' docs/doc_00_overview.dox.in > docs/doc_00_overview.dox
