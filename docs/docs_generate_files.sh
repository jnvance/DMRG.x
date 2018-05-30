#!/bin/bash
# Note: To be called from root directory

# Place contents of readme into overview page
sed $'/[@][@][@]/{r README.md\nd}' docs/doc_00_overview.dox.in > docs/doc_00_overview.dox.tmp
# Remove the header from the readme file
sed '/<!--start01-->/,/<!--end01-->/d' docs/doc_00_overview.dox.tmp > docs/doc_00_overview.dox
# Delete temporary file
rm docs/doc_00_overview.dox.tmp
