#!/bin/bash
# Note: To be called from root directory

# Place contents of readme into overview page
sed $'/[@][@][@]/{r README.md\nd}' docs/doc_00_overview.dox.in > docs/doc_00_overview.dox.tmp.01

# Set all h2 to h1
sed 's/##/#/g' docs/doc_00_overview.dox.tmp.01 > docs/doc_00_overview.dox.tmp.02

# Remove the header from the readme file
sed '/<!--start01-->/,/<!--end01-->/d' docs/doc_00_overview.dox.tmp.02 > docs/doc_00_overview.dox

# Delete temporary file
rm docs/doc_00_overview.dox.tmp.*
