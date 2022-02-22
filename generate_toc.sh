#!/bin/bash
# generate the table of contents for deep atlas tutorial notebook, and print to stdout
# need jupyter notebook and the python package markdown-toc

set -e

if [ $# -lt 1 ]; then 
  echo "need to specify notebook file to create TOC for"
  exit
fi

jupyter nbconvert --to markdown --output temp.md $1
markdown-toc -h 6 -t github --no-write temp.md
rm temp.md
