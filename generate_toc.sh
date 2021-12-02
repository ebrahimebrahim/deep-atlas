#!/bin/bash
# generate the table of contents for deep atlas tutorial notebook, and print to stdout
# need jupyter notebook and the python package markdown-toc


set -e

jupyter nbconvert --to markdown --output temp.md deep_atlas_tutorial.ipynb
markdown-toc -h 6 -t github --no-write temp.md
rm temp.md
