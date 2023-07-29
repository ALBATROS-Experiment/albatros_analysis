#!/opt/homebrew/bin/bash

# [Steve] This is what I use to build the docs. 
# Place this file in the parent to the project root and run it from there with bash. 

# uncomment for dev, builds and serves the docs
pdoc --docformat="numpy" --logo="https://upload.wikimedia.org/wikipedia/commons/7/73/Short_tailed_Albatross1.jpg" --math --mermaid albatros_analysis;

# uncomment for prod, builds docs and output to html
#pdoc --docformat="numpy" --logo="https://upload.wikimedia.org/wikipedia/commons/7/73/Short_tailed_Albatross1.jpg" --math --mermaid albatros_analysis -o albatros_analysis/docs;
