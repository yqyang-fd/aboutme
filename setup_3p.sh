#!bin/bash

# remove the old files in the directory `third_party` if it exists
if [ -d third_party ]; then
    rm -rf third_party
fi

# create the directory `third_party`
mkdir third_party
cd third_party

# clone the repository `mkdocs-gitbook-theme`
# and then copy the patched files to the directory
# manually install the theme
git clone https://gitlab.com/lramage/mkdocs-gitbook-theme.git
cp ../patches/gitbook-theme/*.html mkdocs-gitbook-theme/mkdocs_gitbook/
pip install -U mkdocs-gitbook-theme/

