#!/bin/bash 
set -x
FILEDIR=$(dirname "$0")
echo "Download Amazon data to $FILEDIR"

wget -P $FILEDIR http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/All_Amazon_Meta.json.gz
wget -P $FILEDIR http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/All_Amazon_Review_5.json.gz
wget -P $FILEDIR http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/duplicates.txt 