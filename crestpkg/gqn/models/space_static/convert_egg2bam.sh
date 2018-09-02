#!/bin/bash

SOURCE="egg"
TARGET="bam/essex"


cd egg
for fullfile in `ls *.egg`
do
    filewithext=$(basename -- "$fullfile")
    filename="${filewithext%.*}"
    echo "Processing $filewithext --> $filename.bam"
    egg2bam -ps rel -o "../$TARGET/$filename.bam" $filewithext
done
cd ..
