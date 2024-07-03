#!/usr/bin/env bash

cd $(dirname ${BASH_SOURCE[0]})/..

fnames=""
for fname in $@
do
    if [[ "$fname" == *".py" ]] || [ -d $fname ] && [[ "$fname" == "fms_dgt"* ]]
    then
        fnames="$fnames $fname"
    else
        echo "Ignoring non-library file: $fname"
    fi
done
if [ "$fnames" == "" ]
then
    fnames="fms_dgt"
fi

ruff check $arg $fnames