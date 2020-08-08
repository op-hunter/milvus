#!/bin/bash

#declare -a prgms
prgms=( aa bb cc dd )
lenp=${#prgms[@]}

for(( i=0; i<$lenp; i++));
do
    mkdir -p a/b/c
    ./${prgms[$i]} $i > a/b/c/${prgms[$i]}-$i.out
    echo "-----------------------------"
done
