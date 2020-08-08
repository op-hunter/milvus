#!/bin/bash

#declare -a prgms
prgms=(aa "bb" cc dd)
lenp=${#prgms[@]}

for(( i=0; i<$lenp; i++));
do
    echo "prgms[$i] = " ${prgms[$i]}
done
