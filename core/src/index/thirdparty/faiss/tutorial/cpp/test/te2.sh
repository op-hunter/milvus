#!/bin/bash

#declare -a prgms
prgms=(aa bb "cc" dd)
lenp=${#prgms[@]}

ids=(1 2 3 4)
leni=${#ids[@]}
#for(( i=0; i<$lenp; i++));
#do
#    echo "prgms[$i] = " ${prgms[$i]}
#done
for p in ${prgms[@]};
do 
    echo "echo" $p "ing" 
done

for id in ${ids[@]};
do 
    echo $id 
done

for i in 1 3 4 6;
do 
    echo $i 
done

