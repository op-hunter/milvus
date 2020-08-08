#!/bin/bash


DIR=`pwd`"/test-results/"
output="ana_res.txt"

prgms=(hnsw-milvus hnsw-flat hnsw-pq hnsw-sq)

for prgm in ${prgms[@]};
do
    echo "analysis: " $prgm >> $output
    fpath=${DIR}${prgm}
    echo "file path: " ${fpath} >> $output
    for file in `ls ${fpath}`;
    do
        split=(${file//-/ })
        spl2=(${split[5]//./ })
        echo ${split[2]}  ${split[3]} ${split[4]} ${spl2[0]} >> $output
        cat ${fpath}"/"${file} | awk '{print $0}' | grep "build index costs" >> $output
        cat ${fpath}"/"${file} | awk '{print $0}' | grep "search 10000 times costs" >> $output
        cat ${fpath}"/"${file} | awk '{print $0}' | grep "correct query of" >> $output
        echo "--------------------------------------------" >> $output
    done
done


