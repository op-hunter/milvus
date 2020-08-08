#!/bin/bash

ms=(16 32)
lenms=${#ms[@]}

efC=(16 32 64 100 200)
lenefc=${#efC[@]}

efS=(16 32 64 100 200)
lenefs=${#efS[@]}

topk=(1 10 100)
lentopk=${#topk[@]}

prgms=(hnsw-milvus hnsw-flat hnsw-pq hnsw-sq)
lenp=${#prgms[@]}

for prgm in ${prgms[@]};
do
    echo "test " $prgm "ing"
    echo "-----------------------------------------------------------------------"
    for m in ${ms[@]};
    do
        echo "when m = " $m
        for efc in ${efC[@]};
        do
            echo "when efConstruction = " $efc
            for efs in ${efS[@]};
            do
                echo "when efSearch = " $efs
                for k in ${topk[@]};
                do
                    echo "when topk = " $k
                    mkdir -p test-results/$prgm
                    ./$prgm $m $efc $efs $k > test-results/$prgm/$prgm-$m-$efc-$efs-$k.txt
                done
            done
        done
    done
    echo "-----------------------------------------------------------------------"
    echo ""
done



echo "test hnsw-milvus:"
for(( i=0; i<$len; i++ ));
do
    echo ${ids[$i]}
    ./ont_precision 128 1000000 100 ${ids[$i]} > ont${ids[$i]}.out
    ls -l precision.tree >> ont${ids[$i]}.out
    ls -lh precision.tree >> ont${ids[$i]}.out
    echo "------------------------------------------------------------------------------------"
done

echo "test snt_precision:"
for(( i=0; i<$len; i++ ));
do
    echo ${ids[$i]}
    ./snt_precision 128 1000000 100 ${ids[$i]} > snt${ids[$i]}.out
    ls -l precision.tree >> snt${ids[$i]}.out
    ls -lh precision.tree >> snt${ids[$i]}.out
    echo "------------------------------------------------------------------------------------"
done
#./nt_precision 128 1000000 10 1 > nt1.out
#ls -l precision.tree >> nt1.out
#ls -lh precision.tree >> nt1.out
#echo "---------------------------------------1---------------------------------------------"

#./nt_precision 128 1000000 2 10 1 1 > nt2.out
#ls -l precision.tree >> nt2.out
#ls -lh precision.tree >> nt2.out
#echo "---------------------------------------2---------------------------------------------"
#
#./nt_precision 128 1000000 4 10 1 1 > nt4.out
#ls -l precision.tree >> nt4.out
#ls -lh precision.tree >> nt4.out
#echo "---------------------------------------4---------------------------------------------"
#
#./nt_precision 128 1000000 8 10 1 1 > nt8.out
#ls -l precision.tree >> nt8.out
#ls -lh precision.tree >> nt8.out
#echo "---------------------------------------8---------------------------------------------"
#
#./nt_precision 128 1000000 16 10 1 1 > nt16.out
#ls -l precision.tree >> nt16.out
#ls -lh precision.tree >> nt16.out
#echo "---------------------------------------16----------------------------------------------"
#
#./nt_precision 128 1000000 32 10 1 1 > nt32.out
#ls -l precision.tree >> nt32.out
#ls -lh precision.tree >> nt32.out
#echo "----------------------------------------32---------------------------------------------"
#
#./nt_precision 128 1000000 64 10 1 1 > nt64.out
#ls -l precision.tree >> nt64.out
#ls -lh precision.tree >> nt64.out
#echo "-----------------------------------------64--------------------------------------------"
#
#./nt_precision 128 1000000 128 10 1 1 > nt128.out
#ls -l precision.tree >> nt128.out
#ls -lh precision.tree >> nt128.out
#echo "------------------------------------------128-------------------------------------------"
#
#./nt_precision 128 1000000 256 10 1 1 > nt256.out
#ls -l precision.tree >> nt256.out
#ls -lh precision.tree >> nt256.out
#echo "------------------------------------------256-----------------------------------------"
#
