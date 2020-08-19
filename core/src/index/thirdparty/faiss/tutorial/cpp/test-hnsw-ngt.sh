#!/bin/bash

ms=(16 24 32)
ef=(50 75 100)

for i in {0..2}
do
    echo $i, ${ms[$i]}, ${ef[$i]}
    ./hnsw-ngt-test ${ms[$i]} ${ef[$i]} ${ef[$i]} 1 > test/${ms[$i]}-${ef[$i]}-${ef[$i]}-1.txt
done

