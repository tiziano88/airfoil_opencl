#/bin/bash

trap "kill 0" SIGINT SIGTERM EXIT

filename=res.txt

rm $filename

for block in 32 64 128 256
do
  for part in 32 64 128 256
  do
    for iter in `seq 2`
    do
      make clean
      make BLOCK_SIZE=$block PART_SIZE=$part
      echo -ne "$block, $part, $iter, " | tee -a  $filename
      /usr/bin/time -f "elapsed: %e" ./airfoil_opencl 2>&1 | awk -f extract_results | tee -a $filename
    done
  done
done

