API=opencl
architecture=avx
precision=single

filename=blocksize_${API}_${architecture}_${precision}.csv

rm $filename

for block in 32 64 128 256
do
  for part in 32 64 128 256
  do
    make clean
    make BLOCK_SIZE=$block PART_SIZE=$part
    for iter in `seq 3`
    do
      echo -ne "$block, $part, $iter, $API, $architecture, $precision, " | tee -a  $filename
      /usr/bin/time -f "elapsed: %e" ./airfoil_${API} 2>&1 | awk '$1=="elapsed:"{printf("%.4f\n",$2);}' | tee -a $filename
    done
  done
done

