
API=opencl
architecture=amd
precision=single

filename=inputsize_${API}_${architecture}_${precision}.csv

rm $filename

make clean
make 


for size in 4608 18432 73728 294912 1179648
do
  ln -fs "./naca/new_grid_$size.dat" ./new_grid.dat
  for iter in `seq 3`
  do
    echo -ne "$size, $iter, $API, $architecture, $precision, " | tee -a  $filename
    /usr/bin/time -f "elapsed: %e" ./airfoil_${API} 2>&1 | awk '$1=="elapsed:"{printf("%.4f\n",$2);}' | tee -a $filename
  done
done
