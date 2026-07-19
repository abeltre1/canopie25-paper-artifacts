set terminal postscript eps enhanced color 20
set output 'llama4-scout-w4a16.eps'

#set title 'RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16'
set xlabel 'Maximum Request Concurrency'
set ylabel 'Output Token Throughput (tokens/s)'

set pointsize 1.25
set logscale x 2
#set logscale y 10
set xrange [.9:1050]
set yrange [10:]
#set key top left
#set key bottom right
set key at screen 0.68,0.93

datafile = 'results.dat'

plot \
    datafile using 1:2 title "ClusterA HPC, Run 1 (clustera44)"      with linespoints lw 3 lc rgb 'red', \
    datafile using 1:3 title "ClusterA HPC, Run 2 (clustera01)"      with linespoints lw 3 lc rgb 'orange', \
    datafile using 1:4 title "ClusterA HPC, Run 3 (clustera06)"      with linespoints lw 3 lc rgb 'green', \
    datafile using 1:5 title "ClusterA HPC, Run 4 (clustera29)"      with linespoints lw 3 lc rgb 'blue', \
    datafile using 1:6 title "ClusterA HPC, Run 5 (clustera17)"      with linespoints lw 3 lc rgb 'purple', \
    datafile using 1:7 title "OCPCluster K8s, Run 1 (ocpcluster05)" with linespoints lw 3 lc rgb 'gray', \
    datafile using 1:8 title "OCPCluster K8s, Run 2 (ocpcluster05)" with linespoints lw 3 lc rgb 'black'

set output
