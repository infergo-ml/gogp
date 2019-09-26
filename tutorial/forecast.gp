data = "forecast.dat"

if (ARG1 ne "")
	data = ARG1

set ylabel "y"
set xlabel "x"
plot data using 1:3:($3 + 3*$4) with filledcurves \
		  lc "#eeeeee" title "3*sd", \
	 "" using 1:3:($3 - 3*$4)  with filledcurves \
	      lc "#eeeeee" title "", \
	 "" using 1:3:($3 + 2*$4) with filledcurves \
		  lc "#dddddd" title "2*sd", \
	 "" using 1:3:($3 - 2*$4)  with filledcurves \
	      lc "#dddddd" title "", \
	 "" using 1:3:($3 + $4) with filledcurves \
		  lc "#cccccc" title "sd", \
	 "" using 1:3:($3 - $4)  with filledcurves \
	      lc "#cccccc" title "", \
	 "" using 1:3 with lines lw 2 lc black title "predicted", \
	 "" using 1:2 with points ls 7 lc black title "observed" 
