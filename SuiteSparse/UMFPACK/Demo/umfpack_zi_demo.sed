/::/d
1,$s/_xx_/_zi_/g
1,$s/Int/int/g
1,$s/WSIZE/10/
1,$s/%ld/%d/g
/define ABS/ {
	s/ABS/ABS(x,z) ((x) >= 0 ? (x) : -(x)) + ((z) >= 0 ? (z) : -(z))/
	}
