/::/d
1,$s/_xx_/_di_/g
1,$s/Int/int/g
1,$s/WSIZE/5/
1,$s/%ld/%d/g
/define ABS/ {
	s/ABS/ABS(x) ((x) >= 0 ? (x) : -(x))/
	}
/, rz \[i\]/ {
	s/, rz \[i\]//
	}
/, Avalz/ {
	s/, Avalz//
	}
/, rz/ {
	s/, rz//
	}
/, bz/ {
	s/, bz//
	}
/, xz/ {
	s/, xz//
	}

/, Lz/ {
	s/, Lz//
	}
/, Uz/ {
	s/, Uz//
	}
/, Dz/ {
	s/, Dz//
	}
/, Az/ {
	s/, Az//
	}
/, Cz, TRUE/ {
	s/, Cz, TRUE//
	}
/, Cz/ {
	s/, Cz//
	}
/, Rbz/ {
	s/, Rbz//
	}
/, yz/ {
	s/, yz//
	}

/ || !Lz/ {
	s/ || !Lz//
	}
/ || !Uz/ {
	s/ || !Uz//
	}
/ || !Dz/ {
	s/ || !Dz//
	}
/ || !Az/ {
	s/ || !Az//
	}
/ || !Cz/ {
	s/ || !Cz//
	}

/rz/d
/Rbz/d
/yz/d
/Avalz/d
/Az/d
/Cz/d
/bz/d
/xz/d
/Lz/d
/Uz/d
/Dz/d
/complex/d

