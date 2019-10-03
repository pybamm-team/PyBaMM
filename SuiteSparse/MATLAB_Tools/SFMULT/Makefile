all: sfmult.mexglx

diffs: kernel_diffs kernel_usage_diffs structure_diffs

kernel_diffs:
	- diff sfmult_anxnyt_k.c sfmult_anxtyt_k.c
	- diff sfmult_atxtyn_k.c sfmult_atxtyt_k.c
	- diff sfmult_vector_1.c sfmult_vector_k.c

# each of these pairs use the same sfmult_A*_X*_Y*_[234] kernels
kernel_usage_diffs:
	- diff sfmult_anxtyn.c sfmult_anxtyt.c
	- diff sfmult_anxnyn.c sfmult_anxnyt.c
	- diff sfmult_atxnyn.c sfmult_atxtyn.c
	- diff sfmult_atxnyt.c sfmult_atxtyt.c

# each of these pairs use the same outer structure but different kernels
structure_diffs:
	- diff sfmult_anxnyn.c sfmult_anxtyn.c
	- diff sfmult_anxnyt.c sfmult_anxtyt.c
	- diff sfmult_atxnyn.c sfmult_atxnyt.c
	- diff sfmult_atxtyn.c sfmult_atxtyt.c

OBJ = \
    sfmult_anxnyn.o \
    sfmult_anxnyt.o \
    sfmult_anxnyt_k.o \
    sfmult_anxtyn.o \
    sfmult_anxtyt.o \
    sfmult_anxtyt_k.o \
    sfmult_atxnyn.o \
    sfmult_atxnyt.o \
    sfmult_atxtyn.o \
    sfmult_atxtyn_k.o \
    sfmult_atxtyt.o \
    sfmult_atxtyt_k.o \
    sfmult.o \
    sfmult_vector_1.o \
    sfmult_vector_k.o \
    sfmult_xA.o \
    ssmult_transpose.o

$(OBJ): sfmult.h

.c.o:
	mex -O -c $<

sfmult.mexglx: sfmult_mex.c $(OBJ) sfmult.h
	mex -O -output sfmult sfmult_mex.c $(OBJ)

clean:
	rm -f *.o

purge: clean
	rm sfmult.mexglx
