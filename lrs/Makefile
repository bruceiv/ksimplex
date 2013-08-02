CC = gcc
CPPFLAGS = -DTIMES -DGMP -DLRS_QUIET
#CFLAGS = -O0 -ggdb
CFLAGS = -02
CXXFLAGS = $(CFLAGS) -Wall -Wno-unused
LDFLAGS = -lgmpxx -lgmp

# object files to include in this library
OBJS = lrslib.o lrsgmp.o matrix.o lrs.o

# rules for constructions of objects from sources
.c.o:  
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< $(LDFLAGS)
.cpp.o:  
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< $(LDFLAGS)

.PHONY:  clean

liblrs.a: $(OBJS)
	$(AR) rcs liblrs.a $(OBJS)

clean:  
	-rm $(OBJS) liblrs.a
