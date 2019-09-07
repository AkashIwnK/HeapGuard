# Rules to compile the HeapGuard heap allocator

CC = clang

OPTIMIZATION = -O2
HEAP_FLAGS = `llvm-config --cflags --ldflags --libs --system-libs` -w $(OPTIMIZATION) -g

all:
	$(CC) -o Stringlib.o -c -fPIC Stringlib.c $(HEAP_FLAGS)
	$(CC) -o HeapGuard.o -c -fPIC HeapGuard.c $(HEAP_FLAGS)
	$(CC) -shared -o libHeapGuard.so HeapGuard.o Stringlib.o -O2 -g

clean: 
	rm *.o *.so
	
	
