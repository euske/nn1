# Makefile

RM=rm -f
CC=cc -O -Wall -Werror

LIBS=-lm

DATADIR=./data
MNIST_FILES= \
	$(DATADIR)/train-images-idx3-ubyte \
	$(DATADIR)/train-labels-idx1-ubyte \
	$(DATADIR)/t10k-images-idx3-ubyte \
	$(DATADIR)/t10k-labels-idx1-ubyte

all: test_bnn

clean:
	-$(RM) ./bnn ./mnist *.o

test_bnn: ./bnn
	./bnn

test_mnist: ./mnist
	./mnist $(MNIST_FILES)

./bnn: bnn.c
	$(CC) -o $@ $^ $(LIBS)

./mnist: mnist.c cnn.c
	$(CC) -o $@ $^ $(LIBS)

mnist.c: cnn.h
cnn.c: cnn.h
