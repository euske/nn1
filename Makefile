# Makefile

RM=rm -f
CC=cc -O -Wall -Werror
CURL=curl
GZIP=gzip

LIBS=-lm

DATADIR=./data
MNIST_FILES= \
	$(DATADIR)/train-images-idx3-ubyte \
	$(DATADIR)/train-labels-idx1-ubyte \
	$(DATADIR)/t10k-images-idx3-ubyte \
	$(DATADIR)/t10k-labels-idx1-ubyte

all: test_rnn

clean:
	-$(RM) ./bnn ./mnist ./rnn *.o

get_mnist:
	-mkdir ./data
	-$(CURL) http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz | \
		$(GZIP) -dc > ./data/train-images-idx3-ubyte
	-$(CURL) http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz | \
		$(GZIP) -dc > ./data/train-labels-idx1-ubyte
	-$(CURL) http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz | \
		$(GZIP) -dc > ./data/t10k-images-idx3-ubyte
	-$(CURL) http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz | \
		$(GZIP) -dc > ./data/t10k-labels-idx1-ubyte

test_bnn: ./bnn
	./bnn

test_mnist: ./mnist $(MNIST_FILES)
	./mnist $(MNIST_FILES)

test_rnn: ./rnn
	./rnn

./bnn: bnn.c
	$(CC) -o $@ $^ $(LIBS)

./mnist: mnist.c cnn.c
	$(CC) -o $@ $^ $(LIBS)

./rnn: rnn.c
	$(CC) -o $@ $^ $(LIBS)

mnist.c: cnn.h
cnn.c: cnn.h
