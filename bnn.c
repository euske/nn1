/*
  bnn.c
  Basic Neural Network in C.

  $ cc -o bnn bnn.c -lm
*/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define DEBUG_LAYER 0

/* f: function to learn */
static double f(double a, double b)
{
    /* return a*b; */
    return fabs(a-b);
}


/*  Misc. functions
 */

/* rnd(): uniform random [0.0, 1.0] */
static inline double rnd()
{
    return ((double)rand() / RAND_MAX);
}

/* nrnd(): normal random (std=1.0) */
static inline double nrnd()
{
    return (rnd()+rnd()+rnd()+rnd()-2.0) * 1.724; /* std=1.0 */
}

/* sigmoid(x): sigmoid function */
static inline double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}
/* sigmoid_d(y): sigmoid gradient */
static inline double sigmoid_g(double y)
{
    return y * (1.0 - y);
}


/*  Layer
 */

typedef struct _Layer {

    int lid;                    /* Layer ID */
    struct _Layer* lprev;       /* Previous Layer */
    struct _Layer* lnext;       /* Next Layer */

    int nnodes;                 /* Num. of Nodes */
    double* outputs;            /* Node Outputs */
    double* gradients;          /* Node Gradients */
    double* errors;             /* Node Errors */

    int nbiases;                /* Num. of Biases */
    double* biases;             /* Biases (trained) */
    double* u_biases;           /* Bias Updates */

    int nweights;               /* Num. of Weights */
    double* weights;            /* Weights (trained) */
    double* u_weights;          /* Weight Updates */

} Layer;

/* Layer_create(lprev, nnodes)
   Creates a Layer object.
*/
Layer* Layer_create(Layer* lprev, int nnodes)
{
    Layer* self = (Layer*)calloc(1, sizeof(Layer));
    if (self == NULL) return NULL;

    self->lprev = lprev;
    self->lnext = NULL;
    self->lid = 0;
    if (lprev != NULL) {
        assert (lprev->lnext == NULL);
        lprev->lnext = self;
        self->lid = lprev->lid+1;
    }

    self->nnodes = nnodes;
    self->outputs = (double*)calloc(self->nnodes, sizeof(double));
    self->gradients = (double*)calloc(self->nnodes, sizeof(double));
    self->errors = (double*)calloc(self->nnodes, sizeof(double));

    if (lprev != NULL) {
        /* Fully connected */
        self->nbiases = self->nnodes;
        self->biases = (double*)calloc(self->nbiases, sizeof(double));
        self->u_biases = (double*)calloc(self->nbiases, sizeof(double));
        for (int i = 0; i < self->nbiases; i++) {
            self->biases[i] = 0;
        }

        self->nweights = lprev->nnodes * self->nnodes;
        self->weights = (double*)calloc(self->nweights, sizeof(double));
        self->u_weights = (double*)calloc(self->nweights, sizeof(double));
        for (int i = 0; i < self->nweights; i++) {
            self->weights[i] = 0.1 * nrnd();
        }
    }

    return self;
}

/* Layer_destroy(self)
   Releases the memory.
*/
void Layer_destroy(Layer* self)
{
    assert (self != NULL);

    free(self->outputs);
    free(self->gradients);
    free(self->errors);

    if (self->biases != NULL) {
        free(self->biases);
    }
    if (self->u_biases != NULL) {
        free(self->u_biases);
    }
    if (self->weights != NULL) {
        free(self->weights);
    }
    if (self->u_weights != NULL) {
        free(self->u_weights);
    }

    free(self);
}

/* Layer_dump(self, fp)
   Shows the debug output.
*/
void Layer_dump(const Layer* self, FILE* fp)
{
    assert (self != NULL);
    Layer* lprev = self->lprev;
    fprintf(fp, "Layer%d", self->lid);
    if (lprev != NULL) {
        fprintf(fp, " (<- Layer%d)", lprev->lid);
    }
    fprintf(fp, ": nodes=%d\n", self->nnodes);
    fprintf(fp, "  outputs = [");
    for (int i = 0; i < self->nnodes; i++) {
        fprintf(fp, " %.4f", self->outputs[i]);
    }
    fprintf(fp, "]\n");

    if (self->biases != NULL) {
        fprintf(fp, "  biases = [");
        for (int i = 0; i < self->nbiases; i++) {
            fprintf(fp, " %.4f", self->biases[i]);
        }
        fprintf(fp, "]\n");
    }
    if (self->weights != NULL) {
        fprintf(fp, "  weights = [");
        for (int i = 0; i < self->nweights; i++) {
            fprintf(fp, " %.4f", self->weights[i]);
        }
        fprintf(fp, "]\n");
    }
}

/* Layer_feedForw(self)
   Performs feed forward updates.
*/
static void Layer_feedForw(Layer* self)
{
    assert (self->lprev != NULL);
    Layer* lprev = self->lprev;

    int k = 0;
    for (int i = 0; i < self->nnodes; i++) {
        /* Y = f(W * X + B) */
        double x = self->biases[i];
        for (int j = 0; j < lprev->nnodes; j++) {
            x += (lprev->outputs[j] * self->weights[k++]);
        }
        double y = sigmoid(x);
        self->outputs[i] = y;
        /* Store the gradient at this point. */
        self->gradients[i] = sigmoid_g(y);
    }

#if DEBUG_LAYER
    fprintf(stderr, "Layer_feedForw(Layer%d):\n", self->lid);
    fprintf(stderr, "  outputs = [");
    for (int i = 0; i < self->nnodes; i++) {
        fprintf(stderr, " %.4f", self->outputs[i]);
    }
    fprintf(stderr, "]\n  gradients = [");
    for (int i = 0; i < self->nnodes; i++) {
        fprintf(stderr, " %.4f", self->gradients[i]);
    }
    fprintf(stderr, "]\n");
#endif
}

/* Layer_feedBack(self)
   Performs backpropagation.
*/
static void Layer_feedBack(Layer* self)
{
    if (self->lprev == NULL) return;

    assert (self->lprev != NULL);
    Layer* lprev = self->lprev;

    /* Clear errors. */
    for (int j = 0; j < lprev->nnodes; j++) {
        lprev->errors[j] = 0;
    }

    int k = 0;
    for (int i = 0; i < self->nnodes; i++) {
        /* Computer the weight/bias updates. */
        double dnet = self->errors[i] * self->gradients[i];
        for (int j = 0; j < lprev->nnodes; j++) {
            /* Propagate the errors to the previous layer. */
            lprev->errors[j] += self->weights[k] * dnet;
            self->u_weights[k] += dnet * lprev->outputs[j];
            k++;
        }
        self->u_biases[i] += dnet;
    }

#if DEBUG_LAYER
    fprintf(stderr, "Layer_feedBack(Layer%d):\n", self->lid);
    for (int i = 0; i < self->nnodes; i++) {
        double dnet = self->errors[i] * self->gradients[i];
        fprintf(stderr, "  dnet = %.4f, dw = [", dnet);
        for (int j = 0; j < lprev->nnodes; j++) {
            double dw = dnet * lprev->outputs[j];
            fprintf(stderr, " %.4f", dw);
        }
        fprintf(stderr, "]\n");
    }
#endif
}

/* Layer_setInputs(self, values)
   Sets the input values.
*/
void Layer_setInputs(Layer* self, const double* values)
{
    assert (self != NULL);
    assert (self->lprev == NULL);

#if DEBUG_LAYER
    fprintf(stderr, "Layer_setInputs(Layer%d): values = [", self->lid);
    for (int i = 0; i < self->nnodes; i++) {
        fprintf(stderr, " %.4f", values[i]);
    }
    fprintf(stderr, "]\n");
#endif

    /* Set the values as the outputs. */
    for (int i = 0; i < self->nnodes; i++) {
        self->outputs[i] = values[i];
    }

    /* Start feed forwarding. */
    Layer* layer = self->lnext;
    while (layer != NULL) {
        Layer_feedForw(layer);
        layer = layer->lnext;
    }
}

/* Layer_getOutputs(self, outputs)
   Gets the output values.
*/
void Layer_getOutputs(const Layer* self, double* outputs)
{
    assert (self != NULL);
    for (int i = 0; i < self->nnodes; i++) {
        outputs[i] = self->outputs[i];
    }
}

/* Layer_getErrorTotal(self)
   Gets the error total.
*/
double Layer_getErrorTotal(const Layer* self)
{
    assert (self != NULL);
    double total = 0;
    for (int i = 0; i < self->nnodes; i++) {
        double e = self->errors[i];
        total += e*e;
    }
    return (total / self->nnodes);
}

/* Layer_learnOutputs(self, values)
   Learns the output values.
*/
void Layer_learnOutputs(Layer* self, const double* values)
{
    assert (self != NULL);
    assert (self->lprev != NULL);
    for (int i = 0; i < self->nnodes; i++) {
        self->errors[i] = (self->outputs[i] - values[i]);
    }

#if DEBUG_LAYER
    fprintf(stderr, "Layer_learnOutputs(Layer%d): errors = [", self->lid);
    for (int i = 0; i < self->nnodes; i++) {
        fprintf(stderr, " %.4f", self->errors[i]);
    }
    fprintf(stderr, "]\n");
#endif

    /* Start backpropagation. */
    Layer* layer = self->lprev;
    while (layer != NULL) {
        Layer_feedBack(layer);
        layer = layer->lprev;
    }
}

/* Layer_update(self, rate)
   Updates the weights.
*/
void Layer_update(Layer* self, double rate)
{
#if DEBUG_LAYER
    fprintf(stderr, "Layer_update(Layer%d): rate = %.4f\n", self->lid, rate);
#endif

    /* Update the bias and weights. */
    if (self->biases != NULL) {
        for (int i = 0; i < self->nbiases; i++) {
            self->biases[i] -= rate * self->u_biases[i];
            self->u_biases[i] = 0;
        }
    }
    if (self->weights != NULL) {
        for (int i = 0; i < self->nweights; i++) {
            self->weights[i] -= rate * self->u_weights[i];
            self->u_weights[i] = 0;
        }
    }

    /* Update the previous layer. */
    if (self->lprev != NULL) {
        Layer_update(self->lprev, rate);
    }
}


/* main */
int main(int argc, char* argv[])
{
    /* Use a fixed random seed for debugging. */
    srand(0);
    /* Initialize layers. */
    Layer* linput = Layer_create(NULL, 2);
    Layer* lhidden = Layer_create(linput, 3);
    Layer* loutput = Layer_create(lhidden, 1);
    Layer_dump(linput, stderr);
    Layer_dump(lhidden, stderr);
    Layer_dump(loutput, stderr);

    /* Run the network. */
    double rate = 1.0;
    int nepochs = 10000;
    for (int i = 0; i < nepochs; i++) {
        double x[2];
        double y[1];
        double t[1];
        x[0] = rnd();
        x[1] = rnd();
        t[0] = f(x[0], x[1]);
        Layer_setInputs(linput, x);
        Layer_getOutputs(loutput, y);
        Layer_learnOutputs(loutput, t);
        double etotal = Layer_getErrorTotal(loutput);
        fprintf(stderr, "i=%d, x=[%.4f, %.4f], y=[%.4f], t=[%.4f], etotal=%.4f\n",
                i, x[0], x[1], y[0], t[0], etotal);
        Layer_update(loutput, rate);
    }

    /* Dump the finished network. */
    Layer_dump(linput, stdout);
    Layer_dump(lhidden, stdout);
    Layer_dump(loutput, stdout);

    Layer_destroy(linput);
    Layer_destroy(lhidden);
    Layer_destroy(loutput);
    return 0;
}
