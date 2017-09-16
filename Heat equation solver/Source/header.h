#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <mkl.h>

#define L    1.0
#define T    1.0

#define Nx   50
#define Nt   50

#define h     ((L)/(Nx))
#define tau   ((T)/(Nt))

#define eps 0.00005
#define CP 0.0001

#define PI 	3.1415926535897932384626433832795

#define OMP

#define NOISE 0
//#define ANALYTIC

#define PROBLEM 1

// 0  - only two boundary conditions if x=0
// 1  - problem 0 + two conditions for null flux through t=0 and t=1

#define OMP_THREADS 4

//#define SECOND

double fi_func(double x);
double g0_func(double t);
double gL_func(double t);
double k_func(double x);
double kd_func(double x);

// Kernel

void sweep(int n, double *a, double *b, double *c, double *f, double *x_swp);
void cont_prob_by_inv_sch_no_bound(double *u, int ldu, double *g0, double *a, double *b, double *c, double *u_invc);
void cont_prob_by_inv_sch_yes_bound(double *u, int ldu, double *g0, double *a, double *b, double *c, double *u_invc);
void analit_cont_prob_vs_inv_sch_yes_bound(double *f, double *g0, double *a, double *b, double *c, double *u_invcp);
void cont_prob_by_grad_meth(double *u_dir, int ldu_dir, double *u_ex, int ldu_ex, double *f_data, double *a, double *b, double *c, double *fi_grad, double *gl_grad);
void direct_problem(double *u_dir, int ldu, double *fi, double *gl, double *a, double *b, double *c, int exact);
void adjoint_problem(double *u_adj, int ldu_adj, double *u_dir, int ldu_dir, double *f_data, double *a, double *b, double *c);
void gradient(double *psi, int ldu, double *grad1, double *grad2);

// Support

void fill_continuation1_matrix(double *A, int ldA, double *a, double *b, double *c);
void fill_continuation2_matrix(double *A, int ldA, double *a, double *b, double *c);
void fill_continuation1_vectorF(double *F1, double *u0, double *g0, double *f0);
void fill_continuation2_vectorF(double *F1, double *u0, double *g0, double *f0, double *gL, double *fL);
void validation_CP(int m, int n, double *A, int lda, double *u, double *F, int problem, double *f_diff);
void print_mat(double *A3diag, double *F, int m, int n, int lda);
void matrix_to_file(int m, int n, double *u_adj, int ldu, int iter);
void output_SVD(int n, double *u, int ldu, double *sing, int cross);
void output_DP(int n, double *u, int ldu);
void mem_free(int n, double **arrays);
double* diff_vec(double* u1, double* u2, int size, int problem);
double diff_mat(int m, int n, double* u1, double* u2, int ldu, double* res, int problem);
double rand1(void);

