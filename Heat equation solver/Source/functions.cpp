#include "header.h"


/*The declare_simd construct enables the creation of SIMD versions of a specific function.
These versions can be used to process multiple arguments from a single invocation from a SIMD loop concurrently.
When the function is called, the function must not have any side-effects that would change its execution for concurrent iterations of a SIMD chunk.
When the function is called from a SIMD loop, the function cannot cause the execution of any OpenMP* construct.*/

#ifdef OMP
#pragma omp declare simd
#endif
double fi_func(double x)
{
	//return x + 1;
	//return cos(PI*x);
	return sin(PI*x)+1;
}

#ifdef OMP
#pragma omp declare simd
#endif
double g0_func(double t)
{
	//return t + 0.5;
	//return t + 1;
	//return t + 1;
	return cos(PI*t);
	//sin(PI*t) + 1;
}

#ifdef OMP
#pragma omp declare simd
#endif
double gL_func(double t)
{
	//return t + 1.5;
	//return t - 1;
	//return 2 * t  + 1;
	return cos(PI*t);
}


#ifdef OMP
#pragma omp declare simd
#endif
double k_func(double x)
{
#ifndef ANALYTIC
	return x*x+1;
	//return sin(x+2);
#else
	return 1.0;
#endif
}

#ifdef OMP
#pragma omp declare simd
#endif
double kd_func(double x)
{
#ifndef ANALYTIC
	return 2*x;
#else
	return 0.0;
#endif
}

int inarray(int numb)
{
	int n = Nx + 1;
	int mass[Nx + 1];
	for (int i = 0; i < n + 1; i ++)
		mass[i] = i*n - 1;

	for (int i = 0; i < n + 1;i++)
	if (mass[i] == numb) return 0;

	return 1;
}

void print_mat(double *A3diag, double* F, int m, int n, int lda)
{
	printf("\n");
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
			if(inarray(j)==1) printf("%5.1lf ", A3diag[i + lda*j]);
			else printf("%5.1lf |", A3diag[i + lda*j]);
		
		printf("      %6.3lf\n", F[i]);
		if (!(inarray(i) == 0 && i != 0)) printf("\n");
		else printf("----------------------------------------------------------------------------------------------------------------------------\
-------------------------------------------------------------------------------------------------------------\n");
	}
	printf("\n");
}

#ifdef OMP
#pragma omp declare simd
#endif
void sweep(int n, double* a, double* b, double* c, double* f, double* x_swp)
{
	// Beta and alpha for sweep. Number of elements - Nx: from 1 to Nx;
	double *alpha = (double*)malloc(n * sizeof(double));  // alpha[0]=0 is not used
	double *beta = (double*)malloc(n * sizeof(double)); // beta[0]=0 is not used

	// 1. Finding coefficients betta and alpha
	alpha[0] = 0;
	beta[0] = 0;

	for (int i = 0; i < n - 1; i++)
	{
		alpha[i + 1] = -b[i] / (a[i] * alpha[i] + c[i]);
		beta[i + 1] = (f[i] - a[i] * beta[i]) / (a[i] * alpha[i] + c[i]);
	}

	// 2. Reverse way
	x_swp[n-1] = (f[n - 1] - beta[n - 1] * a[n - 1]) / (c[n - 1] + alpha[n - 1] * a[n - 1]);

	for (int i = n - 1; i > 0; i--)
		x_swp[i - 1] = alpha[i] * x_swp[i] + beta[i];

	free(alpha);
	free(beta);

	return;
}

void fill_continuation1_matrix(double *A, int ldA, double*a, double *b, double *c)
{
	int n = Nx + 1;
	int n2 = n*(Nt + 1);

#if (PROBLEM == 0)
	// Initial condition: U|t=0 = u0(x)
#ifdef OMP
#pragma omp parallel for simd schedule(simd:static) num_threads(OMP_THREADS)
#endif
	for (int i = 0; i < n; i++)
		A[i + ldA*i] = 1.0;

#elif (PROBLEM == 1)
	// Null flux condition: U|t=0 = 0;
#ifdef OMP
#pragma omp parallel for simd schedule(simd:static) num_threads(OMP_THREADS)
#endif
	for (int i = 0; i < n; i++)
	{
		A[i + ldA*i] = 1.0;
		A[i + ldA*(i + n)] = -1.0;
	}

	// Null flux condition: U|t=T = 0;
#ifdef OMP
#pragma omp parallel for simd schedule(simd:static) num_threads(OMP_THREADS)
#endif
	for (int i = 0; i < n; i++)
	{
		A[n2 + i + ldA*(i + n2 - 2*n)] = 1.0;
		A[n2 + i + ldA*(i + n2 - n)] = -1.0;
	}
#endif

	// First boundary condition: U|x=0 = f0(t)
#ifdef OMP
#pragma omp parallel for simd schedule(simd:static) num_threads(OMP_THREADS)
#endif
	for (int i = 0; i < n*n; i += n)
		A[i + ldA*i] = 1.0;

	// Second boundary condition: derivative Ux|x=0 = g0(t)
	double aw, bw, cw;
	cw = 1.0;
	aw = 2.0 * tau / (h*h) - 1;
	bw = -2.0 * tau / (h*h);

#ifdef OMP
#pragma omp parallel for simd schedule(simd:static) num_threads(OMP_THREADS)
#endif
	for (int i = n; i < n2; i += n)
	{
#ifdef SECOND
		A[i + n - 1 + ldA*i] = 1.0;
		A[i + n - 1 + ldA*(i - n)] = aw; 
		A[i + n - 1 + ldA*(i - n + 1)] = bw; 
#else
		A[i + n - 1 + ldA*(i)] = -1.0 / h; // first order
		A[i + n - 1 + ldA*(i + 1)] = 1.0 / h;
#endif
	}

	// Inner points
#ifdef OMP
#pragma omp parallel for simd schedule(simd:static) num_threads(OMP_THREADS)
#endif
	for (int j = 1; j < Nt + 1; j++)
		for (int i = 1; i < Nx; i++)
	{
		A[j*n + i + ldA*(j*n + i - 1)] = a[i];
		A[j*n + i + ldA*(j*n + i)] = c[i];
		A[j*n + i + ldA*(j*n + i + 1)] = b[i];
		A[j*n + i + ldA*(j*n + i - n)] = -1.0;
	}
	return;
}

void fill_continuation1_vectorF(double* F1, double* u0, double* g0, double* f0)
{
	int n = (Nx + 1);
	int n2 = n*n;
#if (PROBLEM == 0)
#ifdef OMP
#pragma omp parallel for simd schedule(simd:static) num_threads(OMP_THREADS)
#endif
	for (int i = 0; i < n; i++)
		F1[i] = u0[i];
#elif (PROBLEM == 1)
#ifdef OMP
#pragma omp parallel for simd schedule(simd:static) num_threads(OMP_THREADS)
#endif
	for (int i = 0; i < n; i++)
		F1[i] = 0;
#ifdef OMP
#pragma omp parallel for simd schedule(simd:static) num_threads(OMP_THREADS)
#endif
	for (int i = n2; i < n2 + n; i++)
		F1[i] = 0;
#endif

#ifdef OMP
#pragma omp parallel for simd schedule(simd:static) num_threads(OMP_THREADS)
#endif
	for (int i = n; i < n2; i += n)
	{
		F1[i] = f0[i/n];
#ifdef SECOND
		F1[i + n - 1] = g0[i / n] * tau*(-2.0 / h + kd_func(0) / k_func(0)); // second order
#else
		F1[i + n - 1] = g0[i / n]; // first order
#endif
	}

}

void fill_continuation2_matrix(double *A, int ldA, double*a, double *b, double *c)
{
	int n = Nx + 1;
	int n2 = n * n;
	
	fill_continuation1_matrix(A, ldA, a, b, c);

	// First boundary condition: U|x=L = fL(t)
	for (int i = n; i < n2; i += n)
		A[i + ldA*(i + n - 1)] = 1.0;

	double aw, bw, cw;
	cw = 1.0;
	aw = 2.0 * tau / (h*h) - 1;
	bw = -2.0 * tau / (h*h);

	// Second boundary condition: derivative Ux|x=L = gL(t)
#ifdef OMP
#pragma omp parallel for simd schedule(simd:static) num_threads(OMP_THREADS)
#endif
	for (int i = n; i < n2; i += n)
	{
#ifdef SECOND
		A[i + n - 1 + ldA*(i + n - 1)] = 1.0;  //second order
		A[i + n - 1 + ldA*(i - 1)] = aw;
		A[i + n - 1 + ldA*(i - 2)] = bw;
#else
		A[i + n - 1 + ldA*(i + n - 1)] = 0;
		A[i + n - 1 + ldA*(i + n - 1)] = 1.0 / h;
		A[i + n - 1 + ldA*(i + n - 2)] = -1.0 / h;
#endif
	}


	return;
}

void fill_continuation2_vectorF(double* F2, double* u0,  double *g0, double *gL, double *f0, double* fL)
{
	int n = (Nx + 1);
	int n2 = n * n;
	for (int i = 0; i < n; i++)
		F2[i] = u0[i];

#ifdef OMP
#pragma omp parallel for simd schedule(simd:static) num_threads(OMP_THREADS)
#endif
	for (int i = n; i < n2; i += n)
	{
		F2[i] = f0[i / n] + fL[i/n];
#ifdef SECOND
		F2[i + n - 1] = g0[i / n] * tau * (-2.0 / h + kd_func(0) / k_func(0)) +
			            gL[i / n] * tau * (-2.0 / h + kd_func(L) / k_func(L));
#else
		F2[i + n - 1] = g0[i / n] + gL[i / n];  // first order
#endif
	}

}

void validation_CP(int m, int n, double* A, int lda, double* u, double* F, int problem, double* f_diff)
{
	char nch = 'N';
	char FileName[255];
	double one = 1.0;
	int ione = 1;
	double zero = 0.0;
	double* y = new double[m];

	dgemv(&nch, &m, &n, &one, A, &lda, u, &ione, &zero, y, &ione);

#ifdef OMP
#pragma omp parallel for simd num_threads(2)
#endif
	for (int i = 0; i < m; i++)
		f_diff[i] = fabs(y[i] - F[i]);

	double norm = dnrm2(&n, f_diff, &ione);
	printf("\n");

	if (problem == 1)
	{
		if (norm < eps) printf("|A1*U - F1|: %8.6lf PASSED\n", norm);
		else printf("|A1*U - F1|: %8.6lf FAILED\n", norm);
		sprintf(FileName, "A1U_F1_Cross%5lf_L%2i_T%2i_Nx%3i_Nt%3i.dat", (double)CP, (int)L, (int)T, (int)Nx, (int)Nt);
	}
	else
	{
		if (norm < eps) printf("|A2*U - F2|: %8.6lf PASSED\n", norm);
		else printf("|A2*U - F2|: %8.6lf FAILED\n", norm);
		sprintf(FileName, "A2U_F2_Cross%5lf_L%2i_T%2i_Nx%3i_Nt%3i.dat", (double)CP, (int)L, (int)T, (int)Nx, (int)Nt);
	}

	FILE* out = fopen(FileName, "w");
	for (int i = 0; i < n; i++)
	{
		fprintf(out, "%.6d %23.20lf\n", i, f_diff[i]);
	}
	fclose(out);

	printf("\n");

	free(y);
}

double* diff_vec(double* u1, double* u2, int size, int problem)
{
	double* rez = new double[size];
	int ione = 1;
	for (int i = 0; i < size; i++)
	{
		rez[i] = u1[i] - u2[i];
		//printf("%lf  -  %lf   =   %lf\n", u1[i], u2[i], rez[i]);
	}
	
	double norm = dnrm2(&size, rez, &ione);

	if (problem < 5) printf("\nSolution: |U-U%d| = %20.17lf\n", problem, norm);

	free(rez);
	return rez;
}

double diff_mat(int m, int n, double* u_sol, double* u_ex, int ldu, double* rez, int problem)
{
	int size = m * n;
	int ione = 1;
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
		{
			rez[i + ldu * j] = u_sol[i + ldu * j] - u_ex[i + ldu * j];
		}

	double norm = dnrm2(&size, rez, &ione) / dnrm2(&size, u_ex, &ione);

	if (problem < 5) printf("\nSolution: |U_ex-U%d| = %20.17lf\n", problem, norm);

	// output
	
	char FileName2[255]; FILE* out2;
	sprintf(FileName2, "Diff_DirProb_InvContProb2D_full_Ns%d.dat", (int)NOISE);
	out2 = fopen(FileName2, "w");
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			fprintf(out2, "%.6d %.6d %17.10lf \n", i, j, fabs(rez[i + ldu*j]));
	fclose(out2);

	sprintf(FileName2, "Diff_Mat_DirProb_InvContProb2D_full_Ns%d.dat", (int)NOISE);
	out2 = fopen(FileName2, "w");
	for (int j = 0; j < n; j++)
	{
		for (int i = 0; i < m; i++)
			fprintf(out2, "%10.8lf ", fabs(rez[i + ldu*j]));
		fprintf(out2, "\n");
	}
	fclose(out2);

	return norm;
}

void cont_prob_by_inv_sch_no_bound(double* f0, int ldu, double* g0, double* a, double* b, double* c, double* u_invcp)
{
	int nx = (Nx + 1);
	int nt = (Nt + 1);
	int n2 = (Nx + 1)*(Nt + 1);
	char FileName2[255]; FILE* out2;

	u_invcp[0:n2] = 0;

#ifdef OMP
#pragma omp parallel for simd schedule(simd:static)
#endif
	for (int j = 0; j < nt; j++)
	{
		u_invcp[0 + ldu*j] = f0[j]; // already with noise
	}

#ifdef OMP
#pragma omp parallel for simd schedule(simd:static)
#endif
	for (int j = 1; j < nt-1; j++)
	{
		u_invcp[1 + ldu*j] = g0[j]*tau + u_invcp[0 + ldu*j];
	}

	for (int i = 2; i < nx; i++)
	{
#ifdef OMP
#pragma omp parallel for schedule(simd:static) if(i < nx - 4)
#endif
		for (int j = i; j < nt - i; j++)
			u_invcp[i + ldu*j] = (u_invcp[i - 1 + ldu*(j - 1)] - c[i - 1] * u_invcp[i - 1 + ldu*j] - b[i - 1] * u_invcp[i - 2 + ldu*j]) / a[i - 1];
	}


	sprintf(FileName2, "InvContProb2D_%d.dat", (int)NOISE);
	out2 = fopen(FileName2, "w");
	for (int i = 0; i < nx; i++)
		for (int j = 0; j < nt; j++)
		fprintf(out2, "%.6d %.6d %17.10lf \n", i, j, u_invcp[i + ldu*j]);
	fclose(out2);

	char FileName3[255]; FILE* out3;
	sprintf(FileName3, "InvContProb2D_triangle_%d.dat", (int)NOISE);
	out3 = fopen(FileName3, "w");
	for (int i = 0; i < nx; i++)
	for (int j = 0; j < nt; j++)
	{
		if (fabs(u_invcp[i + ldu*j]) > 0) fprintf(out3, "%.6d %.6d %17.10lf \n", i, j, u_invcp[i + ldu*j]);
	}

	fclose(out3);

}

void cont_prob_by_inv_sch_yes_bound(double* f0, int ldu, double* g0, double* a, double* b, double* c, double* u_invcp)
{
	int nx = (Nx + 1);
	int nt = (Nt + 1);
	int n2 = nx*nt;

	u_invcp[0:n2] = 0;
#ifdef OMP
#pragma omp parallel for simd schedule(simd:static)
#endif
	for (int j = 0; j < nt; j++)
	{
		u_invcp[0 + ldu*j] = f0[j];
	}
#ifdef OMP
#pragma omp parallel for simd schedule(simd:static)
#endif
	for (int j = 0; j < nt; j++)
	{
		u_invcp[1 + ldu*j] = g0[j] * tau + u_invcp[0 + ldu*j];
	}
	//  u_invcp[1 + ldu * 0] = u_invcp[1 + ldu * 1];
	//	u_invcp[1 + ldu * Nt] = u_invcp[1 + ldu * (Nt - 1)];

	for (int i = 2; i < nx; i++) // only sequential
	{
#ifdef OMP
#pragma omp parallel for simd schedule(simd:static)
#endif
		for (int j = 1; j < nt - 1; j++)
		{
			u_invcp[i + ldu*j] = (u_invcp[i - 1 + ldu*(j - 1)] - c[i - 1] * u_invcp[i - 1 + ldu*j] - b[i - 1] * u_invcp[i - 2 + ldu*j]) / a[i - 1];
		}
		u_invcp[i + ldu * 0] = u_invcp[i + ldu * 1];
		u_invcp[i + ldu * Nt] = u_invcp[i + ldu * (Nt - 1)];
	}
		

	char FileName2[255]; FILE* out2;
	sprintf(FileName2, "InvContProb2D_full_%d.dat", (int)NOISE);
	out2 = fopen(FileName2, "w");
	for (int i = 0; i < nx; i++)
	for (int j = 0; j < nt; j++)
		fprintf(out2, "%.6d %.6d %17.10lf \n", i, j, u_invcp[i + ldu*j]);
	fclose(out2);


}

void analit_cont_prob_vs_inv_sch_yes_bound(double* f, double* g0, double* a, double* b, double* c, double* u_invcp)
{
	int nx = (Nx + 1);
	int nt = (Nt + 1);
	int n2 = nx*nt;
	int ldu = nx;

	u_invcp[0:n2] = 0;
	for (int j = 0; j < nt; j++)
	{
		u_invcp[0 + ldu*j] = f[j];
	}

	for (int j = 0; j < nt; j++)
	{
		u_invcp[1 + ldu*j] = g0[j] * tau + u_invcp[0 + ldu*j];
	}
	//u_invcp[1 + ldu * 0] = u_invcp[1 + ldu * 1];
	//	u_invcp[1 + ldu * Nt] = u_invcp[1 + ldu * (Nt - 1)];

	for (int i = 2; i < nx; i++)
	{
		for (int j = 1; j < nt - 1; j++)
		{
			u_invcp[i + ldu*j] = (u_invcp[i - 1 + ldu*(j - 1)] - c[i - 1] * u_invcp[i - 1 + ldu*j] - b[i - 1] * u_invcp[i - 2 + ldu*j]) / a[i - 1];

		}
		u_invcp[i + ldu * 0] = u_invcp[i + ldu * 1];
		u_invcp[i + ldu * Nt] = u_invcp[i + ldu * (Nt - 1)];
	}

	char FileName2[255]; FILE* out2;
	sprintf(FileName2, "TestInvContProb2D_full.dat", (double)CP, (int)L, (int)T, (int)Nx, (int)Nt);
	out2 = fopen(FileName2, "w");
	for (int i = 0; i < nx; i++)
	for (int j = 0; j < nt; j++)
		fprintf(out2, "%.6d %.6d %17.10lf \n", i, j, u_invcp[i + ldu*j]);
	fclose(out2);

}

void direct_problem(double *u, int ldu, double *fi, double *gl, double *a, double *b, double *c, int exact)
{
	int iter = 0;
	int ione = 1;
	double one = 1.0;
	double zero = 0.0;
	int nx = Nx + 1;
	int nt = Nt + 1;

	// For MKL run
	int n = Nx + 1; // size of matrix
	double* A3diag = new double[n*n];

	// Solution for sweep and MKL
	double* x_swp = new double[nx];
	double* x_mkl = new double[nx];
	double* x_diff = new double[nx];

	// Right part for sweeping algorithm
	double* f = new double[nt];

	int lda = n;
	int nrhs = 1;
	int ldx = lda;
	int *ipiv = new int[nx];
	int info = 0;
	double norm = 0;

	// Conditions of direct problem

	if (exact == 0)
	{
#ifdef OMP
#pragma omp parallel for simd num_threads(2)
#endif
		for (int i = 0; i < nx; i++)
			u[i + ldu * 0] = fi_func(i*h);
	}
	else
	{
#ifdef OMP
#pragma omp parallel for simd num_threads(2)
#endif
		for (int i = 0; i < nx; i++)
			u[i + ldu * 0] = fi[i];
	}

	// Cycle for time step
	for (int j = 1; j < nt; j++)
	{
		iter++;

		f[0] = g0_func(tau*j);

		if (exact == 0)
		{
			f[Nx] = gL_func(tau*j);
		}
		else
		{
			f[Nx] = gl[j];
		}


#ifdef OMP
#pragma omp parallel for simd num_threads(2)
#endif
		for (int i = 1; i < Nx; i++)
			f[i] = u[i + ldu*(j - 1)];

		/************ Begin: Sweep solution *************/

		sweep(nx, a, b, c, f, x_swp); // a,b,c,f - size of Nx + 1

		/************* End: Sweep solution *************/

		// Transfering the sweep solution to array u
#ifdef OMP
#pragma omp parallel for simd num_threads(2)
#endif
		for (int i = 0; i < nx; i++)
			u[i + ldu*j] = x_swp[i];


		// Comparing with solution by MKL
		x_mkl[0:n] = f[0:n];
		A3diag[0:n*n] = 0;
#ifdef OMP
#pragma omp parallel for simd num_threads(2)
#endif
		for (int i = 0; i < nx; i++)
		{
			A3diag[i + lda * i] = c[i];
			if (i < Nx) A3diag[i + lda * (i + 1)] = b[i];
			if (i > 0)  A3diag[i + lda * (i - 1)] = a[i];
		}
		if (j == 1 || j == 2)
		{
			if (Nx == 5) print_mat(A3diag, f, n, n, n);
		}

		dgesv(&n, &nrhs, A3diag, &lda, ipiv, x_mkl, &ldx, &info);

#ifdef OMP
#pragma omp parallel for simd num_threads(2)
#endif
		for (int i = 0; i < nx; i++)
			x_diff[i] = x_swp[i] - x_mkl[i];

		norm = dnrm2(&n, x_diff, &ione);

		//if (norm < eps) printf("DP Iter: %03d Norm: %8.6lf PASSED\n", iter, norm);
		//else printf("DP Iter: %03d Norm: %8.6lf FAILED\n", iter, norm);

	}

	free(x_diff);
	free(x_mkl);
	free(x_swp);
	free(A3diag);
	free(ipiv);
	free(f);
}

void adjoint_problem(double *u_adj, int ldu_adj, double *u_dir, int ldu_dir, double *f /*exact data*/, double *a, double *b, double *c)
{
	int iter = 0;
	int ione = 1;
	double one = 1.0;
	double zero = 0.0;
	int nx = Nx + 1;
	int nt = Nt + 1;

	// For MKL run
	int n = Nx + 1; // size of matrix
	double* A3diag = new double[n*n];

	// Solution for sweep and MKL
	double* x_swp = new double[nx];
	double* x_mkl = new double[nx];
	double* x_diff = new double[nx];

	// Right part for sweeping algorithm
	double* F = new double[nt];

	int lda = n;
	int nrhs = 1;
	int ldx = lda;
	int *ipiv = new int[nx];
	int info = 0;
	double norm = 0;

	// Initial conditions
	for (int i = 0; i < nx; i++)
		u_adj[i + ldu_adj * 0] = 0;

	// Cycle for time step
	for (int j = 1; j < nt; j++) // tau = T - t; 
	{
		iter++;

		F[0] = 2.0 * (u_dir[0 + ldu_dir * (Nt - j)] - f[(Nt - j)]); // inverse order due to change of variables
		F[Nx] = 0;

#ifdef OMP
#pragma omp parallel for simd num_threads(2)
#endif
		for (int i = 1; i < Nx; i++)
			F[i] = u_adj[i + ldu_adj*(j - 1)];

		/************ Begin: Sweep solution *************/

		sweep(nx, a, b, c, F, x_swp); // a,b,c,f - size of Nx + 1

		//************* End: Sweep solution ***********/

		// Transfering the sweep solution to array u
#ifdef OMP
#pragma omp parallel for simd num_threads(2)
#endif
		for (int i = 0; i < nx; i++)
			u_adj[i + ldu_adj*j] = x_swp[i];


		// ------------Comparing with solution by MKL----------------
		x_mkl[0:n] = F[0:n];
		A3diag[0:n*n] = 0;
#ifdef OMP
#pragma omp parallel for simd num_threads(2)
#endif
		for (int i = 0; i < nx; i++)
		{
			A3diag[i + lda * i] = c[i];
			if (i < Nx) A3diag[i + lda * (i + 1)] = b[i];
			if (i > 0)  A3diag[i + lda * (i - 1)] = a[i];
		}
		if (j == 1 || j == 2)
		{
			if (Nx == 5) print_mat(A3diag, F, n, n, n);
		}

		dgesv(&n, &nrhs, A3diag, &lda, ipiv, x_mkl, &ldx, &info);

#ifdef OMP
#pragma omp parallel for simd num_threads(2)
#endif
		for (int i = 0; i < nx; i++)
			x_diff[i] = x_swp[i] - x_mkl[i];

		norm = dnrm2(&n, x_diff, &ione);

		//	if (norm < eps) printf("IP Iter: %03d Norm: %8.6lf PASSED\n", iter, norm);
		//  else printf("IP Iter: %03d Norm: %8.6lf FAILED\n", iter, norm);

	}

	free(x_diff);
	free(x_mkl);
	free(x_swp);
	free(A3diag);
	free(ipiv);
	free(F);
}

void gradient(double *u_adj, int ldu, double *grad1, double *grad2)
{
	int nx = Nx + 1;
	int nt = Nt + 1;

	double inv = 1.0 / h;

	for (int i = 0; i < nx; i++)
		grad1[i] = u_adj[i + ldu * Nt];

	for (int j = 0; j < nt; j++)
	{
		grad2[j] = u_adj[Nx + ldu * (Nt - j)];
		//printf("%lf\n", grad2[j]);
	}

}

double functional(double *u_dir, int ldu, double *f /*exact data*/)
{
	int nt = Nt + 1;
	int one = 1.0;
	double J = 0;

	for (int j = 0; j < nt; j++)
		J += tau * (u_dir[0 + ldu * (j)] - f[j]) * (u_dir[0 + ldu * (j)] - f[j]);

	return J;
}

void matrix_to_file(int m, int n, double *u_adj, int ldu, int iter)
{
	FILE *out;
	char FileName[255];

	sprintf(FileName, "adj_problem_%d.dat", iter);

	out = fopen(FileName, "w+");

	for (int j = 0; j < n; j++)
	{
		for (int i = 0; i < n; i++)
			fprintf(out, "%lf ", u_adj[i + ldu * j]);
		fprintf(out,"\n");
	}

	fclose(out);
}


void cont_prob_by_grad_meth(double *u_dir, int ldu_dir, double *u_ex, int ldu_ex, double *f_data, double *a, double *b, double *c, double *q0_fi, double *q1_gl)
{
	double norm = 0, norm_prev = 100, norm_sol = 0, norm_fi = 0, norm_gl = 0;
	int nx = Nx + 1;
	int nt = Nt + 1;
	int n2 = nx * nt;
	int ldu = nx;
	int one = 1;
	
	double *grad1 = new double[nx];
	double *grad2 = new double[nt];

	double *fi_exact = new double[nx];
	double *gl_exact = new double[nt];

	double *fi_diff = new double[nx];
	double *gl_diff = new double[nt];

	double *u_adj = new double[n2];
	double *u_diff = new double[n2];
	int ldu_adj = ldu_dir;

	double *J = new double[nt];

	int iter = 0;
	double alpha1 = 0.5, alpha2 = 0.5;


#pragma omp parallel
	{
		// Copy exact solution fi
#pragma omp for simd schedule(simd:static)
		for (int i = 0; i < nx; i++)
		{
			fi_exact[i] = fi_func(h*i);
		}

		// Copy exact solution gl
#pragma omp for simd schedule(simd:static)
		for (int i = 0; i < nt; i++)
		{
			gl_exact[i] = gL_func(tau*i);
		}

		// Initial aprroximation
#pragma omp for simd schedule(simd:static)	
		for (int i = 0; i < nx; i++)
		{
			q0_fi[i] = f_data[0];
			//q0_fi[i] = f_data[i];
		}

#pragma omp for simd schedule(simd:static)	
		for (int i = 0; i < nt; i++)
		{
			q1_gl[i] = 0.5;
			//q1_gl[i] = gL_func(tau*i);
		}
	}

	do
	{
		iter++;

		// Solve direct problem
		direct_problem(u_dir, ldu_dir, q0_fi, q1_gl, a, b, c, 1);

		// Solve adjoint problemâ
		adjoint_problem(u_adj, ldu_adj, u_dir, ldu_dir, f_data, a, b, c);

		// Write to file
		matrix_to_file(nx, nt, u_adj, ldu, iter);

		// Compute gradient of the functional
		gradient(u_adj, ldu_adj, grad1, grad2);

		// Compute functional
		norm = functional(u_dir, ldu_dir, f_data);

		// Compute difference norm between u_dir and u_ex
		norm_sol = diff_mat(nx, nt, u_dir, u_ex, ldu_dir, u_diff, 5);

#pragma omp parallel
		{
			// Next step descent
#pragma omp for simd schedule(simd:static)	
			for (int i = 0; i < nx; i++)
			{
				q0_fi[i] += alpha1 * grad1[i];
			}
#pragma omp for simd schedule(simd:static)	
			for (int i = 0; i < nt; i++)
			{
				q1_gl[i] += alpha2 * grad2[i];
			}

			// Compute norm for jl and fi
#pragma omp for simd schedule(simd:static)		
			for (int i = 0; i < nx; i++)
			{
				fi_diff[i] = q0_fi[i] - fi_exact[i];
			}
#pragma omp for simd schedule(simd:static)		
			for (int i = 0; i < nt; i++)
			{
				gl_diff[i] = q1_gl[i] - gl_exact[i];
			}
		}

		norm_fi = dnrm2(&nx, fi_diff, &one) / dnrm2(&nx, fi_exact, &one);
		norm_gl = dnrm2(&nt, gl_diff, &one) / dnrm2(&nt, gl_exact, &one);

		printf("\nIter: %d, func: %10.8lf  sol:%10.8lf  fi:%10.8lf  gl:%10.8lf\n", iter, norm, norm_sol, norm_fi, norm_gl);

		//if (NOISE != 0 && norm_sol < 0.05) break;

	} while (norm > eps);

	// output
	printf("End of iterations\n");

	printf("Output\n");
	
	FILE* out;
	char FileName[255];
	sprintf(FileName, "ip_gradient_iter%d_noise%d.dat", iter, int(NOISE));

	out = fopen(FileName, "w+");
	for (int i = 0; i < nx; i++)
		for (int j = 0; j < nt; j++)
			fprintf(out, "%.6d %.6d %17.10lf \n", i, j, u_dir[i + ldu_dir*j]);
	fclose(out);

	free(grad1);
	free(grad2);
	free(fi_diff);
	free(gl_diff);
	free(fi_exact);
	free(gl_exact);
	free(u_adj);
	free(u_diff);
	free(J);

}

long N1rand = 101; long N2rand = 302;

/************* Random value *******************/
double rand1(void)
{
	double x; long k;
	k = x = ((double)(N1rand = ((N1rand *= 1373) % 1000919)) / 1000919.0 +
		(double)(N2rand = ((N2rand *= 1528) % 1400159)) / 1400159.0);
	return(x - k);
}

void output_SVD(int n, double *uc1, int ldu, double *sing1, int cross1)
{
	FILE *out;
	char FileName[255];

	int n2 = n * n;

	// matrix A [n2_big:n2]

	printf("Singular values:\n");
	for (int i = 0; i < n2; i++)
		printf("%2d  %23.20lf\n", i, sing1[i]);

	printf("\n\n");


	sprintf(FileName, "SingValues_Cross%5lf_L%2i_T%2i_Nx%3i_Nt%3i.dat", (double)CP, (int)L, (int)T, (int)Nx, (int)Nt);
	out = fopen(FileName, "w");
	for (int i = 0; i < n2; i++)
	{
		fprintf(out, "%.6d %23.20lf \n", i, log10(sing1[i]));
	}
	fprintf(out, "\nCross1: %5d\n", cross1);
	fprintf(out, "\nCond1: %13.10lf\n", sing1[0] / sing1[cross1]);
	fclose(out);


	printf("Cond A1: %8.6lf / %8.6lf = %8.6lf\n", sing1[0], sing1[n2 - 1], sing1[0] / sing1[n2 - 1]);


	for (int j = cross1 + 1; j < n2; j++)
		sing1[j] = 0;

	printf("Cross1: %d, Cond A1: %13.10lf / %13.10lf = %13.10lf\n", cross1, sing1[0], sing1[cross1], sing1[0] / sing1[cross1]);


	sprintf(FileName, "ContProb1D_U1_Cross%4f_L%2d_T%2d_Nx%3d_Nt%3d_Ns%d.dat", (double)CP, (int)L, (int)T, (int)Nx, (int)Nt, (int)NOISE);
	out = fopen(FileName, "w");

	for (int i = 0; i < n2; i++)
		fprintf(out, "%.6d %17.10lf \n", i, uc1[i]);

	fclose(out);

	sprintf(FileName, "ContProb2D_U1_Cross%4f_L%2d_T%2d_Nx%3d_Nt%3d_Ns%d.dat", (double)CP, (int)L, (int)T, (int)Nx, (int)Nt, (int)NOISE);
	out = fopen(FileName, "w");
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			fprintf(out, "%.6d %.6d %17.10lf \n", i, j, uc1[i + ldu*j]);
	fclose(out);

#ifdef CP2
	sprintf(FileName, "ContProb2D_U2_Cross%4f_L%2d_T%2d_Nx%3d_Nt%3d_Ns%d.dat", (double)CP, (int)L, (int)T, (int)Nx, (int)Nt, (int)NOISE);
	out2 = fopen(FileName, "w");
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			fprintf(out2, "%.6d %.6d %17.10lf \n", i, j, uc2[i + ldu*j]);
	fclose(out2);
#endif
}

void output_DP(int n, double *u, int ldu)
{
	char FileName[255];

	sprintf(FileName, "DirectProb1D_L%2d_T%2d_Nx%3d_Nt%3d.dat", (int)L, (int)T, (int)Nx, (int)Nt);
	FILE* out = fopen(FileName, "w");
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			fprintf(out, "%.6d %17.10lf \n", i + ldu*j, u[i + ldu*j]);
	fclose(out);

	sprintf(FileName, "DirectProb2D_L%2d_T%2d_Nx%3d_Nt%3d.dat", (int)L, (int)T, (int)Nx, (int)Nt);
	out = fopen(FileName, "w");
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			fprintf(out, "%.6d %.6d %17.10lf \n", i, j, u[i + ldu*j]);
	fclose(out);
}

void mem_free(int n, double** arrays)
{
	for (int i = 0; i < n; i++)
		free(arrays[i]);
}