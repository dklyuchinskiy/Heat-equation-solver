#include "header.h"

int main()
{
	int i, j;
	int iter = 0;
	int ione = 1;
	double one = 1.0;
	double zero = 0.0;
	double rnd, rnd2;
	char tran = 'T';
	char notran = 'N';
	char uplo = 'A';
	int nx = Nx + 1;
	int nt = Nt + 1;

	// Solution for the whole domain of DIRECT PROBLEM
	double* u = new double[(Nx + 1)*(Nt + 1)];
	double* u_ex = new double[(Nx + 1)*(Nt + 1)];
	int ldu = nx;

	// Solution for the whole domain of CONTINUATION PROBLEMS
	double* uc1 = new double[(Nx + 1)*(Nt + 1)];
	double* uc2 = new double[(Nx + 1)*(Nt + 1)];
	double* u_rez = new double[(Nx + 1)*(Nt + 1)];

	// Analitical solution for the whole domain of CONTINUATION PROBLEM for k=1
	double* uc_analit = new double[(Nx + 1)*(Nt + 1)];

	// Solution for triangle inverse CONTINUATION PROBLEM
	double* u_invcp = new double[(Nx + 1)*(Nt + 1)];

	// Coefficients for sweep
	double *a, *b, *c;
	c = new double[Nx + 1];
	a = new double[Nx + 1];
	b = new double[Nx + 1];

	// Function k(x)
	double *k = new double[Nx + 1];

	// Function f0(t) and fL(t)
	double *f0 = new double[Nt + 1];
	double *fL = new double[Nt + 1];

	// Function g0(t) and gL(t)
	double *g0 = new double[Nt + 1];
	double *gL = new double[Nt + 1];

	// Function phi(t) and gL(t) for IP solution by gradient methods
	double *fi_grad = new double[Nt + 1];
	double *gl_grad = new double[Nt + 1];

	// Function u0(x)
	double *u0 = new double[Nx + 1];

	// Exact data of DP solution u|x=0 = f(t)
	double *f_data = new double[Nt + 1];

	// For MKL run
	int n = Nx + 1; // size of matrix
	double* A3diag = new double[n*n];

	// For RNG
	double *rnd_vec1 = new double[Nt + 1];
	double *rnd_vec2 = new double[Nt + 1];

	VSLStreamStatePtr stream;
	int status = 0;
	

	int lda = n;
	int nrhs = 1;
	int ldx = lda;
	int *ipiv = new int[Nx + 1];
	int info = 0;
	double norm = 0;

	double t1, t2;

	clock_t start = clock();

	// Fullfilling vectors
#ifdef OMP
#pragma omp parallel for simd num_threads(OMP_THREADS)
#endif
	for (int i = 0; i < Nx + 1; i++)
	{
		k[i] = k_func(i*h);
		u0[i] = fi_func(i*h);
	}


	// Column major: a[i][j] = a[i+lda*j]
	a[0] = 0.0;  c[0] = -1 / h; b[0] = 1 / h;
	a[Nx] = -1 / h; c[Nx] = 1 / h; b[Nx] = 0.0;


	// Fullfilling coefficints of A3diag matrix
#ifdef OMP
#pragma omp parallel for simd num_threads(OMP_THREADS)
#endif
	for (int i = 1; i < Nx; i++)
	{
		a[i] = -tau / (h*h) * (k[i] + k[i - 1]) / 2.0;
		b[i] = -tau / (h*h) * (k[i] + k[i + 1]) / 2.0;
		c[i] = 1 - a[i] - b[i];
	}
	
	// Solving direct problem to get exact data
	direct_problem(u, ldu, NULL, NULL, a, b, c, 0);

	dlacpy(&uplo, &nx, &nt, u, &ldu, u_ex, &ldu);

	output_DP(nx, u_ex, ldu);

	/******************************************
	******Continuation problem solution ********
	*******************************************/

	int n2 = nx * nt;
	int n2_low = nx * (nt - 1);
	int n2_big = nx * (nt + 1);
	int n4 = n2 * n2;
	double* result = new double[n2];

	// Comparison of analytical solution and SCHEME INVERSE algorithm
#ifdef ANALYTIC
	g0[0:nt] = 0;
	for (int j = 0; j < Nt + 1; j++)
		f0[j] = 2 * 0.1*sin(2 * 0.01*j*tau);

	// Finding solution by Inverse scheme algorithm
	analit_cont_prob_vs_inv_sch_yes_bound(f0, g0, a, b, c, u_invcp);

	for (int i = 0; i < nx; i++)
	for (int j = 0; j < nt; j++)
		uc_analit[i + j*ldu] = 0.1*(exp(0.1*i*h)*sin(2 * 0.01*j*tau + 0.1*i*h) + exp(-0.1*i*h)*sin(2 * 0.01*j*tau - 0.1*i*h));

	char FileName5[255]; FILE* out5;
	sprintf(FileName5, "AnalitContProb2D_full.dat", (double)CP, (int)L, (int)T, (int)Nx, (int)Nt);
	out5 = fopen(FileName5, "w");
	for (int i = 0; i < nx; i++)
	for (int j = 0; j < nt; j++)
		fprintf(out5, "%.6d %.6d %17.10lf \n", i, j, uc_analit[i + ldu*j]);
	fclose(out5);

	diff_mat(nx, nt, uc_analit, u_invcp, nx, result, 4);
	system("pause");
#endif

	// Data for continuation problem!
#ifndef ANALYTIC

	vslNewStream(&stream, VSL_BRNG_MT19937, Nt + 1);

	status = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, stream, Nt + 1, rnd_vec1, -1, 1);
	status = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, stream, Nt + 1, rnd_vec2, -1, 1);

	vslDeleteStream(&stream);

	//for (int i = 0; i < Nt + 1; i++)
		//printf("%lf %lf\n", rnd_vec1[i], rnd_vec2[i]);

	//system("pause");
	
#ifdef OMP
#pragma omp parallel for simd num_threads(2)
#endif
	for (int j = 0; j < Nt + 1; j++)
	{
		f0[j] = u[0 + ldu*j] * (1.0 + NOISE * rnd_vec1[j] / 100.0);
		fL[j] = u[Nx + ldu*j] * (1.0 + NOISE * rnd_vec2[j] / 100.0);
	}

	if (Nx == 5)
	{
		for (int j = 0; j < Nt + 1; j++)
			printf("F: %5.3lf + %5.3lf =  %5.3lf\n", f0[j], fL[j], f0[j] + fL[j]);
	}

#ifdef OMP
#pragma omp parallel for simd num_threads(2)
#endif
	for (int j = 0; j < Nt + 1; j++)
	{
		g0[j] = g0_func(j*tau);
		gL[j] = gL_func(j*tau);
	}

	for (int j = 0; j < Nt + 1; j++)
	{
		if (Nx == 5) printf("G: %5.3lf + %5.3lf =  %5.3lf\n", g0[j], gL[j], g0[j] + gL[j]);
	}


	/******************************************
	********* 1. Gradient method  ***********
	*****************************************/

	printf("Gradient method computing...\n");
	t1 = omp_get_wtime();
	cont_prob_by_grad_meth(u, ldu, u_ex, ldu, f0, a, b, c, fi_grad, gl_grad);
	t2 = omp_get_wtime() - t1;
	printf("Time grad method: %lf\n", t2);
	diff_mat(nx, nt, u_ex, u, ldu, result, 1);

	/********************************************
	********* 2. Scheme Inversion ***************
	*********************************************/

	printf("FDSI...\n");
	t1 = omp_get_wtime();
	cont_prob_by_inv_sch_no_bound(f0, ldu, g0, a, b, c, u_invcp);
	cont_prob_by_inv_sch_yes_bound(f0, ldu, g0, a, b, c, u_invcp);
	t2 = omp_get_wtime() - t1;
	printf("Time fdsi: %lf\n", t2);
	diff_mat(nx, nt, u_ex, u_invcp, ldu, result, 2);

#endif

	/**************************************
	 **************** 3. SVD **************
	/**************************************/

	printf("SVD...\n");
#if (PROBLEM == 1)
	// size of matrix A: [n2_big:n2], where n2_big - number of rows, n2 - number of columns
	// number of columns is equal to the number of unknowns of solution vector

	n = (Nx + 1);
	n2 = (Nx + 1)*(Nt + 1);
	n2_big = (Nx + 1)*(Nt + 1) + (Nx + 1); 
	n4 = n2_big * n2_big; // the bigger dimension
#endif
	double* A1 = new double[n4]; A1[0:n4] = 0;
	double* A2 = new double[n4]; A2[0:n4] = 0;
	double *F1 = new double[n2_big]; F1[0:n2_big] = 0;
	double *F2 = new double[n2_big]; F2[0:n2_big] = 0;
	int ldA = n2_big;
	int cross1 = n2 - 1, cross2 = n2 - 1;

	// Check analytical solution 
#ifdef ANALYTIC
	for (int i = 0; i < nx; i++)
	for (int j = 0; j < nt; j++)
		u[i + ldu*j] = uc_analit[i + ldu*j];
#endif

	// Constructing matrix A1 and vector F1
	printf("Fulfilling of A1 matrix and vector F1...\n");
	fill_continuation1_matrix(A1, ldA, a, b, c);
	fill_continuation1_vectorF(F1, u0, g0, f0);

	// For SVD decompisition
	double *U = new double[n4]; 
	double *VT = new double[n4];
	double *g_svd = new double[n2_big]; g_svd[0:n2_big] = 0;
	double *z_svd = new double[n2_big]; z_svd[0:n2_big] = 0;
	double *q_svd = new double[n2_big]; q_svd[0:n2_big] = 0;

	char no = 'N';
	char A = 'A';
	double *sing1 = new double[n2];
	double *sing2 = new double[n2];
	int ldVT = n2_big;
	int ldU = n2_big;
	int lwork = -1;
	double *work;
	double wquery;
	double* f_diff = new double[n2_big];

#if (PROBLEM == 0)
	// Check A1 and F1 with Direct solution: AU=F
	validation_CP(n21, n2, &A1[n + ldA * 0], ldA, u, &F1[n], 1, f_diff);

	if (Nx == 5) print_mat(&A1[n + ldA * 0], &F1[n], n21, n2, ldA);

	// SVD query
	dgesvd(&A, &A, &n21, &n2, &A1[n+ldA*0], &ldA, sing1, U, &ldU, VT, &ldVT, &wquery, &lwork, &info);
	lwork = int(wquery);
	work = new double[lwork];

	//SVD
	printf("SVD of A1 is computing...\n");
	dgesvd(&A, &A, &n21, &n2, &A1[n + ldA * 0], &ldA, sing1, U, &ldU, VT, &ldVT, work, &lwork, &info);
	free(work);
	printf("\n");
//	system("pause");

	for (int i = 0; i < n2; i++)
	{
		if (sing1[i] < double(CP))
		{
			cross1 = i - 1;
			break;
		}
	}

	// computing g = UT * f
	dgemv(&tran, &n21, &n21, &one, U, &ldU, F1, &ione, &zero, g_svd, &ione);

	z_svd[0:n2] = 0;
	// computing z[i] = g[i] / sing [i]
	for (int i = 0; i < n21; i++)
		if (i<= cross1) z_svd[i] = g_svd[i] / sing1[i];
		else z_svd[i] = 0;

	// computing z = VT * q  ==>  q = V * z
	dgemv(&tran, &n2, &n2, &one, VT, &ldVT, z_svd, &ione, &zero, uc1, &ione);

#elif (PROBLEM==1)

	t1 = omp_get_wtime();

	// Check A1 and F1 with Direct solution: AU=F
	validation_CP(n2_big, n2, &A1[0 + ldA * 0], ldA, u_ex, &F1[0], 1, f_diff);

	if (Nx == 5) print_mat(&A1[0 + ldA * 0], &F1[0], n2_big, n2, ldA);

	// SVD query
	dgesvd(&A, &A, &n2_big, &n2, &A1[0 + ldA * 0], &ldA, sing1, U, &ldU, VT, &ldVT, &wquery, &lwork, &info);
	lwork = int(wquery);
	work = new double[lwork];

	// SVD
	printf("SVD of A1 is computing...\n");
#ifndef LAPACKE
	dgesvd(&A, &A, &n2_big, &n2, &A1[0 + ldA * 0], &ldA, sing1, U, &ldU, VT, &ldVT, work, &lwork, &info);
#else
	double* superb = new double[n2_big];
	LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'A', 'A', n2_big, n2, A1, n2_big, sing1, U, n2_big, VT, n2_big, superb);
#endif
	free(work);
	printf("\n");

	for (int i = 0; i < n2; i++)
	{
		if (sing1[i] < double(CP))
		{
			cross1 = i - 1;
			break;
		}
	}

	printf("computing g = UT * f...\n");
	dgemv(&tran, &n2_big, &n2_big, &one, U, &ldU, F1, &ione, &zero, g_svd, &ione);

	z_svd[0:n2] = 0;
	printf("computing z[i] = g[i] / sing [i]...\n");
	for (int i = 0; i < n2; i++)
		if (i <= cross1) z_svd[i] = g_svd[i] / sing1[i];
		else z_svd[i] = 0;

	printf("computing z = VT * q  ==>  q = V * z...\n");
	dgemv(&tran, &n2, &n2, &one, VT, &ldVT, z_svd, &ione, &zero, uc1, &ione);

	t2 = omp_get_wtime() - t1;
	printf("Time SVD: %lf\n", t2);

	diff_mat(nx, nt, u_ex, uc1, ldu, result, 3);

#endif

#ifdef CP2
	//------------------- 2. Constructing matrix A2 and vector F2 --------------------------------------
	fill_continuation2_matrix(A2, ldA, a, b, c);
	fill_continuation2_vectorF(F2, u0, g0, gL, f0, fL);

	validation_CP(n2_1, n2, &A2[n + ldA * 0], ldA, u, &F2[n], 2, f_diff);
	
	if (Nx == 5) print_mat(&A2[n + ldA * 0], &F2[n], n2_1, n2, ldA);

	// SVD query
	lwork = -1;
	dgesvd(&A, &A, &n2_1, &n2, &A2[n + ldA * 0], &ldA, sing2, U, &ldU, VT, &ldVT, &wquery, &lwork, &info);
	lwork = int(wquery);
	work = new double[lwork];

	//SVD
	printf("SVD of A2 is computing...\n");
	dgesvd(&A, &A, &n2_1, &n2, &A2[n + ldA * 0], &ldA, sing2, U, &ldU, VT, &ldVT, work, &lwork, &info);
	free(work);
//	system("pause");
	for (int i = 0; i < n2_1; i++)
	{
		if (sing2[i] < double(CP))
		{
			cross2 = i - 1;
			break;
		}
	}

	// computing g = UT * f
	dgemv(&tran, &n2_1, &n2_1, &one, U, &ldU, F2, &ione, &zero, g_svd, &ione);

	z_svd[0:n2] = 0;
	// computing z[i] = g[i] / sing [i]
	for (int i = 0; i < n2_1; i++)
	if (i <= cross2) z_svd[i] = g_svd[i] / sing2[i];
	else z_svd[i] = 0;

	// computing q = V * z
	dgemv(&tran, &n2, &n2, &one, VT, &ldVT, z_svd, &ione, &zero, uc2, &ione);
#endif
	clock_t end = clock() - start;

	output_SVD(n, uc1, ldu, sing1, cross1);

	int time = end * 1000 / CLOCKS_PER_SEC ;

	printf("Time: %d msec, %d sec\n", time, time/1000);

	double **arrays = new double*[22];
	arrays[0] = u;
	arrays[1] = u_ex;
	arrays[2] = uc1;
	arrays[3] = uc2;
	arrays[4] = u_rez;
	arrays[5] = u_invcp;
	arrays[6] = uc_analit;
	arrays[7] = a;
	arrays[8] = b;
	arrays[9] = c;
	arrays[10] = u0;
	arrays[11] = f0;
	arrays[12] = k;
	arrays[13] = f_data;
	arrays[14] = f_diff;
	arrays[15] = fi_grad;
	arrays[16] = gl_grad;
	arrays[17] = U;
	arrays[18] = VT;
	arrays[19] = q_svd;
	arrays[20] = g_svd;
	arrays[21] = z_svd;

	mem_free(22, arrays); // the name of array is already a pointer
	free(arrays);

	system("pause");
	return 0;
}