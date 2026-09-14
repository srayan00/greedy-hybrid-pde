/* Compiled periodic 5-point stencil kernels (double precision, batch of B fields of N x N,
   row-major, first index i = x, second j = y), matching fast_pde.FastStencilPDE exactly:
     A u = (ax (2u - u_{i+1} - u_{i-1}) + ay (2u - u_{j+1} - u_{j-1})) / h^2
           + b1 (u_{i+1} - u_{i-1}) / (2h) + b2 (u_{j+1} - u_{j-1}) / (2h)
   Interior columns are handled without index arithmetic; the two boundary columns wrap.
   Build: cc -O3 -shared -fPIC -o libstencil.so stencil.c  */
#include <stddef.h>

typedef struct { double diag, inv, cw, ce, cs, cn; } St;

static inline St coeffs(int N, double ax, double ay, double b1, double b2) {
    double h = 1.0 / N, h2 = h * h; St s;
    s.diag = 2.0 * (ax + ay) / h2; s.inv = 1.0 / s.diag;
    s.cw = -ax / h2 - b1 / (2 * h); s.ce = -ax / h2 + b1 / (2 * h);
    s.cs = -ay / h2 - b2 / (2 * h); s.cn = -ay / h2 + b2 / (2 * h);
    return s;
}

#define ROW_SETUP(U, i) \
    int im = (i == 0) ? N - 1 : i - 1, ip = (i == N - 1) ? 0 : i + 1; \
    const double *rm = U + (size_t)im * N, *rp = U + (size_t)ip * N;

void apply_A(const double *u, double *out, int B, int N, double ax, double ay, double b1, double b2) {
    St s = coeffs(N, ax, ay, b1, b2);
    for (int b = 0; b < B; b++) {
        const double *U = u + (size_t)b * N * N; double *O = out + (size_t)b * N * N;
        for (int i = 0; i < N; i++) {
            ROW_SETUP(U, i)
            const double *r = U + (size_t)i * N; double *o = O + (size_t)i * N;
            o[0] = s.diag * r[0] + s.cw * rm[0] + s.ce * rp[0] + s.cs * r[N - 1] + s.cn * r[1];
            for (int j = 1; j < N - 1; j++)
                o[j] = s.diag * r[j] + s.cw * rm[j] + s.ce * rp[j] + s.cs * r[j - 1] + s.cn * r[j + 1];
            o[N - 1] = s.diag * r[N - 1] + s.cw * rm[N - 1] + s.ce * rp[N - 1] + s.cs * r[N - 2] + s.cn * r[0];
        }
    }
}

void residual(const double *u, const double *f, double *r, int B, int N, double ax, double ay, double b1, double b2) {
    St s = coeffs(N, ax, ay, b1, b2);
    for (int b = 0; b < B; b++) {
        const double *U = u + (size_t)b * N * N, *F = f + (size_t)b * N * N; double *R = r + (size_t)b * N * N;
        for (int i = 0; i < N; i++) {
            ROW_SETUP(U, i)
            const double *ru = U + (size_t)i * N, *fr = F + (size_t)i * N; double *o = R + (size_t)i * N;
            o[0] = fr[0] - (s.diag * ru[0] + s.cw * rm[0] + s.ce * rp[0] + s.cs * ru[N - 1] + s.cn * ru[1]);
            for (int j = 1; j < N - 1; j++)
                o[j] = fr[j] - (s.diag * ru[j] + s.cw * rm[j] + s.ce * rp[j] + s.cs * ru[j - 1] + s.cn * ru[j + 1]);
            o[N - 1] = fr[N - 1] - (s.diag * ru[N - 1] + s.cw * rm[N - 1] + s.ce * rp[N - 1] + s.cs * ru[N - 2] + s.cn * ru[0]);
        }
    }
}


/* residual r = f - A u and its squared L2 norm, in one pass (the stopping test of every method) */
double residual_norm2(const double *u, const double *f, double *r, int B, int N, double ax, double ay, double b1, double b2) {
    St s = coeffs(N, ax, ay, b1, b2); double acc = 0.0;
    for (int b = 0; b < B; b++) {
        const double *U = u + (size_t)b * N * N, *F = f + (size_t)b * N * N; double *R = r + (size_t)b * N * N;
        for (int i = 0; i < N; i++) {
            ROW_SETUP(U, i)
            const double *ru = U + (size_t)i * N, *fr = F + (size_t)i * N; double *o = R + (size_t)i * N;
            double v;
            v = fr[0] - (s.diag * ru[0] + s.cw * rm[0] + s.ce * rp[0] + s.cs * ru[N - 1] + s.cn * ru[1]); o[0] = v; acc += v * v;
            for (int j = 1; j < N - 1; j++) {
                v = fr[j] - (s.diag * ru[j] + s.cw * rm[j] + s.ce * rp[j] + s.cs * ru[j - 1] + s.cn * ru[j + 1]); o[j] = v; acc += v * v;
            }
            v = fr[N - 1] - (s.diag * ru[N - 1] + s.cw * rm[N - 1] + s.ce * rp[N - 1] + s.cs * ru[N - 2] + s.cn * ru[0]); o[N - 1] = v; acc += v * v;
        }
    }
    return acc;
}

/* fused (damped) Jacobi: out = u + w/diag * (f - A u) */
void jacobi(const double *u, const double *f, double *out, int B, int N, double ax, double ay, double b1, double b2, double w) {
    St s = coeffs(N, ax, ay, b1, b2); double sc = w * s.inv;
    for (int b = 0; b < B; b++) {
        const double *U = u + (size_t)b * N * N, *F = f + (size_t)b * N * N; double *O = out + (size_t)b * N * N;
        for (int i = 0; i < N; i++) {
            ROW_SETUP(U, i)
            const double *r = U + (size_t)i * N, *fr = F + (size_t)i * N; double *o = O + (size_t)i * N;
            o[0] = r[0] + sc * (fr[0] - (s.diag * r[0] + s.cw * rm[0] + s.ce * rp[0] + s.cs * r[N - 1] + s.cn * r[1]));
            for (int j = 1; j < N - 1; j++)
                o[j] = r[j] + sc * (fr[j] - (s.diag * r[j] + s.cw * rm[j] + s.ce * rp[j] + s.cs * r[j - 1] + s.cn * r[j + 1]));
            o[N - 1] = r[N - 1] + sc * (fr[N - 1] - (s.diag * r[N - 1] + s.cw * rm[N - 1] + s.ce * rp[N - 1] + s.cs * r[N - 2] + s.cn * r[0]));
        }
    }
}

/* in-place lexicographic SOR sweep (omega = 1: Gauss-Seidel); forward = 1 or backward = 0.
   Equals u + omega (D + omega L)^{-1} (f - A u) with L the strict lower part in row-major order. */
void sor_sweep(double *u, const double *f, int B, int N, double ax, double ay, double b1, double b2, double omega, int forward) {
    St s = coeffs(N, ax, ay, b1, b2); double om = 1.0 - omega, sc = omega * s.inv;
    for (int b = 0; b < B; b++) {
        double *U = u + (size_t)b * N * N; const double *F = f + (size_t)b * N * N;
        for (int ii = 0; ii < N; ii++) {
            int i = forward ? ii : N - 1 - ii;
            ROW_SETUP(U, i)
            double *r = U + (size_t)i * N; const double *fr = F + (size_t)i * N;
            if (forward) {
                r[0] = om * r[0] + sc * (fr[0] - (s.cw * rm[0] + s.ce * rp[0] + s.cs * r[N - 1] + s.cn * r[1]));
                for (int j = 1; j < N - 1; j++)
                    r[j] = om * r[j] + sc * (fr[j] - (s.cw * rm[j] + s.ce * rp[j] + s.cs * r[j - 1] + s.cn * r[j + 1]));
                r[N - 1] = om * r[N - 1] + sc * (fr[N - 1] - (s.cw * rm[N - 1] + s.ce * rp[N - 1] + s.cs * r[N - 2] + s.cn * r[0]));
            } else {
                r[N - 1] = om * r[N - 1] + sc * (fr[N - 1] - (s.cw * rm[N - 1] + s.ce * rp[N - 1] + s.cs * r[N - 2] + s.cn * r[0]));
                for (int j = N - 2; j >= 1; j--)
                    r[j] = om * r[j] + sc * (fr[j] - (s.cw * rm[j] + s.ce * rp[j] + s.cs * r[j - 1] + s.cn * r[j + 1]));
                r[0] = om * r[0] + sc * (fr[0] - (s.cw * rm[0] + s.ce * rp[0] + s.cs * r[N - 1] + s.cn * r[1]));
            }
        }
    }
}

void ssor_sweep(double *u, const double *f, int B, int N, double ax, double ay, double b1, double b2, double omega) {
    sor_sweep(u, f, B, N, ax, ay, b1, b2, omega, 1);
    sor_sweep(u, f, B, N, ax, ay, b1, b2, omega, 0);
}

/* multigrid transfers (periodic): full weighting N -> N/2, bilinear prolongation N/2 -> N */
void restrict_fw(const double *r, double *rc, int B, int N) {
    int M = N / 2;
    for (int b = 0; b < B; b++) {
        const double *R = r + (size_t)b * N * N; double *C = rc + (size_t)b * M * M;
        for (int I = 0; I < M; I++) {
            int i = 2 * I, im = (i == 0) ? N - 1 : i - 1, ip = i + 1;
            const double *r0 = R + (size_t)i * N, *r1 = R + (size_t)im * N, *r2 = R + (size_t)ip * N;
            for (int J = 0; J < M; J++) {
                int j = 2 * J, jm = (j == 0) ? N - 1 : j - 1, jp = j + 1;
                C[(size_t)I * M + J] = 0.25 * r0[j] + 0.125 * (r1[j] + r2[j] + r0[jm] + r0[jp])
                                     + 0.0625 * (r1[jm] + r1[jp] + r2[jm] + r2[jp]);
            }
        }
    }
}

void prolong_bilinear(const double *c, double *u, int B, int M) {
    int N = 2 * M;
    for (int b = 0; b < B; b++) {
        const double *C = c + (size_t)b * M * M; double *U = u + (size_t)b * N * N;
        for (int I = 0; I < M; I++) {
            int Ip = (I == M - 1) ? 0 : I + 1;
            for (int J = 0; J < M; J++) {
                int Jp = (J == M - 1) ? 0 : J + 1;
                double c00 = C[(size_t)I * M + J], c10 = C[(size_t)Ip * M + J], c01 = C[(size_t)I * M + Jp], c11 = C[(size_t)Ip * M + Jp];
                size_t i = 2 * I, j = 2 * J;
                U[i * N + j] = c00;
                U[i * N + j + 1] = 0.5 * (c00 + c01);
                U[(i + 1) * N + j] = 0.5 * (c00 + c10);
                U[(i + 1) * N + j + 1] = 0.25 * (c00 + c01 + c10 + c11);
            }
        }
    }
}

/* ---------------------------------------------------------------------------------------
   Variable-coefficient 5-point stencil: A u = diag u + cw u_{i-1,j} + ce u_{i+1,j} + cs u_{i,j-1} + cn u_{i,j+1}
   with per-cell coefficient arrays (N x N, shared by all B batch elements).               */
#define VROW(i) \
    int im = (i == 0) ? N - 1 : i - 1, ip = (i == N - 1) ? 0 : i + 1; \
    const double *rm = U + (size_t)im * N, *rp = U + (size_t)ip * N; \
    const double *DG = diag + (size_t)i * N, *CW = cw + (size_t)i * N, *CE = ce + (size_t)i * N, *CS = cs + (size_t)i * N, *CN = cn + (size_t)i * N;
#define VAU(j, jm, jp) (DG[j] * r[j] + CW[j] * rm[j] + CE[j] * rp[j] + CS[j] * r[jm] + CN[j] * r[jp])

void apply_A_var(const double *u, double *out, int B, int N, const double *cw, const double *ce, const double *cs, const double *cn, const double *diag) {
    for (int b = 0; b < B; b++) {
        const double *U = u + (size_t)b * N * N; double *O = out + (size_t)b * N * N;
        for (int i = 0; i < N; i++) {
            VROW(i) const double *r = U + (size_t)i * N; double *o = O + (size_t)i * N;
            o[0] = VAU(0, N - 1, 1);
            for (int j = 1; j < N - 1; j++) o[j] = VAU(j, j - 1, j + 1);
            o[N - 1] = VAU(N - 1, N - 2, 0);
        }
    }
}

void residual_var(const double *u, const double *f, double *res, int B, int N, const double *cw, const double *ce, const double *cs, const double *cn, const double *diag) {
    for (int b = 0; b < B; b++) {
        const double *U = u + (size_t)b * N * N, *F = f + (size_t)b * N * N; double *R = res + (size_t)b * N * N;
        for (int i = 0; i < N; i++) {
            VROW(i) const double *r = U + (size_t)i * N, *fr = F + (size_t)i * N; double *o = R + (size_t)i * N;
            o[0] = fr[0] - VAU(0, N - 1, 1);
            for (int j = 1; j < N - 1; j++) o[j] = fr[j] - VAU(j, j - 1, j + 1);
            o[N - 1] = fr[N - 1] - VAU(N - 1, N - 2, 0);
        }
    }
}

double residual_norm2_var(const double *u, const double *f, double *res, int B, int N, const double *cw, const double *ce, const double *cs, const double *cn, const double *diag) {
    double acc = 0.0;
    for (int b = 0; b < B; b++) {
        const double *U = u + (size_t)b * N * N, *F = f + (size_t)b * N * N; double *R = res + (size_t)b * N * N;
        for (int i = 0; i < N; i++) {
            VROW(i) const double *r = U + (size_t)i * N, *fr = F + (size_t)i * N; double *o = R + (size_t)i * N;
            double v;
            v = fr[0] - VAU(0, N - 1, 1); o[0] = v; acc += v * v;
            for (int j = 1; j < N - 1; j++) { v = fr[j] - VAU(j, j - 1, j + 1); o[j] = v; acc += v * v; }
            v = fr[N - 1] - VAU(N - 1, N - 2, 0); o[N - 1] = v; acc += v * v;
        }
    }
    return acc;
}

void jacobi_var(const double *u, const double *f, double *out, int B, int N, const double *cw, const double *ce, const double *cs, const double *cn, const double *diag, double w) {
    for (int b = 0; b < B; b++) {
        const double *U = u + (size_t)b * N * N, *F = f + (size_t)b * N * N; double *O = out + (size_t)b * N * N;
        for (int i = 0; i < N; i++) {
            VROW(i) const double *r = U + (size_t)i * N, *fr = F + (size_t)i * N; double *o = O + (size_t)i * N;
            o[0] = r[0] + w * (fr[0] - VAU(0, N - 1, 1)) / DG[0];
            for (int j = 1; j < N - 1; j++) o[j] = r[j] + w * (fr[j] - VAU(j, j - 1, j + 1)) / DG[j];
            o[N - 1] = r[N - 1] + w * (fr[N - 1] - VAU(N - 1, N - 2, 0)) / DG[N - 1];
        }
    }
}

#define VOFF(j, jm, jp) (CW[j] * rm[j] + CE[j] * rp[j] + CS[j] * r[jm] + CN[j] * r[jp])
void sor_sweep_var(double *u, const double *f, int B, int N, const double *cw, const double *ce, const double *cs, const double *cn, const double *diag, double omega, int forward) {
    double om = 1.0 - omega;
    for (int b = 0; b < B; b++) {
        double *U = u + (size_t)b * N * N; const double *F = f + (size_t)b * N * N;
        for (int ii = 0; ii < N; ii++) {
            int i = forward ? ii : N - 1 - ii;
            VROW(i) double *r = U + (size_t)i * N; const double *fr = F + (size_t)i * N;
            if (forward) {
                r[0] = om * r[0] + omega * (fr[0] - VOFF(0, N - 1, 1)) / DG[0];
                for (int j = 1; j < N - 1; j++) r[j] = om * r[j] + omega * (fr[j] - VOFF(j, j - 1, j + 1)) / DG[j];
                r[N - 1] = om * r[N - 1] + omega * (fr[N - 1] - VOFF(N - 1, N - 2, 0)) / DG[N - 1];
            } else {
                r[N - 1] = om * r[N - 1] + omega * (fr[N - 1] - VOFF(N - 1, N - 2, 0)) / DG[N - 1];
                for (int j = N - 2; j >= 1; j--) r[j] = om * r[j] + omega * (fr[j] - VOFF(j, j - 1, j + 1)) / DG[j];
                r[0] = om * r[0] + omega * (fr[0] - VOFF(0, N - 1, 1)) / DG[0];
            }
        }
    }
}

void ssor_sweep_var(double *u, const double *f, int B, int N, const double *cw, const double *ce, const double *cs, const double *cn, const double *diag, double omega) {
    sor_sweep_var(u, f, B, N, cw, ce, cs, cn, diag, omega, 1);
    sor_sweep_var(u, f, B, N, cw, ce, cs, cn, diag, omega, 0);
}
