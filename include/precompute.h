#ifndef precompute_h
#define precompute_h
#include "exafmm_t.h"
#include "geometry.h"
#include "kernel.h"

namespace exafmm_t {
  void PrecompCheck2Equiv() {
    int level = 0;
    real_t c[3] = {0, 0, 0};
    // caculate M2M_U and M2M_V
    RealVec uc_coord = u_check_surf(c, level);
    RealVec ue_coord = u_equiv_surf(c, level);
    RealVec M_e2c(NSURF*NSURF);
    kernelMatrix(&ue_coord[0], NSURF, &uc_coord[0], NSURF, &M_e2c[0]);
    RealVec U(NSURF*NSURF), S(NSURF*NSURF), V(NSURF*NSURF);
    svd(NSURF, NSURF, &M_e2c[0], &S[0], &U[0], &V[0]);
    // inverse S
    real_t max_S = 0;
    for(size_t i=0; i<NSURF; i++) {
      max_S = fabs(S[i*NSURF+i])>max_S ? fabs(S[i*NSURF+i]) : max_S;
    }
    for(size_t i=0; i<NSURF; i++) {
      S[i*NSURF+i] = S[i*NSURF+i]>EPS*max_S*4 ? 1.0/S[i*NSURF+i] : 0.0;
    }
    // save matrix
    RealVec VT = transpose(V, NSURF, NSURF);
    M2M_V.resize(NSURF*NSURF);
    M2M_U = transpose(U, NSURF, NSURF);
    gemm(NSURF, NSURF, NSURF, &VT[0], &S[0], &M2M_V[0]);

    L2L_V.resize(NSURF*NSURF);
    L2L_U = V;
    gemm(NSURF, NSURF, NSURF, &U[0], &S[0], &L2L_V[0]);
  }

  void PrecompM2M() {
    int level = 0;
    real_t parent_coord[3] = {0, 0, 0};
    RealVec p_check_surf = u_check_surf(parent_coord, level);
    real_t s = powf(0.5, level+2);

    int numRelCoord = rel_coord[M2M_Type].size();
    mat_M2M.resize(numRelCoord);
    mat_L2L.resize(numRelCoord);
#pragma omp parallel for
    for(int i=0; i<numRelCoord; i++) {
      ivec3& coord = rel_coord[M2M_Type][i];
      real_t child_coord[3] = {(coord[0]+1)*s, (coord[1]+1)*s, (coord[2]+1)*s};
      RealVec c_equiv_surf = u_equiv_surf(child_coord, level+1);
      RealVec M_e2c(NSURF*NSURF);
      kernelMatrix(&c_equiv_surf[0], NSURF, &p_check_surf[0], NSURF, &M_e2c[0]);
      // M2M: child's upward_equiv to parent's check
      RealVec buffer(NSURF*NSURF);
      mat_M2M[i].resize(NSURF*NSURF);
      gemm(NSURF, NSURF, NSURF, &M_e2c[0], &M2M_V[0], &buffer[0]);
      gemm(NSURF, NSURF, NSURF, &buffer[0], &M2M_U[0], &(mat_M2M[i][0]));
      // L2L: parent's dnward_equiv to child's check, reuse surface coordinates
      M_e2c = transpose(M_e2c, NSURF, NSURF);
      mat_L2L[i].resize(NSURF*NSURF);
      gemm(NSURF, NSURF, NSURF, &L2L_U[0], &M_e2c[0], &buffer[0]);
      gemm(NSURF, NSURF, NSURF, &L2L_V[0], &buffer[0], &(mat_L2L[i][0]));
    }
  }

  void PrecompM2LHelper() {
    // create fftw plan
    RealVec fftw_in(N3);
    RealVec fftw_out(2*N3_);
    int dim[3] = {2*MULTIPOLE_ORDER, 2*MULTIPOLE_ORDER, 2*MULTIPOLE_ORDER};
    fft_plan plan = fft_plan_many_dft_r2c(3, dim, 1, &fftw_in[0], NULL, 1, N3,
                    (fft_complex*)(&fftw_out[0]), NULL, 1, N3_, FFTW_ESTIMATE);
    // evaluate DFTs of potentials at convolution grids
    int numRelCoord = rel_coord[M2L_Helper_Type].size();
    mat_M2L_Helper.resize(numRelCoord);
    #pragma omp parallel for
    for(int i=0; i<numRelCoord; i++) {
      real_t coord[3];
      for(int d=0; d<3; d++) {
        coord[d] = rel_coord[M2L_Helper_Type][i][d];
      }
      RealVec conv_coord = conv_grid(coord, 0);
      RealVec r_trg(3, 0.0);
      RealVec conv_poten(N3);
      kernelMatrix(&conv_coord[0], N3, &r_trg[0], 1, &conv_poten[0]);
      mat_M2L_Helper[i].resize(2*N3_);
      fft_execute_dft_r2c(plan, &conv_poten[0], (fft_complex*)(&mat_M2L_Helper[i][0]));
    }
    fft_destroy_plan(plan);
  }

  void PrecompM2L() {
    int numParentRelCoord = rel_coord[M2L_Type].size();
    int numChildRelCoord = rel_coord[M2L_Helper_Type].size();
    mat_M2L.resize(numParentRelCoord);
    RealVec zero_vec(N3_*2, 0);
    #pragma omp parallel for schedule(dynamic)
    for(int i=0; i<numParentRelCoord; i++) {
      ivec3& parentRelCoord = rel_coord[M2L_Type][i];
      std::vector<real_t*> M_ptr(NCHILD*NCHILD, &zero_vec[0]);
      for(int j1=0; j1<NCHILD; j1++) {
        for(int j2=0; j2<NCHILD; j2++) {
          int childRelCoord[3]= { parentRelCoord[0]*2 - (j1/1)%2 + (j2/1)%2,
                                  parentRelCoord[1]*2 - (j1/2)%2 + (j2/2)%2,
                                  parentRelCoord[2]*2 - (j1/4)%2 + (j2/4)%2 };
          for(int k=0; k<numChildRelCoord; k++) {
            ivec3& childRefCoord = rel_coord[M2L_Helper_Type][k];
            if (childRelCoord[0] == childRefCoord[0] &&
                childRelCoord[1] == childRefCoord[1] &&
                childRelCoord[2] == childRefCoord[2]) {
              M_ptr[j2*NCHILD+j1] = &mat_M2L_Helper[k][0];
              break;
            }
          }
        }
      }
      mat_M2L[i].resize(N3_*2*NCHILD*NCHILD);  // N3 by (2*NCHILD*NCHILD) matrix
      for(int k=0; k<N3_; k++) {                      // loop over frequencies
        for(size_t j=0; j<NCHILD*NCHILD; j++) {       // loop over child's relative positions
          int index = k*(2*NCHILD*NCHILD) + 2*j;
          mat_M2L[i][index+0] = M_ptr[j][k*2+0]/N3;   // real
          mat_M2L[i][index+1] = M_ptr[j][k*2+1]/N3;   // imag
        }
      }
    }
  }

  void Precompute() {
    PrecompCheck2Equiv();
    PrecompM2M();
    PrecompM2LHelper();
    PrecompM2L();
  }
}//end namespace
#endif