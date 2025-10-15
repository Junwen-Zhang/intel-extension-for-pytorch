
#include <ATen/record_function.h>
#include <aten/TPPGEMM.h>
#include <torch/all.h>
#include <iostream>
#include <vector>
#include "tpp/ext_tpp.h"
#include "tpp/utils.h"
#ifndef NO_PARLOOPER
#include "tpp/threaded_loops.h"
#endif
#include <cstdint>
#include "../../utils/isa_utils.h"
#include "tpp/tensor_helper.h"
#include "tpp/xsmm_functors.h"

namespace torch_ipex {
namespace tpp {

static int use_at_vnni = false; // env2int("USE_AT_VNNI");
static int FT_OPT_SIZE = env2int("FT_OPT_SIZE", 256);
static int NCB_BLOCK_SIZE = env2int("NCB_BLOCK_SIZE", 64);
static const char* GEMM_LOOP_SCHEME =
    getenv("GEMM_LOOP_SCHEME") ? getenv("GEMM_LOOP_SCHEME") : "aCB";

REGISTER_LOCAL_SCOPE(
    tpp_linear_krnl,
    "tpp_linear_krnl"); //  linear W/ and W/O bias
REGISTER_LOCAL_SCOPE(
    tpp_linear_add_add_krnl,
    "tpp_linear_add_add_krnl"); // linear bias + add + add
REGISTER_LOCAL_SCOPE(
    tpp_linear_gelu_krnl,
    "tpp_linear_gelu_krnl"); // linear bias + gelu
REGISTER_LOCAL_SCOPE(
    tpp_linear_gelu_tanh_krnl,
    "tpp_linear_gelu_tanh_krnl"); // linear bias + gelu_tanh
REGISTER_LOCAL_SCOPE(
    tpp_linear_mul_krnl,
    "tpp_linear_mul_krnl"); // linear bias + mul
REGISTER_LOCAL_SCOPE(
    tpp_linear_add_krnl,
    "tpp_linear_add_krnl"); // linear bias + add
REGISTER_LOCAL_SCOPE(
    tpp_linear_silu_krnl,
    "tpp_linear_silu_krnl"); // linear bias + silu
REGISTER_LOCAL_SCOPE(
    tpp_fused_gate_up_proj_krnl,
    "tpp_fused_gate_up_proj_krnl"); // fused gate_proj and up_proj
REGISTER_LOCAL_SCOPE(
    tpp_linear_relu_krnl,
    "tpp_linear_relu_krnl"); // linear bias + relu

REGISTER_LOCAL_SCOPE(fftkn, "fftkn");

template <typename T>
inline at::Tensor wt_tensor_for_first_token(at::Tensor& t) {
  RECORD_SCOPE(fftkn, {t});
  auto dim = t.dim();
  if (dim < 5)
    return t;
  auto sizes = t.sizes();
  constexpr long RBS = 4;
  auto K1 = sizes[0];
  if (K1 % RBS != 0)
    return t;
  auto C1 = sizes[1];
  auto C2 = sizes[2];
  auto K2 = sizes[3];
  auto C3 = sizes[4];
  if (K2 > 32)
    return t;
  auto t_new = t.new_empty({K1 / RBS, C1, C2, RBS * K2, C3});
  auto in = GetVLAPtr<T>(t, {RBS, C1, C2, K2 * C3});
  auto out = GetVLAPtr<T>(t_new, {C1, C2, RBS, K2 * C3});

  auto cpy_tpp =
      SCOPEIT(CpyTPP<T>(C2, K2 * C3, K2 * C3, RBS * K2 * C3), EW_COPY);

#pragma omp parallel for collapse(2)
  for (int i = 0; i < K1 / RBS; i++) {
    for (int j = 0; j < C1; j++) {
      for (int k = 0; k < RBS; k++) {
        cpy_tpp(in[i][k][j][0], out[i][j][0][k]);
      }
    }
  }

  return t_new;
}

template <typename T, typename Tout = T>
inline void tpp_linear_bias(
    const at::Tensor& t_in,
    const at::Tensor& t_wt,
    const at::Tensor& t_bias,
    at::Tensor& t_out,
    int b_vnni) {
  auto t_wt_ = t_wt;
  auto in_sizes = t_in.sizes();
  auto wt_sizes = t_wt_.sizes();
  auto BS = in_sizes[0] * in_sizes[1];
  bool large_cache_opt = false;
  if (BS > FT_OPT_SIZE) { // first token compute
    if (wt_sizes[3] != 100) {
      t_wt_ = wt_tensor_for_first_token<T>(t_wt_);
      wt_sizes = t_wt_.sizes();
    }
    large_cache_opt = true;
  }

  auto C = in_sizes[2];

  auto Nc = wt_sizes[1];
  auto Hc = C / Nc;
  auto Nk = wt_sizes[0];
  auto Hk = wt_sizes[3];
  auto K = Nk * Hk;

  auto t_wt_V = torch_ipex::tpp::wt_tensor_for_fwd(Nk, Hk, Nc, Hc, t_wt_);

  auto in = GetVLAPtr<T>(t_in, {Nc, Hc});

  auto wt_V = GetVLAPtr<T>(t_wt_V, {Nc, Hc * Hk});

  auto bias = GetVLAPtr<Tout>(t_bias, {Hk});

  auto out = GetVLAPtr<Tout>(t_out, {Nk, Hk});

  auto Ncb = Nc;
  auto BSb = 64L;
  auto rem = BS % 64;
  if (large_cache_opt)
    Ncb = NCB_BLOCK_SIZE;

  bool with_bias = (t_bias.numel() > 0);
  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<Tout>(BSb, Hk, K), BIAS);
  auto copy_bias_tpp_rem = SCOPEIT(CpyBiasTPP<Tout>(rem, Hk, K), BIAS);
  auto zero_tpp = SCOPEIT(SetZeroTPP<Tout>(BSb, Hk, K), EW_ZERO);
  auto zero_tpp_rem = SCOPEIT(SetZeroTPP<Tout>(rem, Hk, K), EW_ZERO);
  auto brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T, Tout>(
      BSb, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb, b_vnni)));
  auto brgemm_tpp_rem = SCOPEITGEMM((BrgemmTPP<T, Tout>(
      rem, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb, b_vnni)));

  {
    RECORD_SCOPE(tpp_linear_krnl, {t_in, t_wt_V});
    auto loop_scheme = large_cache_opt ? GEMM_LOOP_SCHEME : "aCb";
    auto ogemm_loop = torch_ipex::tpp::ThreadedLoop<3>(
        {{0, Nc, Ncb, false}, {0L, BS, BSb}, {Nk}}, loop_scheme);
    ogemm_loop(
        [&](int* ind) {
          int nc = ind[0], s1 = ind[1], nk = ind[2];
          auto count = nc + Ncb < Nc ? Ncb : Nc - nc;
          bool is_rem = (s1 + BSb > BS);
          if (!is_rem) {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp(bias[nk], out[s1][nk]);
              } else {
                zero_tpp(out[s1][nk]);
              }
            }
            brgemm_tpp(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, true);
          } else {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp_rem(bias[nk], out[s1][nk]);
              } else {
                zero_tpp_rem(out[s1][nk]);
              }
            }
            brgemm_tpp_rem(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, false);
            brgemm_tpp.config();
          }
        },
        [&]() { brgemm_tpp.config(); },
        [&]() { brgemm_tpp.release(); });
  }
}

template <typename T, typename Tout = T>
inline void tpp_linear_no_bias(
    const at::Tensor& t_in,
    const at::Tensor& t_wt,
    at::Tensor& t_out,
    int b_vnni) {
  auto t_wt_ = t_wt;
  auto in_sizes = t_in.sizes();
  auto BS = in_sizes[0] * in_sizes[1];
  auto wt_sizes = t_wt_.sizes();
  bool large_cache_opt = false;
  if (BS > FT_OPT_SIZE) { // first token compute
    if (wt_sizes[3] != 100) {
      t_wt_ = wt_tensor_for_first_token<T>(t_wt_);
      wt_sizes = t_wt_.sizes();
    }
    large_cache_opt = true;
  }

  auto C = in_sizes[2];

  auto Nc = wt_sizes[1];
  auto Hc = C / Nc;
  auto Nk = wt_sizes[0];
  auto Hk = wt_sizes[3];
  auto K = Nk * Hk;
  auto t_wt_V = torch_ipex::tpp::wt_tensor_for_fwd(Nk, Hk, Nc, Hc, t_wt_);

  auto in = GetVLAPtr<T>(t_in, {Nc, Hc});
  auto wt_V = GetVLAPtr<T>(t_wt_V, {Nc, Hc * Hk});
  auto out = GetVLAPtr<Tout>(t_out, {Nk, Hk});

  auto Ncb = Nc;
  auto BSb = 64L;
  auto rem = BS % BSb;
  if (large_cache_opt)
    Ncb = NCB_BLOCK_SIZE;

  auto zero_tpp = SCOPEIT(SetZeroTPP<Tout>(BSb, Hk, K), EW_ZERO);
  auto zero_tpp_rem = SCOPEIT(SetZeroTPP<Tout>(rem, Hk, K), EW_ZERO);
  auto brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T, Tout>(
      BSb, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb, b_vnni)));
  auto brgemm_tpp_rem = SCOPEITGEMM((BrgemmTPP<T, Tout>(
      rem, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb, b_vnni)));

  {
    RECORD_SCOPE(tpp_linear_krnl, {t_in, t_wt_V});
    auto loop_scheme = large_cache_opt ? GEMM_LOOP_SCHEME : "aCb";
    auto gemm_loop = torch_ipex::tpp::ThreadedLoop<3>(
        {{0, Nc, Ncb, false}, {0, BS, BSb}, {Nk}}, loop_scheme);
    gemm_loop(
        [&](int* ind) {
          int nc = ind[0], s1 = ind[1], nk = ind[2];
          auto count = nc + Ncb < Nc ? Ncb : Nc - nc;
          bool is_rem = (s1 + BSb > BS);
          if (!is_rem) {
            if (nc == 0) {
              zero_tpp(out[s1][nk]);
            }
            brgemm_tpp(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, true);
          } else {
            if (nc == 0) {
              zero_tpp_rem(out[s1][nk]);
            }
            brgemm_tpp_rem(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, false);
            brgemm_tpp.config();
          }
        },
        [&]() { brgemm_tpp.config(); },
        [&]() { brgemm_tpp.release(); });
  }
}

template <typename T, typename Tout = T>
inline void tpp_linear_mul(
    const at::Tensor t_in,
    const at::Tensor t_in1,
    const at::Tensor t_wt,
    const at::Tensor t_bias,
    at::Tensor t_out,
    int b_vnni) {
  auto t_wt_ = t_wt;
  auto in_sizes = t_in.sizes();
  auto BS = in_sizes[0] * in_sizes[1];
  bool large_cache_opt = false;
  if (BS > FT_OPT_SIZE) { // first token compute
    t_wt_ = wt_tensor_for_first_token<T>(t_wt_);
    large_cache_opt = true;
  }

  auto wt_sizes = t_wt_.sizes();
  auto C = in_sizes[2];

  auto Nc = wt_sizes[1];
  auto Hc = C / Nc;
  auto Nk = wt_sizes[0];
  auto Hk = wt_sizes[3];
  auto K = Nk * Hk;

  auto t_wt_V = torch_ipex::tpp::wt_tensor_for_fwd(Nk, Hk, Nc, Hc, t_wt_);

  auto in = GetVLAPtr<T>(t_in, {Nc, Hc});
  auto in1 = GetVLAPtr<Tout>(t_in1, {Nk, Hk});
  auto wt_V = GetVLAPtr<T>(t_wt_V, {Nc, Hc * Hk});
  auto bias = GetVLAPtr<Tout>(t_bias, {Hk});
  auto out = GetVLAPtr<Tout>(t_out, {Nk, Hk});

  auto Ncb = Nc;
  auto BSb = 64L;
  auto rem = BS % 64;
  if (large_cache_opt)
    Ncb = NCB_BLOCK_SIZE;

  bool with_bias = (t_bias.numel() > 0);
  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<Tout>(BSb, Hk, K), BIAS);
  auto copy_bias_tpp_rem = SCOPEIT(CpyBiasTPP<Tout>(rem, Hk, K), BIAS);
  auto zero_tpp = SCOPEIT(SetZeroTPP<Tout>(BSb, Hk, K), EW_ZERO);
  auto zero_tpp_rem = SCOPEIT(SetZeroTPP<Tout>(rem, Hk, K), EW_ZERO);
  auto brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T, Tout>(
      BSb, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb, b_vnni)));
  auto brgemm_tpp_rem = SCOPEITGEMM((BrgemmTPP<T, Tout>(
      rem, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb, b_vnni)));
  auto mul_tpp = SCOPEIT((MulTPP<Tout, Tout>(BSb, Hk, K, K)), EW_MUL);
  auto mul_tpp_rem = SCOPEIT((MulTPP<Tout, Tout>(rem, Hk, K, K)), EW_MUL);

  {
    RECORD_SCOPE(tpp_linear_mul_krnl, {t_in, t_wt_V});

    auto loop_scheme = large_cache_opt ? GEMM_LOOP_SCHEME : "aCb";
    auto ogemm_loop = torch_ipex::tpp::ThreadedLoop<3>(
        {{0, Nc, Ncb, false}, {0L, BS, BSb}, {Nk}}, loop_scheme);
    ogemm_loop(
        [&](int* ind) {
          int nc = ind[0], s1 = ind[1], nk = ind[2];
          auto count = nc + Ncb < Nc ? Ncb : Nc - nc;
          bool is_rem = (s1 + BSb > BS);
          if (!is_rem) {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp(bias[nk], out[s1][nk]);
              } else {
                zero_tpp(out[s1][nk]);
              }
            }
            brgemm_tpp(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, true);
            if (!(nc + Ncb < Nc)) { // last nc iter
              mul_tpp(in1[s1][nk], out[s1][nk], out[s1][nk]);
            }
          } else {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp_rem(bias[nk], out[s1][nk]);
              } else {
                zero_tpp_rem(out[s1][nk]);
              }
            }
            brgemm_tpp_rem(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, false);
            brgemm_tpp.config();
            if (!(nc + Ncb < Nc)) { // last nc iter
              mul_tpp_rem(in1[s1][nk], out[s1][nk], out[s1][nk]);
            }
          }
        },
        [&]() { brgemm_tpp.config(); },
        [&]() { brgemm_tpp.release(); });
  }
}

template <typename T>
inline void tpp_linear_add_add(
    const at::Tensor& t_in,
    const at::Tensor& t_in1,
    const at::Tensor& t_in2,
    const at::Tensor& t_wt,
    const at::Tensor& t_bias,
    at::Tensor& t_out,
    double scale,
    int b_vnni) {
  auto t_wt_ = t_wt;
  auto in_sizes = t_in.sizes();
  auto BS = in_sizes[0] * in_sizes[1];
  bool large_cache_opt = false;
  if (BS > FT_OPT_SIZE) { // first token compute
    t_wt_ = wt_tensor_for_first_token<T>(t_wt_);
    large_cache_opt = true;
  }

  auto wt_sizes = t_wt_.sizes();
  auto C = in_sizes[2];

  auto Nc = wt_sizes[1];
  auto Hc = C / Nc;
  auto Nk = wt_sizes[0];
  auto Hk = wt_sizes[3];
  auto K = Nk * Hk;

  auto t_wt_V = torch_ipex::tpp::wt_tensor_for_fwd(Nk, Hk, Nc, Hc, t_wt_);

  auto in = GetVLAPtr<T>(t_in, {Nc, Hc});
  auto in1 = GetVLAPtr<T>(t_in1, {Nk, Hk});
  auto in2 = GetVLAPtr<T>(t_in2, {Nk, Hk});
  auto wt_V = GetVLAPtr<T>(t_wt_V, {Nc, Hc * Hk});
  auto bias = GetVLAPtr<T>(t_bias, {Hk});
  auto out = GetVLAPtr<T>(t_out, {Nk, Hk});

  auto Ncb = Nc;
  auto BSb = 64L;
  auto rem = BS % 64;
  if (large_cache_opt)
    Ncb = NCB_BLOCK_SIZE;
  bool with_bias = (t_bias.numel() > 0);
  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<T>(BSb, Hk, K), BIAS);
  auto copy_bias_tpp_rem = SCOPEIT(CpyBiasTPP<T>(rem, Hk, K), BIAS);
  auto zero_tpp = SCOPEIT(SetZeroTPP<T>(BSb, Hk, K), EW_ZERO);
  auto zero_tpp_rem = SCOPEIT(SetZeroTPP<T>(rem, Hk, K), EW_ZERO);
  auto brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T, T>(
      BSb, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb, b_vnni)));
  auto brgemm_tpp_rem = SCOPEITGEMM((BrgemmTPP<T, T>(
      rem, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb, b_vnni)));
  auto add_tpp = SCOPEIT((AddTPP<T, T>(BSb, Hk, K, K)), EW_ADD);
  auto add_tpp_rem = SCOPEIT((AddTPP<T, T>(rem, Hk, K, K)), EW_ADD);
  auto sadd_tpp = SCOPEIT((ScaleAddTPP<T, T>(BSb, Hk, K, K)), EW_ADD);
  auto sadd_tpp_rem = SCOPEIT((ScaleAddTPP<T, T>(rem, Hk, K, K)), EW_ADD);

  {
    RECORD_SCOPE(tpp_linear_add_add_krnl, {t_in, t_wt_V});

    auto loop_scheme = large_cache_opt ? GEMM_LOOP_SCHEME : "aCb";
    auto ogemm_loop = torch_ipex::tpp::ThreadedLoop<3>(
        {{0, Nc, Ncb, false}, {0L, BS, BSb}, {Nk}}, loop_scheme);
    ogemm_loop(
        [&](int* ind) {
          int nc = ind[0], s1 = ind[1], nk = ind[2];
          auto count = nc + Ncb < Nc ? Ncb : Nc - nc;
          bool is_rem = (s1 + BSb > BS);
          if (!is_rem) {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp(bias[nk], out[s1][nk]);
              } else {
                zero_tpp(out[s1][nk]);
              }
            }
            brgemm_tpp(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, true);
            if (!(nc + Ncb < Nc)) { // last nc iter
              add_tpp(out[s1][nk], in1[s1][nk], out[s1][nk]);
              sadd_tpp(in2[s1][nk], out[s1][nk], scale);
            }
          } else {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp_rem(bias[nk], out[s1][nk]);
              } else {
                zero_tpp_rem(out[s1][nk]);
              }
            }
            brgemm_tpp_rem(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, false);
            brgemm_tpp.config();
            if (!(nc + Ncb < Nc)) { // last nc iter
              add_tpp_rem(out[s1][nk], in1[s1][nk], out[s1][nk]);
              sadd_tpp_rem(in2[s1][nk], out[s1][nk], scale);
            }
          }
        },
        [&]() { brgemm_tpp.config(); },
        [&]() { brgemm_tpp.release(); });
  }
}

template <typename T, typename Tout = T>
inline void tpp_linear_gelu(
    const at::Tensor& t_in,
    const at::Tensor& t_wt,
    const at::Tensor& t_bias,
    at::Tensor& t_out,
    int b_vnni) {
  auto t_wt_ = t_wt;
  auto in_sizes = t_in.sizes();
  auto BS = in_sizes[0] * in_sizes[1];
  bool large_cache_opt = false;
  if (BS > FT_OPT_SIZE) { // first token compute
    t_wt_ = wt_tensor_for_first_token<T>(t_wt_);
    large_cache_opt = true;
  }

  auto wt_sizes = t_wt_.sizes();
  auto C = in_sizes[2];

  auto Nc = wt_sizes[1];
  auto Hc = C / Nc;
  auto Nk = wt_sizes[0];
  auto Hk = wt_sizes[3];
  auto K = Nk * Hk;

  auto t_wt_V = torch_ipex::tpp::wt_tensor_for_fwd(Nk, Hk, Nc, Hc, t_wt_);

  auto in = GetVLAPtr<T>(t_in, {Nc, Hc});
  auto wt_V = GetVLAPtr<T>(t_wt_V, {Nc, Hc * Hk});
  auto bias = GetVLAPtr<Tout>(t_bias, {Hk});
  auto out = GetVLAPtr<Tout>(t_out, {Nk, Hk});

  auto Ncb = Nc;
  auto BSb = 64L;
  auto rem = BS % 64;
  if (large_cache_opt)
    Ncb = NCB_BLOCK_SIZE;
  bool with_bias = (t_bias.numel() > 0);
  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<Tout>(BSb, Hk, K), BIAS);
  auto copy_bias_tpp_rem = SCOPEIT(CpyBiasTPP<Tout>(rem, Hk, K), BIAS);
  auto zero_tpp = SCOPEIT(SetZeroTPP<Tout>(BSb, Hk, K), EW_ZERO);
  auto zero_tpp_rem = SCOPEIT(SetZeroTPP<Tout>(rem, Hk, K), EW_ZERO);
  auto brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T, Tout>(
      BSb, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb, b_vnni)));
  auto brgemm_tpp_rem = SCOPEITGEMM((BrgemmTPP<T, Tout>(
      rem, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb, b_vnni)));
  auto gelu_fwd_tpp = SCOPEIT(GeluFwdTPP<Tout>(BSb, Hk, K, K), ACT);
  auto gelu_fwd_tpp_rem = SCOPEIT(GeluFwdTPP<Tout>(rem, Hk, K, K), ACT);

  {
    RECORD_SCOPE(tpp_linear_gelu_krnl, {t_in, t_wt_V});

    auto loop_scheme = large_cache_opt ? GEMM_LOOP_SCHEME : "aCb";
    auto igemm_loop = torch_ipex::tpp::ThreadedLoop<3>(
        {{0, Nc, Ncb, false}, {0, BS, BSb}, {Nk}}, loop_scheme);
    igemm_loop(
        [&](int* ind) {
          int nc = ind[0], s1 = ind[1], nk = ind[2];
          auto count = nc + Ncb < Nc ? Ncb : Nc - nc;
          bool is_rem = (s1 + BSb > BS);
          if (!is_rem) {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp(bias[nk], out[s1][nk]);
              } else {
                zero_tpp(out[s1][nk]);
              }
            }
            brgemm_tpp(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, true);
            if (!(nc + Ncb < Nc)) { // last nc iter
              gelu_fwd_tpp(out[s1][nk], out[s1][nk]);
            }
          } else {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp_rem(bias[nk], out[s1][nk]);
              } else {
                zero_tpp_rem(out[s1][nk]);
              }
            }
            brgemm_tpp_rem(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, false);
            brgemm_tpp.config();
            if (!(nc + Ncb < Nc)) { // last nc iter
              gelu_fwd_tpp_rem(out[s1][nk], out[s1][nk]);
            }
          }
        },
        [&]() { brgemm_tpp.config(); },
        [&]() { brgemm_tpp.release(); });
  }
}

template <typename T, typename Tout = T>
inline void tpp_linear_gelu_tanh(
    const at::Tensor& t_in,
    const at::Tensor& t_wt,
    const at::Tensor& t_bias,
    at::Tensor& t_out,
    int b_vnni) {
  auto t_wt_ = t_wt;
  auto in_sizes = t_in.sizes();
  auto BS = in_sizes[0] * in_sizes[1];
  bool large_cache_opt = false;
  if (BS > FT_OPT_SIZE) { // first token compute
    t_wt_ = wt_tensor_for_first_token<T>(t_wt_);
    large_cache_opt = true;
  }

  auto wt_sizes = t_wt_.sizes();
  auto C = in_sizes[2];

  auto Nc = wt_sizes[1];
  auto Hc = C / Nc;
  auto Nk = wt_sizes[0];
  auto Hk = wt_sizes[3];
  auto K = Nk * Hk;

  auto t_wt_V = torch_ipex::tpp::wt_tensor_for_fwd(Nk, Hk, Nc, Hc, t_wt_);

  auto in = GetVLAPtr<T>(t_in, {Nc, Hc});
  auto wt_V = GetVLAPtr<T>(t_wt_V, {Nc, Hc * Hk});
  auto bias = GetVLAPtr<Tout>(t_bias, {Hk});
  auto out = GetVLAPtr<Tout>(t_out, {Nk, Hk});

  auto Ncb = Nc;
  auto BSb = 64L;
  auto rem = BS % 64;
  if (large_cache_opt)
    Ncb = NCB_BLOCK_SIZE;
  bool with_bias = (t_bias.numel() > 0);
  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<Tout>(BSb, Hk, K), BIAS);
  auto copy_bias_tpp_rem = SCOPEIT(CpyBiasTPP<Tout>(rem, Hk, K), BIAS);
  auto zero_tpp = SCOPEIT(SetZeroTPP<Tout>(BSb, Hk, K), EW_ZERO);
  auto zero_tpp_rem = SCOPEIT(SetZeroTPP<Tout>(rem, Hk, K), EW_ZERO);
  auto brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T, Tout>(
      BSb, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb, b_vnni)));
  auto brgemm_tpp_rem = SCOPEITGEMM((BrgemmTPP<T, Tout>(
      rem, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb, b_vnni)));
  auto gelu_fwd_tpp = SCOPEIT(GeluTanhFwdTPP<Tout>(BSb, Hk, K, K), ACT);
  auto gelu_fwd_tpp_rem = SCOPEIT(GeluTanhFwdTPP<Tout>(rem, Hk, K, K), ACT);

  {
    RECORD_SCOPE(tpp_linear_gelu_krnl, {t_in, t_wt_V});

    auto loop_scheme = large_cache_opt ? GEMM_LOOP_SCHEME : "aCb";
    auto igemm_loop = torch_ipex::tpp::ThreadedLoop<3>(
        {{0, Nc, Ncb, false}, {0, BS, BSb}, {Nk}}, loop_scheme);
    igemm_loop(
        [&](int* ind) {
          int nc = ind[0], s1 = ind[1], nk = ind[2];
          auto count = nc + Ncb < Nc ? Ncb : Nc - nc;
          bool is_rem = (s1 + BSb > BS);
          if (!is_rem) {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp(bias[nk], out[s1][nk]);
              } else {
                zero_tpp(out[s1][nk]);
              }
            }
            brgemm_tpp(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, true);
            if (!(nc + Ncb < Nc)) { // last nc iter
              gelu_fwd_tpp(out[s1][nk], out[s1][nk]);
            }
          } else {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp_rem(bias[nk], out[s1][nk]);
              } else {
                zero_tpp_rem(out[s1][nk]);
              }
            }
            brgemm_tpp_rem(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, false);
            brgemm_tpp.config();
            if (!(nc + Ncb < Nc)) { // last nc iter
              gelu_fwd_tpp_rem(out[s1][nk], out[s1][nk]);
            }
          }
        },
        [&]() { brgemm_tpp.config(); },
        [&]() { brgemm_tpp.release(); });
  }
}

// Fused kernel for the gate_proj and the up_proj related computation in the MLP
// of LLaMA. The ref computation of the kernel is:
//    act_fn(gate_proj(x)) * up_proj(x) where act_fn is silu, gate_proj and
//    up_proj are two nn.Linear with the same weight shapes and bias = False.
// t_in is the input activation
// t_wt_gate is the prepacked weight of the gate_proj
// t_wt_up is the prepacked weight of the up_proj
// t_bias_gate is the bias of the gate_proj
// t_bias_up is the bias of the up_proj
// t_out is the output result of the kernel
template <typename T>
inline void tpp_fused_gate_up_proj(
    const at::Tensor& t_in,
    const at::Tensor& t_wt_gate,
    const at::Tensor& t_bias_gate,
    const at::Tensor& t_wt_up,
    const at::Tensor& t_bias_up,
    at::Tensor& t_out,
    int b_vnni) {
  auto t_wt_gate_ = t_wt_gate;
  auto t_wt_up_ = t_wt_up;
  auto in_sizes = t_in.sizes();
  auto BS = in_sizes[0] * in_sizes[1];
  bool large_cache_opt = false;
  if (BS > FT_OPT_SIZE) { // first token compute
    t_wt_gate_ = wt_tensor_for_first_token<T>(t_wt_gate_);
    t_wt_up_ = wt_tensor_for_first_token<T>(t_wt_up_);
    large_cache_opt = true;
  }

  auto wt_sizes = t_wt_gate_.sizes();
  auto C = in_sizes[2];

  auto Nc = wt_sizes[1];
  auto Hc = C / Nc;
  auto Nk = wt_sizes[0];
  auto Hk = wt_sizes[3];
  auto K = Nk * Hk;

  auto t_wt_gate_V =
      torch_ipex::tpp::wt_tensor_for_fwd(Nk, Hk, Nc, Hc, t_wt_gate_);
  auto t_wt_up_V = torch_ipex::tpp::wt_tensor_for_fwd(Nk, Hk, Nc, Hc, t_wt_up_);

  // This is used to store the intermediate result of the up_proj layer
  auto t_out_tmp = at::empty_like(t_out);

  auto in = GetVLAPtr<T>(t_in, {Nc, Hc});
  auto wt_gate_V = GetVLAPtr<T>(t_wt_gate_V, {Nc, Hc * Hk});
  auto wt_up_V = GetVLAPtr<T>(t_wt_up_V, {Nc, Hc * Hk});
  auto bias_gate = GetVLAPtr<T>(t_bias_gate, {Hk});
  auto bias_up = GetVLAPtr<T>(t_bias_up, {Hk});
  auto out = GetVLAPtr<T>(t_out, {Nk, Hk});
  auto out_tmp = GetVLAPtr<T>(t_out_tmp, {Nk, Hk});

  auto Ncb = Nc;
  auto BSb = 64L;
  auto rem = BS % 64;
  if (large_cache_opt)
    Ncb = NCB_BLOCK_SIZE;

  bool with_bias_gate = (t_bias_gate.numel() > 0);
  bool with_bias_up = (t_bias_up.numel() > 0);
  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<T>(BSb, Hk, K), BIAS);
  auto copy_bias_tpp_rem = SCOPEIT(CpyBiasTPP<T>(rem, Hk, K), BIAS);
  auto zero_tpp = SCOPEIT(SetZeroTPP<T>(BSb, Hk, K), EW_ZERO);
  auto zero_tpp_rem = SCOPEIT(SetZeroTPP<T>(rem, Hk, K), EW_ZERO);
  auto brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T, T>(
      BSb, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb, b_vnni)));
  auto brgemm_tpp_rem = SCOPEITGEMM((BrgemmTPP<T, T>(
      rem, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb, b_vnni)));
  auto silu_fwd_tpp = SCOPEIT(SiLUFwdTPP<T>(BSb, Hk, K, K), ACT);
  auto silu_fwd_tpp_rem = SCOPEIT(SiLUFwdTPP<T>(rem, Hk, K, K), ACT);
  auto mul_tpp = SCOPEIT((MulTPP<T, T>(BSb, Hk, K, K)), EW_MUL);
  auto mul_tpp_rem = SCOPEIT((MulTPP<T, T>(rem, Hk, K, K)), EW_MUL);

  {
    RECORD_SCOPE(tpp_fused_gate_up_proj_krnl, {t_in, t_wt_gate_V});

    auto loop_scheme = large_cache_opt ? GEMM_LOOP_SCHEME : "aCb";
    auto igemm_loop = torch_ipex::tpp::ThreadedLoop<3>(
        {{0, Nc, Ncb, false}, {0, BS, BSb}, {Nk}}, loop_scheme);
    igemm_loop(
        [&](int* ind) {
          int nc = ind[0], s1 = ind[1], nk = ind[2];
          auto count = nc + Ncb < Nc ? Ncb : Nc - nc;
          bool is_rem = (s1 + BSb > BS);
          if (!is_rem) {
            if (nc == 0) {
              if (with_bias_gate) {
                copy_bias_tpp(bias_gate[nk], out[s1][nk]);
              } else {
                zero_tpp(out[s1][nk]);
              }

              if (with_bias_up) {
                copy_bias_tpp(bias_up[nk], out_tmp[s1][nk]);
              } else {
                zero_tpp(out_tmp[s1][nk]);
              }
            }
            brgemm_tpp(in[s1][nc], wt_gate_V[nk][nc], out[s1][nk], count, true);
            brgemm_tpp(
                in[s1][nc], wt_up_V[nk][nc], out_tmp[s1][nk], count, true);
            if (!(nc + Ncb < Nc)) { // last nc iter
              silu_fwd_tpp(out[s1][nk], out[s1][nk]);
              mul_tpp(out[s1][nk], out_tmp[s1][nk], out[s1][nk]);
            }
          } else {
            if (nc == 0) {
              if (with_bias_gate) {
                copy_bias_tpp_rem(bias_gate[nk], out[s1][nk]);
              } else {
                zero_tpp_rem(out[s1][nk]);
              }

              if (with_bias_up) {
                copy_bias_tpp_rem(bias_up[nk], out_tmp[s1][nk]);
              } else {
                zero_tpp_rem(out_tmp[s1][nk]);
              }
            }
            brgemm_tpp_rem(
                in[s1][nc], wt_gate_V[nk][nc], out[s1][nk], count, false);
            brgemm_tpp_rem(
                in[s1][nc], wt_up_V[nk][nc], out_tmp[s1][nk], count, false);
            brgemm_tpp.config();
            if (!(nc + Ncb < Nc)) { // last nc iter
              silu_fwd_tpp_rem(out[s1][nk], out[s1][nk]);
              mul_tpp_rem(out[s1][nk], out_tmp[s1][nk], out[s1][nk]);
            }
          }
        },
        [&]() { brgemm_tpp.config(); },
        [&]() { brgemm_tpp.release(); });
  }
}

template <typename T>
inline void tpp_ffn_swiglu(
    const at::Tensor& t_in,
    const at::Tensor& t_wt_gate,
    const at::Tensor& t_bias_gate,
    const at::Tensor& t_wt_up,
    const at::Tensor& t_bias_up,
    const at::Tensor& t_wt_down,
    const at::Tensor& t_bias_down,
    at::Tensor& t_out,
    int b_vnni) {
  // This is a placeholder implementation for FFN with SwiGLU activation
  // The detailed AMX assembly-level optimization should be implemented here
  // by the user as mentioned in the requirements.
  
  // Step 1: Fuse gate and up projections (similar to tpp_fused_gate_up_proj)
  auto t_wt_gate_ = t_wt_gate;
  auto t_wt_up_ = t_wt_up;
  auto t_wt_down_ = t_wt_down;
  
  auto in_sizes = t_in.sizes();
  auto BS = in_sizes[0] * in_sizes[1];
  bool large_cache_opt = false;
  if (BS > FT_OPT_SIZE) {
    t_wt_gate_ = wt_tensor_for_first_token<T>(t_wt_gate_);
    t_wt_up_ = wt_tensor_for_first_token<T>(t_wt_up_);
    t_wt_down_ = wt_tensor_for_first_token<T>(t_wt_down_);
    large_cache_opt = true;
  }

  auto wt_gate_sizes = t_wt_gate_.sizes();
  auto wt_down_sizes = t_wt_down_.sizes();
  auto C = in_sizes[2];

  auto Nc = wt_gate_sizes[1];
  auto Hc = C / Nc;
  auto Nk_inter = wt_gate_sizes[0];  // Intermediate dimension
  auto Hk_inter = wt_gate_sizes[3];
  auto K_inter = Nk_inter * Hk_inter;
  
  auto Nk = wt_down_sizes[0];  // Output dimension
  auto Hk = wt_down_sizes[3];
  auto K = Nk * Hk;

  // Prepare weight tensors for forward pass
  auto t_wt_gate_V =
      torch_ipex::tpp::wt_tensor_for_fwd(Nk_inter, Hk_inter, Nc, Hc, t_wt_gate_);
  auto t_wt_up_V = 
      torch_ipex::tpp::wt_tensor_for_fwd(Nk_inter, Hk_inter, Nc, Hc, t_wt_up_);
  auto t_wt_down_V = 
      torch_ipex::tpp::wt_tensor_for_fwd(Nk, Hk, Nk_inter, Hk_inter, t_wt_down_);

  // Intermediate tensor to store SwiGLU output
  auto t_swiglu_out = at::empty({in_sizes[0], in_sizes[1], K_inter}, t_in.options());

  auto in = GetVLAPtr<T>(t_in, {Nc, Hc});
  auto wt_gate_V = GetVLAPtr<T>(t_wt_gate_V, {Nc, Hc * Hk_inter});
  auto wt_up_V = GetVLAPtr<T>(t_wt_up_V, {Nc, Hc * Hk_inter});
  auto wt_down_V = GetVLAPtr<T>(t_wt_down_V, {Nk_inter, Hk_inter * Hk});
  auto bias_gate = GetVLAPtr<T>(t_bias_gate, {Hk_inter});
  auto bias_up = GetVLAPtr<T>(t_bias_up, {Hk_inter});
  auto bias_down = GetVLAPtr<T>(t_bias_down, {Hk});
  auto swiglu_out = GetVLAPtr<T>(t_swiglu_out, {Nk_inter, Hk_inter});
  auto out = GetVLAPtr<T>(t_out, {Nk, Hk});

  // Temporary tensor for gate output before SwiGLU
  auto t_gate_tmp = at::empty_like(t_swiglu_out);
  auto gate_tmp = GetVLAPtr<T>(t_gate_tmp, {Nk_inter, Hk_inter});

  auto Ncb = Nc;
  auto BSb = 64L;
  auto rem = BS % 64;
  if (large_cache_opt)
    Ncb = NCB_BLOCK_SIZE;

  bool with_bias_gate = (t_bias_gate.numel() > 0);
  bool with_bias_up = (t_bias_up.numel() > 0);
  bool with_bias_down = (t_bias_down.numel() > 0);

  // Step 1: Compute gate and up projections with SwiGLU activation
  auto copy_bias_gate_tpp = SCOPEIT(CpyBiasTPP<T>(BSb, Hk_inter, K_inter), BIAS);
  auto copy_bias_gate_tpp_rem = SCOPEIT(CpyBiasTPP<T>(rem, Hk_inter, K_inter), BIAS);
  auto copy_bias_up_tpp = SCOPEIT(CpyBiasTPP<T>(BSb, Hk_inter, K_inter), BIAS);
  auto copy_bias_up_tpp_rem = SCOPEIT(CpyBiasTPP<T>(rem, Hk_inter, K_inter), BIAS);
  auto zero_tpp = SCOPEIT(SetZeroTPP<T>(BSb, Hk_inter, K_inter), EW_ZERO);
  auto zero_tpp_rem = SCOPEIT(SetZeroTPP<T>(rem, Hk_inter, K_inter), EW_ZERO);
  
  auto brgemm_gate_tpp = SCOPEITGEMM((BrgemmTPP<T, T>(
      BSb, Hk_inter, Hc, Hc, Hk_inter * Hc, C, Hk_inter, K_inter, 1.0, 0, Ncb, b_vnni)));
  auto brgemm_gate_tpp_rem = SCOPEITGEMM((BrgemmTPP<T, T>(
      rem, Hk_inter, Hc, Hc, Hk_inter * Hc, C, Hk_inter, K_inter, 1.0, 0, Ncb, b_vnni)));
  
  auto silu_fwd_tpp = SCOPEIT(SiLUFwdTPP<T>(BSb, Hk_inter, K_inter, K_inter), ACT);
  auto silu_fwd_tpp_rem = SCOPEIT(SiLUFwdTPP<T>(rem, Hk_inter, K_inter, K_inter), ACT);
  auto mul_tpp = SCOPEIT((MulTPP<T, T>(BSb, Hk_inter, K_inter, K_inter)), EW_MUL);
  auto mul_tpp_rem = SCOPEIT((MulTPP<T, T>(rem, Hk_inter, K_inter, K_inter)), EW_MUL);

  {
    RECORD_SCOPE(tpp_fused_gate_up_proj_krnl, {t_in, t_wt_gate_V});
    
    auto loop_scheme = large_cache_opt ? GEMM_LOOP_SCHEME : "aCb";
    auto igemm_loop = torch_ipex::tpp::ThreadedLoop<3>(
        {{0, Nc, Ncb, false}, {0, BS, BSb}, {Nk_inter}}, loop_scheme);
    igemm_loop(
        [&](int* ind) {
          int nc = ind[0], s1 = ind[1], nk = ind[2];
          auto count = nc + Ncb < Nc ? Ncb : Nc - nc;
          bool is_rem = (s1 + BSb > BS);
          if (!is_rem) {
            if (nc == 0) {
              if (with_bias_gate) {
                copy_bias_gate_tpp(bias_gate[nk], gate_tmp[s1][nk]);
              } else {
                zero_tpp(gate_tmp[s1][nk]);
              }
              if (with_bias_up) {
                copy_bias_up_tpp(bias_up[nk], swiglu_out[s1][nk]);
              } else {
                zero_tpp(swiglu_out[s1][nk]);
              }
            }
            brgemm_gate_tpp(in[s1][nc], wt_gate_V[nk][nc], gate_tmp[s1][nk], count, true);
            brgemm_gate_tpp(in[s1][nc], wt_up_V[nk][nc], swiglu_out[s1][nk], count, true);
            if (!(nc + Ncb < Nc)) { // last nc iter - apply SwiGLU: silu(gate) * up
              silu_fwd_tpp(gate_tmp[s1][nk], gate_tmp[s1][nk]);
              mul_tpp(gate_tmp[s1][nk], swiglu_out[s1][nk], swiglu_out[s1][nk]);
            }
          } else {
            if (nc == 0) {
              if (with_bias_gate) {
                copy_bias_gate_tpp_rem(bias_gate[nk], gate_tmp[s1][nk]);
              } else {
                zero_tpp_rem(gate_tmp[s1][nk]);
              }
              if (with_bias_up) {
                copy_bias_up_tpp_rem(bias_up[nk], swiglu_out[s1][nk]);
              } else {
                zero_tpp_rem(swiglu_out[s1][nk]);
              }
            }
            brgemm_gate_tpp_rem(in[s1][nc], wt_gate_V[nk][nc], gate_tmp[s1][nk], count, false);
            brgemm_gate_tpp_rem(in[s1][nc], wt_up_V[nk][nc], swiglu_out[s1][nk], count, false);
            brgemm_gate_tpp.config();
            if (!(nc + Ncb < Nc)) {
              silu_fwd_tpp_rem(gate_tmp[s1][nk], gate_tmp[s1][nk]);
              mul_tpp_rem(gate_tmp[s1][nk], swiglu_out[s1][nk], swiglu_out[s1][nk]);
            }
          }
        },
        [&]() { brgemm_gate_tpp.config(); },
        [&]() { brgemm_gate_tpp.release(); });
  }

  // Step 2: Down projection
  // NOTE: Here is where AMX assembly-level optimization can be integrated
  // The user should implement optimized matrix multiplication with AMX intrinsics
  auto Ncb_down = Nk_inter;
  if (large_cache_opt)
    Ncb_down = NCB_BLOCK_SIZE;
  
  auto copy_bias_down_tpp = SCOPEIT(CpyBiasTPP<T>(BSb, Hk, K), BIAS);
  auto copy_bias_down_tpp_rem = SCOPEIT(CpyBiasTPP<T>(rem, Hk, K), BIAS);
  auto zero_down_tpp = SCOPEIT(SetZeroTPP<T>(BSb, Hk, K), EW_ZERO);
  auto zero_down_tpp_rem = SCOPEIT(SetZeroTPP<T>(rem, Hk, K), EW_ZERO);
  
  auto brgemm_down_tpp = SCOPEITGEMM((BrgemmTPP<T, T>(
      BSb, Hk, Hk_inter, Hk_inter, Hk * Hk_inter, K_inter, Hk, K, 1.0, 0, Ncb_down, b_vnni)));
  auto brgemm_down_tpp_rem = SCOPEITGEMM((BrgemmTPP<T, T>(
      rem, Hk, Hk_inter, Hk_inter, Hk * Hk_inter, K_inter, Hk, K, 1.0, 0, Ncb_down, b_vnni)));

  {
    RECORD_SCOPE(tpp_linear_krnl, {t_swiglu_out, t_wt_down_V});
    
    auto loop_scheme_down = large_cache_opt ? GEMM_LOOP_SCHEME : "aCb";
    auto down_loop = torch_ipex::tpp::ThreadedLoop<3>(
        {{0, Nk_inter, Ncb_down, false}, {0, BS, BSb}, {Nk}}, loop_scheme_down);
    down_loop(
        [&](int* ind) {
          int nc = ind[0], s1 = ind[1], nk = ind[2];
          auto count = nc + Ncb_down < Nk_inter ? Ncb_down : Nk_inter - nc;
          bool is_rem = (s1 + BSb > BS);
          if (!is_rem) {
            if (nc == 0) {
              if (with_bias_down) {
                copy_bias_down_tpp(bias_down[nk], out[s1][nk]);
              } else {
                zero_down_tpp(out[s1][nk]);
              }
            }
            brgemm_down_tpp(swiglu_out[s1][nc], wt_down_V[nk][nc], out[s1][nk], count, true);
          } else {
            if (nc == 0) {
              if (with_bias_down) {
                copy_bias_down_tpp_rem(bias_down[nk], out[s1][nk]);
              } else {
                zero_down_tpp_rem(out[s1][nk]);
              }
            }
            brgemm_down_tpp_rem(swiglu_out[s1][nc], wt_down_V[nk][nc], out[s1][nk], count, false);
            brgemm_down_tpp.config();
          }
        },
        [&]() { brgemm_down_tpp.config(); },
        [&]() { brgemm_down_tpp.release(); });
  }
}

template <typename T>
inline void tpp_linear_add(
    const at::Tensor t_in,
    const at::Tensor t_in1,
    const at::Tensor t_wt,
    const at::Tensor t_bias,
    at::Tensor t_out,
    float scale,
    int b_vnni) {
  auto t_wt_ = t_wt;
  auto in_sizes = t_in.sizes();
  auto BS = in_sizes[0] * in_sizes[1];
  bool large_cache_opt = false;
  if (BS > FT_OPT_SIZE) { // first token compute
    t_wt_ = wt_tensor_for_first_token<T>(t_wt_);
    large_cache_opt = true;
  }

  auto wt_sizes = t_wt_.sizes();
  auto C = in_sizes[2];

  auto Nc = wt_sizes[1];
  auto Hc = C / Nc;
  auto Nk = wt_sizes[0];
  auto Hk = wt_sizes[3];
  auto K = Nk * Hk;

  auto t_wt_V = torch_ipex::tpp::wt_tensor_for_fwd(Nk, Hk, Nc, Hc, t_wt_);

  auto in = GetVLAPtr<T>(t_in, {Nc, Hc});
  auto in1 = GetVLAPtr<T>(t_in1, {Nk, Hk});
  auto wt_V = GetVLAPtr<T>(t_wt_V, {Nc, Hc * Hk});
  auto bias = GetVLAPtr<T>(t_bias, {Hk});
  auto out = GetVLAPtr<T>(t_out, {Nk, Hk});

  auto Ncb = Nc;
  auto BSb = 64L;
  auto rem = BS % 64;
  if (large_cache_opt)
    Ncb = NCB_BLOCK_SIZE;

  bool with_bias = (t_bias.numel() > 0);
  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<T>(BSb, Hk, K), BIAS);
  auto copy_bias_tpp_rem = SCOPEIT(CpyBiasTPP<T>(rem, Hk, K), BIAS);
  auto zero_tpp = SCOPEIT(SetZeroTPP<T>(BSb, Hk, K), EW_ZERO);
  auto zero_tpp_rem = SCOPEIT(SetZeroTPP<T>(rem, Hk, K), EW_ZERO);
  auto brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T, T>(
      BSb, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb, b_vnni)));
  auto brgemm_tpp_rem = SCOPEITGEMM((BrgemmTPP<T, T>(
      rem, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb, b_vnni)));
  auto sadd_tpp = SCOPEIT((ScaleAddTPP<T, T>(BSb, Hk, K, K)), EW_ADD);
  auto sadd_tpp_rem = SCOPEIT((ScaleAddTPP<T, T>(rem, Hk, K, K)), EW_ADD);

  {
    RECORD_SCOPE(tpp_linear_add_krnl, {t_in, t_wt_V});

    auto loop_scheme = large_cache_opt ? GEMM_LOOP_SCHEME : "aCb";
    auto ogemm_loop = torch_ipex::tpp::ThreadedLoop<3>(
        {{0, Nc, Ncb, false}, {0L, BS, BSb}, {Nk}}, loop_scheme);
    ogemm_loop(
        [&](int* ind) {
          int nc = ind[0], s1 = ind[1], nk = ind[2];
          auto count = nc + Ncb < Nc ? Ncb : Nc - nc;
          bool is_rem = (s1 + BSb > BS);
          if (!is_rem) {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp(bias[nk], out[s1][nk]);
              } else {
                zero_tpp(out[s1][nk]);
              }
            }
            brgemm_tpp(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, true);
            if (!(nc + Ncb < Nc)) { // last nc iter
              sadd_tpp(in1[s1][nk], out[s1][nk], scale);
            }
          } else {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp_rem(bias[nk], out[s1][nk]);
              } else {
                zero_tpp_rem(out[s1][nk]);
              }
            }
            brgemm_tpp_rem(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, false);
            brgemm_tpp.config();
            if (!(nc + Ncb < Nc)) { // last nc iter
              sadd_tpp_rem(in1[s1][nk], out[s1][nk], scale);
            }
          }
        },
        [&]() { brgemm_tpp.config(); },
        [&]() { brgemm_tpp.release(); });
  }
}

template <typename T, typename Tout = T>
inline void tpp_linear_silu(
    const at::Tensor t_in,
    const at::Tensor t_wt,
    const at::Tensor t_bias,
    at::Tensor t_out,
    int b_vnni) {
  auto t_wt_ = t_wt;
  auto in_sizes = t_in.sizes();
  auto BS = in_sizes[0] * in_sizes[1];
  bool large_cache_opt = false;
  if (BS > FT_OPT_SIZE) { // first token compute
    t_wt_ = wt_tensor_for_first_token<T>(t_wt_);
    large_cache_opt = true;
  }

  auto wt_sizes = t_wt_.sizes();
  auto C = in_sizes[2];

  auto Nc = wt_sizes[1];
  auto Hc = C / Nc;
  auto Nk = wt_sizes[0];
  auto Hk = wt_sizes[3];
  auto K = Nk * Hk;

  auto t_wt_V = torch_ipex::tpp::wt_tensor_for_fwd(Nk, Hk, Nc, Hc, t_wt_);

  auto in = GetVLAPtr<T>(t_in, {Nc, Hc});
  auto wt_V = GetVLAPtr<T>(t_wt_V, {Nc, Hc * Hk});
  auto bias = GetVLAPtr<Tout>(t_bias, {Hk});
  auto out = GetVLAPtr<Tout>(t_out, {Nk, Hk});

  auto Ncb = Nc;
  auto BSb = 64L;
  auto rem = BS % 64;
  if (large_cache_opt)
    Ncb = NCB_BLOCK_SIZE;

  bool with_bias = (t_bias.numel() > 0);
  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<Tout>(BSb, Hk, K), BIAS);
  auto copy_bias_tpp_rem = SCOPEIT(CpyBiasTPP<Tout>(rem, Hk, K), BIAS);
  auto zero_tpp = SCOPEIT(SetZeroTPP<Tout>(BSb, Hk, K), EW_ZERO);
  auto zero_tpp_rem = SCOPEIT(SetZeroTPP<Tout>(rem, Hk, K), EW_ZERO);
  auto brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T, Tout>(
      BSb, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb, b_vnni)));
  auto brgemm_tpp_rem = SCOPEITGEMM((BrgemmTPP<T, Tout>(
      rem, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb, b_vnni)));
  auto silu_fwd_tpp = SCOPEIT(SiLUFwdTPP<Tout>(BSb, Hk, K, K), ACT);
  auto silu_fwd_tpp_rem = SCOPEIT(SiLUFwdTPP<Tout>(rem, Hk, K, K), ACT);

  {
    RECORD_SCOPE(tpp_linear_silu_krnl, {t_in, t_wt_V});

    auto loop_scheme = large_cache_opt ? GEMM_LOOP_SCHEME : "aCb";
    auto igemm_loop = torch_ipex::tpp::ThreadedLoop<3>(
        {{0, Nc, Ncb, false}, {0, BS, BSb}, {Nk}}, loop_scheme);
    igemm_loop(
        [&](int* ind) {
          int nc = ind[0], s1 = ind[1], nk = ind[2];
          auto count = nc + Ncb < Nc ? Ncb : Nc - nc;
          bool is_rem = (s1 + BSb > BS);
          if (!is_rem) {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp(bias[nk], out[s1][nk]);
              } else {
                zero_tpp(out[s1][nk]);
              }
            }
            brgemm_tpp(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, true);
            if (!(nc + Ncb < Nc)) { // last nc iter
              silu_fwd_tpp(out[s1][nk], out[s1][nk]);
            }
          } else {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp_rem(bias[nk], out[s1][nk]);
              } else {
                zero_tpp_rem(out[s1][nk]);
              }
            }
            brgemm_tpp_rem(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, false);
            brgemm_tpp.config();
            if (!(nc + Ncb < Nc)) { // last nc iter
              silu_fwd_tpp_rem(out[s1][nk], out[s1][nk]);
            }
          }
        },
        [&]() { brgemm_tpp.config(); },
        [&]() { brgemm_tpp.release(); });
  }
}

template <typename T, typename Tout = T>
inline void tpp_linear_relu(
    const at::Tensor t_in,
    const at::Tensor t_wt,
    const at::Tensor t_bias,
    at::Tensor t_out,
    int b_vnni) {
  auto t_wt_ = t_wt;
  auto in_sizes = t_in.sizes();
  auto BS = in_sizes[0] * in_sizes[1];
  bool large_cache_opt = false;
  if (BS > FT_OPT_SIZE) { // first token compute
    t_wt_ = wt_tensor_for_first_token<T>(t_wt_);
    large_cache_opt = true;
  }

  auto wt_sizes = t_wt_.sizes();
  auto C = in_sizes[2];

  auto Nc = wt_sizes[1];
  auto Hc = C / Nc;
  auto Nk = wt_sizes[0];
  auto Hk = wt_sizes[3];
  auto K = Nk * Hk;

  auto t_wt_V = torch_ipex::tpp::wt_tensor_for_fwd(Nk, Hk, Nc, Hc, t_wt_);

  auto in = GetVLAPtr<T>(t_in, {Nc, Hc});
  auto wt_V = GetVLAPtr<T>(t_wt_V, {Nc, Hc * Hk});
  auto bias = GetVLAPtr<Tout>(t_bias, {Hk});
  auto out = GetVLAPtr<Tout>(t_out, {Nk, Hk});

  auto Ncb = Nc;
  auto BSb = 64L;
  auto rem = BS % 64;
  if (large_cache_opt)
    Ncb = NCB_BLOCK_SIZE;

  bool with_bias = (t_bias.numel() > 0);
  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<Tout>(BSb, Hk, K), BIAS);
  auto copy_bias_tpp_rem = SCOPEIT(CpyBiasTPP<Tout>(rem, Hk, K), BIAS);
  auto zero_tpp = SCOPEIT(SetZeroTPP<Tout>(BSb, Hk, K), EW_ZERO);
  auto zero_tpp_rem = SCOPEIT(SetZeroTPP<Tout>(rem, Hk, K), EW_ZERO);
  auto brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T, Tout>(
      BSb, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb, b_vnni)));
  auto brgemm_tpp_rem = SCOPEITGEMM((BrgemmTPP<T, Tout>(
      rem, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb, b_vnni)));
  auto relu_fwd_tpp = SCOPEIT(ReLUFwdTPP<Tout>(BSb, Hk, K, K, false), ACT);
  auto relu_fwd_tpp_rem = SCOPEIT(ReLUFwdTPP<Tout>(rem, Hk, K, K, false), ACT);

  {
    RECORD_SCOPE(tpp_linear_relu_krnl, {t_in, t_wt_V});

    auto loop_scheme = large_cache_opt ? GEMM_LOOP_SCHEME : "aCb";
    auto igemm_loop = torch_ipex::tpp::ThreadedLoop<3>(
        {{0, Nc, Ncb, false}, {0, BS, BSb}, {Nk}}, loop_scheme);
    igemm_loop(
        [&](int* ind) {
          int nc = ind[0], s1 = ind[1], nk = ind[2];
          auto count = nc + Ncb < Nc ? Ncb : Nc - nc;
          bool is_rem = (s1 + BSb > BS);
          if (!is_rem) {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp(bias[nk], out[s1][nk]);
              } else {
                zero_tpp(out[s1][nk]);
              }
            }
            brgemm_tpp(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, true);
            if (!(nc + Ncb < Nc)) { // last nc iter
              relu_fwd_tpp(out[s1][nk], out[s1][nk]);
            }
          } else {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp_rem(bias[nk], out[s1][nk]);
              } else {
                zero_tpp_rem(out[s1][nk]);
              }
            }
            brgemm_tpp_rem(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, false);
            brgemm_tpp.config();
            if (!(nc + Ncb < Nc)) { // last nc iter
              relu_fwd_tpp_rem(out[s1][nk], out[s1][nk]);
            }
          }
        },
        [&]() { brgemm_tpp.config(); },
        [&]() { brgemm_tpp.release(); });
  }
}

} // namespace tpp
} // namespace torch_ipex
