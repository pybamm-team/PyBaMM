#include "BlockPreconditioner.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include <klu.h>

namespace {

bool usable_pivot(double value) { return value != 0.0 && std::isfinite(value); }

}  // namespace

/**
 * @brief Runtime state of a tree node: setup reads the node's matrix from a
 * source array, and solve applies the node's inverse in place
 */
class BlockNode {
public:
  virtual ~BlockNode() = default;
  virtual int setup(const double *source) = 0;
  virtual void solve(double *x) = 0;
};

namespace {

std::unique_ptr<BlockNode> make_node(const BlockNodeData &data);

class LeafNode : public BlockNode {
public:
  explicit LeafNode(const BlockNodeData &data) : d_(data) {
    values_.assign(d_.gather.size(), 0.0);
    if (d_.solver == BlockNodeData::KLU) {
      klu_l_defaults(&common_);
    } else if (d_.solver == BlockNodeData::ILU) {
      marker_.assign(d_.size, -1);
    } else if (d_.solver == BlockNodeData::EXTERNAL) {
      context_ = d_.external.create(d_.size, d_.indptr.data(), d_.indices.data(),
                                    d_.external_options.c_str());
      if (context_ == nullptr) {
        throw std::runtime_error("the external block solver could not be created");
      }
    }
  }

  ~LeafNode() override {
    if (context_ != nullptr) d_.external.destroy(context_);
    if (numeric_ != nullptr) klu_l_free_numeric(&numeric_, &common_);
    if (symbolic_ != nullptr) klu_l_free_symbolic(&symbolic_, &common_);
  }

  int setup(const double *source) override {
    for (size_t k = 0; k < values_.size(); k++) {
      values_[k] = d_.gather[k] >= 0 ? source[d_.gather[k]] : 0.0;
    }
    switch (d_.solver) {
      case BlockNodeData::KLU: return factor_klu();
      case BlockNodeData::ILU: return factor_ilu();
      case BlockNodeData::JACOBI: return factor_jacobi();
      case BlockNodeData::TRIDIAGONAL: return factor_tridiagonal();
      case BlockNodeData::EXTERNAL: return d_.external.setup(context_, values_.data());
    }
    return 1;
  }

  void solve(double *x) override {
    const int64_t m = d_.size;
    switch (d_.solver) {
      case BlockNodeData::KLU:
        klu_l_solve(symbolic_, numeric_, m, 1, x, &common_);
        return;
      case BlockNodeData::EXTERNAL:
        d_.external.solve(context_, x);
        return;
      case BlockNodeData::JACOBI:
        for (int64_t i = 0; i < m; i++) x[i] *= values_[i];
        return;
      case BlockNodeData::TRIDIAGONAL: {
        // values_ holds rows of (sub-diagonal, inverse pivot, scaled super)
        for (int64_t i = 0; i < m; i++) {
          const double previous = i > 0 ? values_[3 * i] * x[i - 1] : 0.0;
          x[i] = (x[i] - previous) * values_[3 * i + 1];
        }
        for (int64_t i = m - 2; i >= 0; i--) x[i] -= values_[3 * i + 2] * x[i + 1];
        return;
      }
      case BlockNodeData::ILU: {
        const int64_t *pointer = d_.indptr.data();
        const int64_t *index = d_.indices.data();
        const int64_t *diagonal = d_.diagonal.data();
        const double *a = values_.data();
        for (int64_t i = 0; i < m; i++) {
          double sum = x[i];
          for (int64_t p = pointer[i]; p < diagonal[i]; p++) sum -= a[p] * x[index[p]];
          x[i] = sum;
        }
        for (int64_t i = m - 1; i >= 0; i--) {
          double sum = x[i];
          for (int64_t p = diagonal[i] + 1; p < pointer[i + 1]; p++) {
            sum -= a[p] * x[index[p]];
          }
          x[i] = sum / a[diagonal[i]];
        }
        return;
      }
    }
  }

private:
  int factor_klu() {
    // KLU reads the pattern only
    auto *pointer = const_cast<int64_t *>(d_.indptr.data());
    auto *index = const_cast<int64_t *>(d_.indices.data());
    if (symbolic_ == nullptr) {
      symbolic_ = klu_l_analyze(d_.size, pointer, index, &common_);
      if (symbolic_ == nullptr) return 1;
    }
    bool refactored = false;
    if (numeric_ != nullptr) {
      refactored = klu_l_refactor(pointer, index, values_.data(), symbolic_, numeric_,
                                  &common_) != 0 &&
                   common_.status == KLU_OK &&
                   klu_l_rcond(symbolic_, numeric_, &common_) != 0 &&
                   common_.rcond > 1e-12;
    }
    if (!refactored) {
      // A pivot order chosen for earlier values can be unstable for these
      if (numeric_ != nullptr) klu_l_free_numeric(&numeric_, &common_);
      numeric_ = klu_l_factor(pointer, index, values_.data(), symbolic_, &common_);
      if (numeric_ == nullptr || common_.status != KLU_OK) return 1;
    }
    return 0;
  }

  int factor_ilu() {
    // ILU(0) on the stored pattern, which includes any level-of-fill entries
    const int64_t *pointer = d_.indptr.data();
    const int64_t *index = d_.indices.data();
    const int64_t *diagonal = d_.diagonal.data();
    double *a = values_.data();
    for (int64_t i = 0; i < d_.size; i++) {
      for (int64_t p = pointer[i]; p < pointer[i + 1]; p++) marker_[index[p]] = p;
      for (int64_t p = pointer[i]; p < diagonal[i]; p++) {
        const int64_t k = index[p];
        if (!usable_pivot(a[diagonal[k]])) return 1;
        a[p] /= a[diagonal[k]];
        for (int64_t q = diagonal[k] + 1; q < pointer[k + 1]; q++) {
          const int64_t position = marker_[index[q]];
          if (position >= 0) a[position] -= a[p] * a[q];
        }
      }
      for (int64_t p = pointer[i]; p < pointer[i + 1]; p++) marker_[index[p]] = -1;
      if (!usable_pivot(a[diagonal[i]])) return 1;
    }
    return 0;
  }

  int factor_jacobi() {
    for (double &value : values_) {
      if (!usable_pivot(value)) return 1;
      value = 1.0 / value;
    }
    return 0;
  }

  int factor_tridiagonal() {
    // Thomas algorithm: rows of (sub, diagonal, super) become
    // (sub, inverse pivot, super / pivot)
    for (int64_t i = 0; i < d_.size; i++) {
      double *row = values_.data() + 3 * i;
      const double pivot = row[1] - (i > 0 ? row[0] * values_[3 * (i - 1) + 2] : 0.0);
      if (!usable_pivot(pivot)) return 1;
      row[1] = 1.0 / pivot;
      row[2] *= row[1];
    }
    return 0;
  }

  const BlockNodeData &d_;
  std::vector<double> values_;
  klu_l_common common_;
  klu_l_symbolic *symbolic_ = nullptr;
  klu_l_numeric *numeric_ = nullptr;
  IndexVector marker_;
  void *context_ = nullptr;
};

// Additive (block Jacobi) or multiplicative (block Gauss-Seidel) combination
class CompositeNode : public BlockNode {
public:
  explicit CompositeNode(const BlockNodeData &data) : d_(data) {
    size_t largest = 0;
    for (size_t c = 0; c < d_.children.size(); c++) {
      children_.push_back(make_node(*d_.children[c]));
      largest = std::max(largest, d_.positions[c].size());
    }
    local_.assign(largest, 0.0);
    sweep_.assign(d_.sweep_slot.size(), 0.0);
  }

  int setup(const double *source) override {
    for (size_t k = 0; k < sweep_.size(); k++) sweep_[k] = source[d_.sweep_slot[k]];
    for (auto &child : children_) {
      if (child->setup(source) != 0) return 1;
    }
    return 0;
  }

  void solve(double *x) override {
    for (size_t c = 0; c < children_.size(); c++) {
      if (d_.type == BlockNodeData::MULTIPLICATIVE) {
        // x holds the solutions of the earlier children
        for (int64_t e = d_.sweep_ptr[c]; e < d_.sweep_ptr[c + 1]; e++) {
          x[d_.sweep_row[e]] -= sweep_[e] * x[d_.sweep_col[e]];
        }
      }
      const IndexVector &position = d_.positions[c];
      for (size_t i = 0; i < position.size(); i++) local_[i] = x[position[i]];
      children_[c]->solve(local_.data());
      for (size_t i = 0; i < position.size(); i++) x[position[i]] = local_[i];
    }
  }

private:
  const BlockNodeData &d_;
  std::vector<std::unique_ptr<BlockNode>> children_;
  std::vector<double> local_, sweep_;
};

class SchurNode : public BlockNode {
public:
  explicit SchurNode(const BlockNodeData &data)
      : d_(data),
        eliminated_(make_node(*data.children[0])),
        complement_(make_node(*data.children[1])) {
    x0_.assign(d_.positions[0].size(), 0.0);
    x1_.assign(d_.positions[1].size(), 0.0);
    t_.assign(d_.positions[0].size(), 0.0);
    s_.assign(d_.s_gather.size(), 0.0);
    m01_.assign(d_.m01_slot.size(), 0.0);
    m10_.assign(d_.m10_slot.size(), 0.0);
  }

  int setup(const double *source) override {
    if (eliminated_->setup(source) != 0) return 1;
    for (size_t k = 0; k < m01_.size(); k++) m01_[k] = source[d_.m01_slot[k]];
    for (size_t k = 0; k < m10_.size(); k++) m10_[k] = source[d_.m10_slot[k]];
    for (size_t k = 0; k < s_.size(); k++) {
      s_[k] = d_.s_gather[k] >= 0 ? source[d_.s_gather[k]] : 0.0;
    }
    // S = M11 - M10 M00^-1 M01, exactly or with M00 replaced by its diagonal
    for (size_t u = 0; u < d_.diagonal_s.size(); u++) {
      const double pivot = source[d_.diagonal_m00[u]];
      if (!usable_pivot(pivot)) return 1;
      s_[d_.diagonal_s[u]] -=
          source[d_.diagonal_m10[u]] * source[d_.diagonal_m01[u]] / pivot;
    }
    // No two columns of a colour reach the same independent block of M00, so
    // one solve per colour separates them
    const size_t colours = d_.colour_ptr.empty() ? 0 : d_.colour_ptr.size() - 1;
    for (size_t c = 0; c < colours; c++) {
      std::fill(t_.begin(), t_.end(), 0.0);
      for (int64_t e = d_.colour_ptr[c]; e < d_.colour_ptr[c + 1]; e++) {
        t_[d_.colour_row[e]] += source[d_.colour_slot[e]];
      }
      eliminated_->solve(t_.data());
      for (int64_t u = d_.update_ptr[c]; u < d_.update_ptr[c + 1]; u++) {
        s_[d_.update_s[u]] -= source[d_.update_m10[u]] * t_[d_.update_state[u]];
      }
    }
    return complement_->setup(s_.data());
  }

  void solve(double *x) override {
    const IndexVector &e = d_.positions[0];
    const IndexVector &c = d_.positions[1];
    for (size_t i = 0; i < e.size(); i++) x0_[i] = x[e[i]];
    for (size_t i = 0; i < c.size(); i++) x1_[i] = x[c[i]];

    eliminated_->solve(x0_.data());
    for (size_t k = 0; k < m10_.size(); k++) {
      x1_[d_.m10_row[k]] -= m10_[k] * x0_[d_.m10_col[k]];
    }
    complement_->solve(x1_.data());
    std::fill(t_.begin(), t_.end(), 0.0);
    for (size_t k = 0; k < m01_.size(); k++) {
      t_[d_.m01_row[k]] += m01_[k] * x1_[d_.m01_col[k]];
    }
    eliminated_->solve(t_.data());

    for (size_t i = 0; i < e.size(); i++) x[e[i]] = x0_[i] - t_[i];
    for (size_t i = 0; i < c.size(); i++) x[c[i]] = x1_[i];
  }

private:
  const BlockNodeData &d_;
  std::unique_ptr<BlockNode> eliminated_, complement_;
  std::vector<double> x0_, x1_, t_, s_, m01_, m10_;
};

std::unique_ptr<BlockNode> make_node(const BlockNodeData &data) {
  switch (data.type) {
    case BlockNodeData::LEAF: return std::make_unique<LeafNode>(data);
    case BlockNodeData::ADDITIVE:
    case BlockNodeData::MULTIPLICATIVE: return std::make_unique<CompositeNode>(data);
    case BlockNodeData::SCHUR: return std::make_unique<SchurNode>(data);
  }
  throw std::invalid_argument("unknown preconditioner node type");
}

}  // namespace

BlockPreconditioner::BlockPreconditioner(std::shared_ptr<const BlockAnalysis> analysis)
    : analysis_(std::move(analysis)),
      root_(make_node(*analysis_->root)),
      size_(analysis_->size) {}

BlockPreconditioner::~BlockPreconditioner() = default;

int BlockPreconditioner::setup(const double *matrix_data) {
  return root_->setup(matrix_data);
}

void BlockPreconditioner::solve(const double *r, double *z) {
  std::memcpy(z, r, size_ * sizeof(double));
  root_->solve(z);
}
