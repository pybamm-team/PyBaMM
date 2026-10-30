#include "BlockPreconditioner.hpp"

#include <algorithm>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

#include <pybind11/numpy.h>

namespace {

/**
 * @brief A block of a matrix pattern in CSC form; slot[k] is the position of
 * entry k in the source array the block's node reads
 */
struct Block {
  int64_t rows = 0, cols = 0;
  IndexVector indptr, indices, slot;

  int64_t nnz() const { return static_cast<int64_t>(indices.size()); }
};

IndexVector read_indices(const py::dict &node, const char *key) {
  if (!node.contains(key)) {
    throw std::invalid_argument(
        std::string("preconditioner tree node is missing \"") + key + "\"");
  }
  auto array = node[key].cast<
      py::array_t<int64_t, py::array::c_style | py::array::forcecast>>();
  return IndexVector(array.data(), array.data() + array.size());
}

std::string read_string(const py::dict &node, const char *key) {
  if (!node.contains(key)) {
    throw std::invalid_argument(
        std::string("preconditioner tree node is missing \"") + key + "\"");
  }
  return node[key].cast<std::string>();
}

void collect_indices(const py::dict &node, IndexVector &out) {
  if (read_string(node, "type") == "leaf") {
    IndexVector indices = read_indices(node, "indices");
    out.insert(out.end(), indices.begin(), indices.end());
    return;
  }
  for (const auto &child : node["children"].cast<py::list>()) {
    collect_indices(child.cast<py::dict>(), out);
  }
}

IndexVector sorted_indices(const py::dict &node) {
  IndexVector out;
  collect_indices(node, out);
  std::sort(out.begin(), out.end());
  if (std::adjacent_find(out.begin(), out.end()) != out.end()) {
    throw std::invalid_argument("a state appears in two preconditioner blocks");
  }
  return out;
}

// The rows x cols sub-block of p, for sorted local index lists
Block extract(const Block &p, const IndexVector &rows, const IndexVector &cols) {
  IndexVector row_map(p.rows, -1);
  for (size_t k = 0; k < rows.size(); k++) row_map[rows[k]] = static_cast<int64_t>(k);
  Block b;
  b.rows = static_cast<int64_t>(rows.size());
  b.cols = static_cast<int64_t>(cols.size());
  b.indptr.reserve(cols.size() + 1);
  b.indptr.push_back(0);
  for (int64_t c : cols) {
    for (int64_t e = p.indptr[c]; e < p.indptr[c + 1]; e++) {
      const int64_t r = row_map[p.indices[e]];
      if (r >= 0) {
        b.indices.push_back(r);
        b.slot.push_back(p.slot[e]);
      }
    }
    b.indptr.push_back(b.nnz());
  }
  return b;
}

template <class F>
void for_each_entry(const Block &b, F f) {
  for (int64_t c = 0; c < b.cols; c++) {
    for (int64_t e = b.indptr[c]; e < b.indptr[c + 1]; e++) f(b.indices[e], c, b.slot[e]);
  }
}

// Connected components of the symmetrised pattern of a square block
IndexVector components(const Block &b) {
  IndexVector parent(b.rows);
  std::iota(parent.begin(), parent.end(), 0);
  auto find = [&](int64_t x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };
  for_each_entry(b, [&](int64_t r, int64_t c, int64_t) {
    const int64_t a = find(r), d = find(c);
    if (a != d) parent[std::max(a, d)] = std::min(a, d);
  });
  IndexVector label(b.rows, -1), id(b.rows, -1);
  int64_t next = 0;
  for (int64_t i = 0; i < b.rows; i++) {
    const int64_t root = find(i);
    if (id[root] < 0) id[root] = next++;
    label[i] = id[root];
  }
  return label;
}

// Group (key, value) pairs by key in [0, number_of_keys) into ptr/values
void group(const IndexVector &key, int64_t number_of_keys, IndexVector &ptr,
           IndexVector &order) {
  ptr.assign(number_of_keys + 1, 0);
  for (int64_t k : key) ptr[k + 1]++;
  for (int64_t k = 0; k < number_of_keys; k++) ptr[k + 1] += ptr[k];
  order.assign(key.size(), 0);
  IndexVector next(ptr.begin(), ptr.end() - 1);
  for (size_t e = 0; e < key.size(); e++) order[next[key[e]]++] = static_cast<int64_t>(e);
}

template <class T>
IndexVector permuted(const T &values, const IndexVector &order) {
  IndexVector out(order.size());
  for (size_t k = 0; k < order.size(); k++) out[k] = values[order[k]];
  return out;
}

// CSC block from (row, col) pairs, unique and sorted; returns lookup for pairs
Block assemble(int64_t size, std::vector<std::pair<int64_t, int64_t>> &col_row) {
  std::sort(col_row.begin(), col_row.end());
  col_row.erase(std::unique(col_row.begin(), col_row.end()), col_row.end());
  Block b;
  b.rows = b.cols = size;
  b.indptr.assign(size + 1, 0);
  for (const auto &[c, r] : col_row) {
    b.indptr[c + 1]++;
    b.indices.push_back(r);
  }
  for (int64_t c = 0; c < size; c++) b.indptr[c + 1] += b.indptr[c];
  b.slot.resize(b.indices.size());
  std::iota(b.slot.begin(), b.slot.end(), 0);
  return b;
}

int64_t position(const Block &b, int64_t r, int64_t c) {
  const auto first = b.indices.begin() + b.indptr[c];
  const auto last = b.indices.begin() + b.indptr[c + 1];
  const auto it = std::lower_bound(first, last, r);
  if (it == last || *it != r) throw std::logic_error("entry missing from pattern");
  return static_cast<int64_t>(it - b.indices.begin());
}

std::shared_ptr<BlockNodeData> lower(const py::dict &spec, const Block &p,
                                      const IndexVector &node_indices,
                                      IndexVector &local_of);

std::shared_ptr<BlockNodeData> lower_leaf(const py::dict &spec, const Block &p) {
  auto node = std::make_shared<BlockNodeData>();
  node->type = BlockNodeData::LEAF;
  node->size = p.rows;
  const std::string solver = read_string(spec, "solver");
  const int64_t m = p.rows;

  if (solver == "klu" || solver == "external") {
    node->indptr = p.indptr;
    node->indices = p.indices;
    node->gather = p.slot;
    node->exact = solver == "klu";
    node->solver = solver == "klu" ? BlockNodeData::KLU : BlockNodeData::EXTERNAL;
    if (solver == "external") {
      auto capsule = spec["capsule"].cast<py::capsule>();
      if (capsule.name() == nullptr ||
          std::string(capsule.name()) != PYBAMM_BLOCK_SOLVER_CAPSULE) {
        throw std::invalid_argument(
            "an external block solver must be a capsule named \"" PYBAMM_BLOCK_SOLVER_CAPSULE "\"");
      }
      const auto *plugin = capsule.get_pointer<pybamm_block_solver>();
      if (plugin == nullptr || plugin->abi_version != PYBAMM_BLOCK_SOLVER_ABI_VERSION) {
        throw std::invalid_argument(
            "the external block solver was built for another pybammsolvers interface");
      }
      node->external = *plugin;
      node->external_options = read_string(spec, "options");
    }
    return node;
  }
  if (solver == "jacobi") {
    node->solver = BlockNodeData::JACOBI;
    node->gather.assign(m, -1);
    bool diagonal_only = true;
    for_each_entry(p, [&](int64_t r, int64_t c, int64_t slot) {
      if (r == c) node->gather[r] = slot;
      else diagonal_only = false;
    });
    node->exact = diagonal_only;
    return node;
  }
  if (solver == "tridiagonal") {
    // Rows of (sub-diagonal, diagonal, super-diagonal)
    node->solver = BlockNodeData::TRIDIAGONAL;
    node->gather.assign(3 * m, -1);
    bool retained = true;
    for_each_entry(p, [&](int64_t r, int64_t c, int64_t slot) {
      if (r == c) node->gather[3 * r + 1] = slot;
      else if (r == c + 1) node->gather[3 * r] = slot;
      else if (r == c - 1) node->gather[3 * r + 2] = slot;
      else retained = false;
    });
    node->exact = retained;
    return node;
  }
  if (solver == "ilu") {
    node->solver = BlockNodeData::ILU;
    const int64_t fill = spec["fill"].cast<int64_t>();
    // Rows of (column, slot), with a stored diagonal to pivot on
    std::vector<std::vector<std::pair<int64_t, int64_t>>> rows(m);
    for_each_entry(p, [&](int64_t r, int64_t c, int64_t slot) { rows[r].push_back({c, slot}); });
    for (int64_t i = 0; i < m; i++) {
      auto &row = rows[i];
      std::sort(row.begin(), row.end());
      auto it = std::lower_bound(row.begin(), row.end(), i,
                                 [](const auto &entry, int64_t c) { return entry.first < c; });
      if (it == row.end() || it->first != i) row.insert(it, {i, -1});
    }
    // ILU(k) level-of-fill: fill entries have no source slot
    std::vector<std::vector<std::pair<int64_t, int64_t>>> upper(m);
    node->indptr.assign(1, 0);
    for (int64_t i = 0; i < m; i++) {
      std::map<int64_t, std::pair<int64_t, int64_t>> row;  // column -> (level, slot)
      for (const auto &[c, slot] : rows[i]) row.emplace(c, std::make_pair(0, slot));
      if (fill > 0) {
        for (auto it = row.begin(); it != row.end() && it->first < i; ++it) {
          const int64_t level_ik = it->second.first;
          for (const auto &[j, level_kj] : upper[it->first]) {
            const int64_t level = level_ik + level_kj + 1;
            if (level > fill) continue;
            auto found = row.find(j);
            if (found == row.end()) row.emplace(j, std::make_pair(level, -1));
            else found->second.first = std::min(found->second.first, level);
          }
        }
      }
      for (const auto &[j, entry] : row) {
        if (j == i) node->diagonal.push_back(static_cast<int64_t>(node->indices.size()));
        node->indices.push_back(j);
        node->gather.push_back(entry.second);
        if (j > i) upper[i].push_back({j, entry.first});
      }
      node->indptr.push_back(static_cast<int64_t>(node->indices.size()));
    }
    node->exact = false;
    return node;
  }
  throw std::invalid_argument("unknown block solver \"" + solver + "\"");
}

std::shared_ptr<BlockNodeData> lower_schur(const py::dict &spec, const Block &p,
                                            const IndexVector &e, const IndexVector &c,
                                            const py::list &children,
                                            const IndexVector &child0_indices,
                                            const IndexVector &child1_indices,
                                            IndexVector &local_of) {
  auto node = std::make_shared<BlockNodeData>();
  node->type = BlockNodeData::SCHUR;
  node->size = p.rows;
  node->positions = {e, c};
  const std::string mode = read_string(spec, "schur_complement");
  if (mode == "exact") node->schur_complement = BlockNodeData::EXACT;
  else if (mode == "diagonal") node->schur_complement = BlockNodeData::DIAGONAL;
  else if (mode == "none") node->schur_complement = BlockNodeData::NONE;
  else throw std::invalid_argument("unknown Schur complement \"" + mode + "\"");

  const Block m00 = extract(p, e, e), m01 = extract(p, e, c);
  const Block m10 = extract(p, c, e), m11 = extract(p, c, c);
  for_each_entry(m01, [&](int64_t r, int64_t col, int64_t slot) {
    node->m01_row.push_back(r);
    node->m01_col.push_back(col);
    node->m01_slot.push_back(slot);
  });
  for_each_entry(m10, [&](int64_t r, int64_t col, int64_t slot) {
    node->m10_row.push_back(r);
    node->m10_col.push_back(col);
    node->m10_slot.push_back(slot);
  });

  const int64_t nc = m11.rows;
  std::vector<std::pair<int64_t, int64_t>> s_entries;  // (column, row)
  for_each_entry(m11, [&](int64_t r, int64_t col, int64_t) { s_entries.push_back({col, r}); });

  // Updates as (complement row, complement column) plus their operands
  IndexVector update_row, update_col, update_a, update_b, update_c;
  if (node->schur_complement == BlockNodeData::EXACT) {
    // M00^-1 couples only states in one connected component of M00, so columns
    // of M01 touching disjoint components share a solve
    const IndexVector component = components(m00);
    const int64_t number_of_components =
        component.empty() ? 0 : *std::max_element(component.begin(), component.end()) + 1;
    std::vector<IndexVector> columns_of(number_of_components);
    std::vector<IndexVector> colours_of(number_of_components);
    IndexVector colour(nc, 0), stamp(number_of_components, -1), taken;
    for (int64_t j = 0; j < nc; j++) {
      IndexVector touched;
      for (int64_t q = m01.indptr[j]; q < m01.indptr[j + 1]; q++) {
        const int64_t k = component[m01.indices[q]];
        if (stamp[k] != j) {
          stamp[k] = j;
          touched.push_back(k);
        }
      }
      taken.clear();
      for (int64_t k : touched) taken.insert(taken.end(), colours_of[k].begin(), colours_of[k].end());
      std::sort(taken.begin(), taken.end());
      int64_t free = 0;
      for (int64_t t : taken) {
        if (t == free) free++;
        else if (t > free) break;
      }
      colour[j] = free;
      for (int64_t k : touched) {
        colours_of[k].push_back(free);
        columns_of[k].push_back(j);
      }
    }
    const int64_t number_of_colours =
        nc == 0 ? 0 : *std::max_element(colour.begin(), colour.end()) + 1;

    IndexVector entry_colour(node->m01_col.size());
    for (size_t q = 0; q < entry_colour.size(); q++) entry_colour[q] = colour[node->m01_col[q]];
    IndexVector order;
    group(entry_colour, number_of_colours, node->colour_ptr, order);
    node->colour_row = permuted(node->m01_row, order);
    node->colour_slot = permuted(node->m01_slot, order);

    IndexVector update_colour;
    for (size_t q = 0; q < node->m10_row.size(); q++) {
      const int64_t i = node->m10_col[q];
      for (int64_t j : columns_of[component[i]]) {
        update_row.push_back(node->m10_row[q]);
        update_col.push_back(j);
        update_a.push_back(node->m10_slot[q]);
        update_b.push_back(i);
        update_colour.push_back(colour[j]);
      }
    }
    for (size_t u = 0; u < update_row.size(); u++) s_entries.push_back({update_col[u], update_row[u]});
    group(update_colour, number_of_colours, node->update_ptr, order);
    update_row = permuted(update_row, order);
    update_col = permuted(update_col, order);
    node->update_m10 = permuted(update_a, order);
    node->update_state = permuted(update_b, order);
  } else if (node->schur_complement == BlockNodeData::DIAGONAL) {
    IndexVector m00_diagonal(m00.rows, -1);
    for_each_entry(m00, [&](int64_t r, int64_t col, int64_t slot) {
      if (r == col) m00_diagonal[r] = slot;
    });
    // M01 by row: (column, slot)
    std::vector<std::vector<std::pair<int64_t, int64_t>>> m01_rows(m01.rows);
    for_each_entry(m01, [&](int64_t r, int64_t col, int64_t slot) { m01_rows[r].push_back({col, slot}); });
    for (size_t q = 0; q < node->m10_row.size(); q++) {
      const int64_t i = node->m10_col[q];
      for (const auto &[j, slot01] : m01_rows[i]) {
        update_row.push_back(node->m10_row[q]);
        update_col.push_back(j);
        node->diagonal_m10.push_back(node->m10_slot[q]);
        node->diagonal_m01.push_back(slot01);
        node->diagonal_m00.push_back(m00_diagonal[i]);
        s_entries.push_back({j, node->m10_row[q]});
      }
    }
  }

  Block s = assemble(nc, s_entries);
  node->s_gather.assign(s.nnz(), -1);
  for_each_entry(m11, [&](int64_t r, int64_t col, int64_t slot) {
    node->s_gather[position(s, r, col)] = slot;
  });
  IndexVector &targets = node->schur_complement == BlockNodeData::EXACT
                             ? node->update_s
                             : node->diagonal_s;
  for (size_t u = 0; u < update_row.size(); u++) {
    targets.push_back(position(s, update_row[u], update_col[u]));
  }

  auto eliminated = lower(children[0].cast<py::dict>(), m00, child0_indices, local_of);
  auto complement = lower(children[1].cast<py::dict>(), s, child1_indices, local_of);
  bool exact = eliminated->exact && complement->exact;
  if (node->schur_complement == BlockNodeData::DIAGONAL) {
    bool diagonal_only = true;
    for_each_entry(m00, [&](int64_t r, int64_t col, int64_t) { diagonal_only &= r == col; });
    exact &= diagonal_only;
  } else if (node->schur_complement == BlockNodeData::NONE) {
    exact &= m01.nnz() == 0 || m10.nnz() == 0;
  }
  node->exact = exact;
  node->children = {eliminated, complement};
  return node;
}

std::shared_ptr<BlockNodeData> lower(const py::dict &spec, const Block &p,
                                      const IndexVector &node_indices,
                                      IndexVector &local_of) {
  const std::string type = read_string(spec, "type");
  if (type == "leaf") return lower_leaf(spec, p);

  const auto children = spec["children"].cast<py::list>();
  std::vector<IndexVector> child_indices, positions;
  for (size_t k = 0; k < node_indices.size(); k++) {
    local_of[node_indices[k]] = static_cast<int64_t>(k);
  }
  for (const auto &child : children) {
    child_indices.push_back(sorted_indices(child.cast<py::dict>()));
    IndexVector position;
    for (int64_t state : child_indices.back()) position.push_back(local_of[state]);
    positions.push_back(std::move(position));
  }

  if (type == "schur") {
    if (children.size() != 2) throw std::invalid_argument("a Schur node needs two children");
    return lower_schur(spec, p, positions[0], positions[1], children, child_indices[0],
                       child_indices[1], local_of);
  }
  if (type != "additive" && type != "multiplicative") {
    throw std::invalid_argument("unknown preconditioner tree node \"" + type + "\"");
  }
  auto node = std::make_shared<BlockNodeData>();
  node->type = type == "additive" ? BlockNodeData::ADDITIVE : BlockNodeData::MULTIPLICATIVE;
  node->size = p.rows;

  IndexVector label(p.rows, -1);
  for (size_t k = 0; k < positions.size(); k++) {
    for (int64_t i : positions[k]) label[i] = static_cast<int64_t>(k);
  }
  bool decoupled = true;
  IndexVector sweep_child;
  for_each_entry(p, [&](int64_t r, int64_t c, int64_t slot) {
    if (label[r] == label[c]) return;
    if (node->type == BlockNodeData::MULTIPLICATIVE && label[r] > label[c]) {
      // Entries from earlier children are applied before each child's solve
      sweep_child.push_back(label[r]);
      node->sweep_row.push_back(r);
      node->sweep_col.push_back(c);
      node->sweep_slot.push_back(slot);
    } else {
      decoupled = false;
    }
  });
  if (node->type == BlockNodeData::MULTIPLICATIVE) {
    IndexVector order;
    group(sweep_child, static_cast<int64_t>(positions.size()), node->sweep_ptr, order);
    node->sweep_row = permuted(node->sweep_row, order);
    node->sweep_col = permuted(node->sweep_col, order);
    node->sweep_slot = permuted(node->sweep_slot, order);
  }

  bool exact = decoupled;
  for (size_t k = 0; k < positions.size(); k++) {
    node->children.push_back(lower(children[k].cast<py::dict>(),
                                   extract(p, positions[k], positions[k]),
                                   child_indices[k], local_of));
    exact &= node->children.back()->exact;
  }
  node->positions = std::move(positions);
  node->exact = exact;
  return node;
}

// Positions a node reads in its source array. A Schur complement's child reads
// the complement's own array, which the Schur node gathers from the source.
void collect_reads(const BlockNodeData &node, IndexVector &out) {
  auto append = [&](const IndexVector &slots) {
    for (int64_t slot : slots) {
      if (slot >= 0) out.push_back(slot);
    }
  };
  switch (node.type) {
    case BlockNodeData::LEAF:
      append(node.gather);
      return;
    case BlockNodeData::ADDITIVE:
    case BlockNodeData::MULTIPLICATIVE:
      append(node.sweep_slot);
      for (const auto &child : node.children) collect_reads(*child, out);
      return;
    case BlockNodeData::SCHUR:
      append(node.m01_slot);
      append(node.m10_slot);
      append(node.s_gather);
      append(node.diagonal_m00);
      collect_reads(*node.children[0], out);
      return;
  }
}

}  // namespace

BlockAnalysis::BlockAnalysis(const py::dict &tree, const IndexVector &indptr,
                               const IndexVector &indices)
    : size(static_cast<int64_t>(indptr.size()) - 1) {
  Block p;
  p.rows = p.cols = size;
  p.indptr = indptr;
  p.indices = indices;
  p.slot.resize(indices.size());
  std::iota(p.slot.begin(), p.slot.end(), 0);
  for (int64_t c = 0; c < size; c++) {
    if (!std::is_sorted(indices.begin() + indptr[c], indices.begin() + indptr[c + 1])) {
      throw std::invalid_argument("the iteration matrix pattern must have sorted rows");
    }
  }
  const IndexVector states = sorted_indices(tree);
  if (static_cast<int64_t>(states.size()) != size ||
      (size > 0 && (states.front() != 0 || states.back() != size - 1))) {
    throw std::invalid_argument(
        "the preconditioner tree must cover every state exactly once");
  }
  IndexVector local_of(size, -1);
  root = lower(tree, p, states, local_of);
  collect_reads(*root, read_entries);
  std::sort(read_entries.begin(), read_entries.end());
  read_entries.erase(std::unique(read_entries.begin(), read_entries.end()),
                     read_entries.end());
}
