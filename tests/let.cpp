#include "build_tree.h"
#include "dataset.h"
#include "exafmm_t.h"
#include "local_essential_tree.h"
#include "partition.h"
#include "test.h"

using namespace exafmm_t;

int main(int argc, char** argv) {
  Args args(argc, argv);
  startMPI(argc, argv);

  int n = 10000;
  Bodies<real_t> sources = init_sources<real_t>(n, args.distribution, MPIRANK);
  Bodies<real_t> targets = init_targets<real_t>(n, args.distribution, MPIRANK+10);

  // partition
  vec3 x0;
  real_t r0;
  allreduceBounds(sources, targets, x0, r0);
  std::vector<int> offset;   // based on the distribution of sources
  partition(sources, targets, x0, r0, offset, args.maxlevel);
  // writeBodies(targets);

  // build tree
  DummyFmm<real_t> fmm(args.ncrit);
  fmm.x0 = x0;
  fmm.r0 = r0;
  NodePtrs<real_t> leafs, nonleafs;  
  Nodes<real_t> nodes = build_tree(sources, targets, leafs, nonleafs, fmm);
  // writeNodes(nodes);

  // upward pass
  Node<real_t>* root = nodes.data();
  fmm.P2M(leafs);
  fmm.M2M(root);
  print("root's monopole", root->up_equiv[0]);

  // let
  localEssentialTree(sources, nodes, offset, args.maxlevel, args.ncrit);

  std::unordered_map<int, int> leaf_count;
  for (auto& node : nodes) {
    if (node.is_leaf) {
      int level = node.level;
      if (leaf_count.find(level) == leaf_count.end()) leaf_count[level] = 0;
      leaf_count[level]++;
    }
  }

  if (MPIRANK == 0)
  for (auto& it : leaf_count) {
    std::cout << it.first << " " << it.second << std::endl;
  }

  // start tree balancing
  std::unordered_map<uint64_t, size_t> key2id;
  Keys keys = breadth_first_traversal(&nodes[0], key2id);
  Keys balanced_keys = balance_tree(keys, key2id, nodes);
  Keys leaf_keys = find_leaf_keys(balanced_keys);
#if 10
  if (MPIRANK == 0) {
    int level = 0;
    for (auto& keys : leaf_keys) {
      std::cout << level++ << " " << keys.size() << std::endl;
    }
  }
#endif
#if 0
  if (MPIRANK == 0) {
    for (auto& src : sources) {
      std::cout << src.ibody << " " << src.key << " "
                << src.X[0] << " " << src.X[1] << " " << src.X[2] << std::endl;
    }
  }
#endif
  nodes.clear();
  leafs.clear();
  nonleafs.clear();
  nodes = build_tree(sources, targets, leafs, nonleafs, fmm, leaf_keys);
  fmm.depth = keys.size() - 1;

  stopMPI();
}
