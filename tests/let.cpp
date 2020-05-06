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
  writeBodies(targets);

  // build tree
  DummyFmm<real_t> fmm(args.ncrit);
  fmm.x0 = x0;
  fmm.r0 = r0;
  NodePtrs<real_t> leafs, nonleafs;  
  Nodes<real_t> nodes = build_tree(sources, targets, leafs, nonleafs, fmm);
  writeNodes(nodes);

  // upward pass
  Node<real_t>* root = nodes.data();
  fmm.P2M(leafs);
  fmm.M2M(root);
  print("root's monopole", root->up_equiv[0]);

  // let
  localEssentialTree(sources, nodes, offset, args.maxlevel, args.ncrit);
  stopMPI();
}
