#if NON_ADAPTIVE
#include "build_non_adaptive_tree.h"
#else
#include "build_tree.h"
#endif
#include "build_list.h"
#include "config.h"
#include "dataset.h"
#include "laplace.h"

using namespace exafmm_t;

int main(int argc, char **argv) {
  Args args(argc, argv);
  print_divider("Parameters");
  args.print();

#if HAVE_OPENMP
  omp_set_num_threads(args.threads);
#endif

  print_divider("Time");
  start("Total");
  Bodies<real_t> sources = init_sources<real_t>(args.numBodies, args.distribution, 0);
  Bodies<real_t> targets = init_targets<real_t>(args.numBodies, args.distribution, 5);

  LaplaceFMM fmm(args.P, args.ncrit, args.maxlevel);

  start("Build Tree");
  get_bounds(sources, targets, fmm.x0, fmm.r0);
  NodePtrs<real_t> leafs, nonleafs;
#if NON_ADAPTIVE
  Nodes<real_t> nodes = build_tree(sources, targets, leafs, nonleafs, fmm);
#else
  Nodes<real_t> nodes = build_tree(sources, targets, leafs, nonleafs, fmm);
  balance_tree(nodes, sources, targets, leafs, nonleafs, fmm);
#endif
  stop("Build Tree");

  init_rel_coord();

  start("Build Lists");
  set_colleagues(nodes);
  build_list(nodes, fmm);
  stop("Build Lists");

  start("Precomputation");
  fmm.precompute();
  stop("Precomputation");

  fmm.M2L_setup(nonleafs);
  fmm.upward_pass(nodes, leafs);
  fmm.downward_pass(nodes, leafs);

#if DEBUG /* check downward check potential at leaf level*/
  for (auto dn_check : leafs[0]->dn_equiv) {
    std::cout << dn_check << std::endl;
  }
#endif

  stop("Total");

  RealVec error = fmm.verify(leafs);
  print_divider("Error");
  print("Potential Error", error[0]);
  print("Gradient Error", error[1]);

  print_divider("Tree");
  print("Root Center x", fmm.x0[0]);
  print("Root Center y", fmm.x0[1]);
  print("Root Center z", fmm.x0[2]);
  print("Root Radius R", fmm.r0);
  print("Tree Depth", fmm.depth);
  print("Leaf Nodes", leafs.size());

  return 0;
}
