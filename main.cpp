#include "build_tree.h"
#include "dataset.h"
#include "interaction_list.h"
#if HELMHOLTZ
#include "helmholtz.h"
#include "precompute_helmholtz.h"
#elif COMPLEX
#include "laplace_c.h"
#include "precompute_c.h"
#else
#include "laplace.h"
#include "precompute.h"
#endif
#include "traverse.h"

using namespace exafmm_t;
using namespace std;
int main(int argc, char **argv) {
#if HELMHOLTZ
  MU = 20;
#endif
  Args args(argc, argv);
  omp_set_num_threads(args.threads);
  size_t N = args.numBodies;
  P = args.P;
  NSURF = 6*(P-1)*(P-1) + 2;
  Profile::Enable(true);

  Profile::Tic("Total");
  Bodies sources = init_bodies(args.numBodies, args.distribution, 0);
  Bodies targets = init_bodies(args.numBodies, args.distribution, 0);

  Profile::Tic("Build Tree");
  get_bounds(sources, targets, XMIN0, R0);
  NodePtrs leafs, nonleafs;
  Nodes nodes = build_tree(sources, targets, XMIN0, R0, leafs, nonleafs, args);
  balance_tree(nodes, sources, targets, XMIN0, R0, leafs, nonleafs, args);
  Profile::Toc();

  init_rel_coord();
  Profile::Tic("Precomputation");
  precompute();
  Profile::Toc();
  Profile::Tic("Build Lists");
  set_colleagues(nodes);
  build_list(nodes);
  Profile::Toc();
  M2L_setup(nonleafs);
  upward_pass(nodes, leafs);
  downward_pass(nodes, leafs);
  Profile::Toc();

  RealVec error = verify(leafs);
  std::cout << std::setw(20) << std::left << "Leaf Nodes" << " : "<< leafs.size() << std::endl;
  std::cout << std::setw(20) << std::left << "Tree Depth" << " : "<< MAXLEVEL << std::endl;
  std::cout << std::setw(20) << std::left << "Potn Error" << " : " << std::scientific << error[0] << std::endl;
  std::cout << std::setw(20) << std::left << "Grad Error" << " : " << std::scientific << error[1] << std::endl;
  Profile::print();
  return 0;
}
