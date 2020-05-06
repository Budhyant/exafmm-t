#ifndef local_essential_tree_h
#define local_essential_tree_h
#include "alltoall.h"
#include "hilbert.h"
#include <map>
#include "timer.h"
#define SEND_ALL 0 //! Set to 1 for debugging

namespace exafmm_t {
  template <typename T> using BodyMap = std::multimap<uint64_t, Body<T>>;
  template <typename T> using NodeMap = std::map<uint64_t, Node<T>>;
  // int LEVEL;                                    //!< Octree level used for partitioning
  // std::vector<int> OFFSET;                      //!< Offset of Hilbert index for partitions

  //! Distance between cell center and edge of a remote domain
  template <typename T>
  real_t getDistance(Node<T>* C, int irank, std::vector<int>& OFFSET, int LEVEL, vec3 X0, real_t R0) {
    real_t distance = R0;
    real_t R = R0 / (1 << LEVEL);
    for (int key=OFFSET[irank]; key<OFFSET[irank+1]; key++) {
      ivec3 iX = get3DIndex(key, LEVEL);
      vec3 X = getCoordinates(iX, LEVEL, X0, R0);
      vec3 Xmin = X - R;
      vec3 Xmax = X + R;
      vec3 dX;
			for (int d=0; d<3; d++) {
				dX[d] = (C->x[d] > Xmax[d]) * (C->x[d] - Xmax[d]) + (C->x[d] < Xmin[d]) * (C->x[d] - Xmin[d]);
			}
			distance = std::min(distance, norm(dX));
    }
    return distance;
  }

  //! Recursive call to pre-order tree traversal for selecting cells to send
  template <typename T>
  void selectCells(Node<T>* Cj, int irank, Bodies<T>& bodyBuffer, std::vector<int> & sendBodyCount,
                   Nodes<T>& cellBuffer, std::vector<int> & sendCellCount,
                   std::vector<int>& OFFSET, int LEVEL, vec3 X0, real_t R0) {
    real_t R = getDistance(Cj, irank, OFFSET, LEVEL, X0, R0);
    real_t THETA = 0.5;
    real_t R2 = R * R * THETA * THETA;
    sendCellCount[irank]++;
    cellBuffer.push_back(*Cj);

    if (R2 <= (Cj->r + Cj->r) * (Cj->r + Cj->r)) {   // if near range
      if (Cj->is_leaf) {
        sendBodyCount[irank] += Cj->nsrcs;
        for (int b=0; b<Cj->nsrcs; b++) {
          bodyBuffer.push_back(Cj->first_src[b]);
        }
      } else {
        for (auto & child : Cj->children) {
          selectCells(child, irank, bodyBuffer, sendBodyCount, cellBuffer, sendCellCount,
                      OFFSET, LEVEL, X0, R0);
        }
      }
    }
  }

  template <typename T>
  void whatToSend(Nodes<T> & cells, Bodies<T> & bodyBuffer, std::vector<int> & sendBodyCount,
                  Nodes<T> & cellBuffer, std::vector<int> & sendCellCount,
                  std::vector<int>& OFFSET, int LEVEL, vec3 X0, real_t R0) {
#if SEND_ALL //! Send everything (for debugging)
    for (int irank=0; irank<MPISIZE; irank++) {
      sendCellCount[irank] = cells.size();
      for (size_t i=0; i<cells.size(); i++) {
        if (cells[i].is_leaf) {
          sendBodyCount[irank] += cells[i].nsrcs;
          for (int b=0; b<cells[i].nsrcs; b++) {
            bodyBuffer.push_back(cells[i].first_src[b]);
          }
        }
      }
      cellBuffer.insert(cellBuffer.end(), cells.begin(), cells.end());
    }
#else //! Send only necessary cells
    for (int irank=0; irank<MPISIZE; irank++) {
      selectCells(&cells[0], irank, bodyBuffer, sendBodyCount, cellBuffer, sendCellCount,
                  OFFSET, LEVEL, X0, R0);
    }
#endif
  }

  //! Reapply Ncrit recursively to account for bodies from other ranks
  template <typename T>
  void reapplyNcrit(BodyMap<T> & bodyMap, NodeMap<T> & nodeMap, uint64_t key,
                    int ncrit, int nsurf, vec3 x0, real_t r0) {
    // recursion exit condition: key is leaf
    bool noChildSent = true;
    for (int i=0; i<8; i++) {
      uint64_t childKey = getChild(key) + i;
      if (nodeMap.find(childKey) != nodeMap.end()) noChildSent = false;
    }
    if (nodeMap[key].nsrcs <= ncrit && noChildSent) {
      // nodeMap[key].numChilds = 0;
      nodeMap[key].is_leaf = true;
      nodeMap[key].nsrcs = bodyMap.count(key);
      return;
    }

    // Assign key of child to bodyMap
    nodeMap[key].is_leaf = false;
    int level = getLevel(key);
    std::vector<int> counter(8, 0);
    // std::pair<BodyMap::iterator,BodyMap::iterator> range = bodyMap.equal_range(key);
    auto range = bodyMap.equal_range(key);
    Bodies<T> bodies(bodyMap.count(key));
    size_t b = 0;
    // for (BodyMap<T>::iterator B=range.first; B!=range.second; B++, b++) {
    for (auto B=range.first; B!=range.second; B++, b++) {
      bodies[b] = B->second;
    }
    for (b=0; b<bodies.size(); b++) {
      ivec3 iX = get3DIndex(bodies[b].X, level+1, x0, r0);
      uint64_t childKey = getKey(iX, level+1);
      int octant = getOctant(childKey);
      counter[octant]++;
      bodies[b].key = childKey;
      bodyMap.insert(std::pair<uint64_t, Body<T>>(childKey, bodies[b]));
    }
    if (bodyMap.count(key) != 0) bodyMap.erase(key);

    //! Create a new cell if it didn't exist
    for (int i=0; i<8; i++) {
      uint64_t childKey = getChild(key) + i;
    //  if (counter[i] != 0) {
      if (nodeMap.find(childKey) == nodeMap.end()) {
        Node<T> node;
        node.nsrcs = counter[i];
        // node.numChilds = 0;
        ivec3 iX = get3DIndex(childKey);
        node.x = getCoordinates(iX, level+1, x0, r0);
        node.r = r0 / (1 << (level+1));
        node.key = childKey;
        node.up_equiv.resize(nsurf, 0.0);
        // node.L.resize(nsurf, 0.0);
        nodeMap[childKey] = node;
      } else {
        nodeMap[childKey].nsrcs += counter[i];
        nodeMap[childKey].up_equiv.resize(nsurf, 0);
      }
    //}
      if (nodeMap.find(childKey) != nodeMap.end()) {
        reapplyNcrit(bodyMap, nodeMap, childKey, ncrit, nsurf, x0, r0);
      }
    }
    //! Update number of bodies and child cells
    int nsrcs = 0;
    // int numChilds = 0;
    for (int i=0; i<8; i++) {
      uint64_t childKey = getChild(key) + i;
      if (nodeMap.find(childKey) != nodeMap.end()) {
        nsrcs += nodeMap[childKey].nsrcs;
        // numChilds++;
      }
    }
    // if (numChilds == 0) numBodies = bodyMap.count(key);
    nodeMap[key].nsrcs = nsrcs;
    //nodeMap[key].numChilds = numChilds;
  }

  //! Check integrity of local essential tree
/*
  void sanityCheck(BodyMap & bodyMap, NodeMap & cellMap, uint64_t key) {
    Cell cell = cellMap[key];
    assert(cell.key == key);
    if (cell.numChilds == 0) assert(cell.numBodies == int(bodyMap.count(key)));
    if (bodyMap.count(key) != 0) {
      assert(cell.numChilds == 0);
      std::pair<BodyMap::iterator,BodyMap::iterator> range = bodyMap.equal_range(key);
      for (BodyMap::iterator B=range.first; B!=range.second; B++) {
        assert(B->second.key == key);
      }
    }
    int numBodies = 0;
    int numChilds = 0;
    for (int i=0; i<8; i++) {
      uint64_t childKey = getChild(key) + i;
      if (cellMap.find(childKey) != cellMap.end()) {
        sanityCheck(bodyMap, cellMap, childKey);
        numBodies += cellMap[childKey].numBodies;
        numChilds++;
      }
    }
    assert((cell.numBodies == numBodies) || (numBodies == 0));
    assert((cell.numChilds == numChilds));
  }
*/

  //! Build cells of LET recursively
/*
  template <typename T>
  void buildCells(BodyMap<T> & bodyMap, NodeMap<T> & cellMap, uint64_t key, Bodies<T> & bodies, Node<T> * cell, Nodes<T> & cells) {
    *cell = cellMap[key];
    if (bodyMap.count(key) != 0) {
      std::pair<BodyMap::iterator,BodyMap::iterator> range = bodyMap.equal_range(key);
      bodies.resize(bodies.size()+cell->numBodies);
      Body * body = &bodies.back() - cell->numBodies + 1;
      cell->body = body;
      int b = 0;
      for (BodyMap::iterator B=range.first; B!=range.second; B++, b++) {
        body[b] = B->second;
      }
    } else {
      cell->body = NULL;
    }
    if (cell->numChilds != 0) {
      cells.resize(cells.size()+cell->numChilds);
      Cell * child = &cells.back() - cell->numChilds + 1;
      cell->child = child;
      for (int i=0, c=0; i<8; i++) {
        uint64_t childKey = getChild(key) + i;
        if (cellMap.find(childKey) != cellMap.end()) {
          buildCells(bodyMap, cellMap, childKey, bodies, &child[c++], cells);
        }
      }
    } else {
      cell->child = NULL;
    }
    if (cell->numChilds != 0) cell->body = cell->child->body;
  }
*/
  //! Build local essential tree
  template <typename T>
  void buildLocalEssentialTree(Bodies<T> & recvBodies, Nodes<T> & recvCells,
                               Bodies<T> & bodies, Nodes<T> & nodes,
                               int ncrit, int nsurf, vec3 x0, real_t r0) {
    BodyMap<T> bodyMap;
    NodeMap<T> nodeMap;
    //! Insert bodies to multimap
    for (size_t i=0; i<recvBodies.size(); i++) {
      bodyMap.insert(std::pair<uint64_t, Body<T>>(recvBodies[i].key, recvBodies[i]));
    }
    //! Insert cells to map and merge cells
    for (size_t i=0; i<recvCells.size(); i++) {
      uint64_t key = recvCells[i].key;    // key with level offset
      if (nodeMap.find(key) == nodeMap.end()) {
        nodeMap[key] = recvCells[i];
      } else {
        for (int n=0; n<nsurf; n++) {
          nodeMap[key].up_equiv[n] += recvCells[i].up_equiv[n];
        }
        nodeMap[key].nsrcs += recvCells[i].nsrcs;
      }
    }

    //! Reapply Ncrit recursively to account for bodies from other ranks
    reapplyNcrit(bodyMap, nodeMap, 0, ncrit, nsurf, x0, r0);

#if 1
    std::cout << "rank " << MPIRANK
              << " , node map size: " << nodeMap.size() 
              << " , root nsrcs: " << nodeMap[0].nsrcs << std::endl;
#endif
/*
    //! Check integrity of local essential tree
//    sanityCheck(bodyMap, nodeMap, 0);
    //! Copy bodyMap to bodies
    bodies.clear();
    bodies.reserve(bodyMap.size());
    //! Build cells of LET recursively
    nodes.reserve(nodeMap.size());
    nodes.resize(1);
    buildCells(bodyMap, nodeMap, 0, bodies, &nodes[0], nodes);
    //! Check correspondence between vector and map sizes
    assert(bodies.size() == bodyMap.size());
    assert(nodes.size() == nodeMap.size());
*/
  }

  //! MPI communication for local essential tree
  template <typename T>
  void localEssentialTree(Bodies<T> & bodies, Nodes<T> & cells,
                  std::vector<int>& OFFSET, int LEVEL, int ncrit) {
    vec3 x0 = cells[0].x;
    real_t r0 = cells[0].r;
    int nsurf = cells[0].up_equiv.size();
    std::vector<int> sendBodyCount(MPISIZE, 0);
    std::vector<int> recvBodyCount(MPISIZE, 0);
    std::vector<int> sendBodyDispl(MPISIZE, 0);
    std::vector<int> recvBodyDispl(MPISIZE, 0);
    std::vector<int> sendCellCount(MPISIZE, 0);
    std::vector<int> recvCellCount(MPISIZE, 0);
    std::vector<int> sendCellDispl(MPISIZE, 0);
    std::vector<int> recvCellDispl(MPISIZE, 0);
    Bodies<T> sendBodies, recvBodies;
    Nodes<T> sendCells, recvCells;
    //! Decide which cells & bodies to send
    whatToSend(cells, sendBodies, sendBodyCount, sendCells, sendCellCount,
               OFFSET, LEVEL, x0, r0);
    //! Use alltoall to get recv count and calculate displacement (defined in alltoall.h)
    getCountAndDispl(sendBodyCount, sendBodyDispl, recvBodyCount, recvBodyDispl);
    getCountAndDispl(sendCellCount, sendCellDispl, recvCellCount, recvCellDispl);
    //! Alltoallv for cells (defined in alltoall.h)
    alltoallCells(sendCells, sendCellCount, sendCellDispl, recvCells, recvCellCount, recvCellDispl);
    //! Alltoallv for sources (defined in alltoall.h)
    alltoallBodies(sendBodies, sendBodyCount, sendBodyDispl, recvBodies, recvBodyCount, recvBodyDispl);
#if 0
    if (MPIRANK==1) {
      for (auto count : sendBodyCount)
        std::cout << count << std::endl;
    }
#endif

    //! Build local essential tree
    buildLocalEssentialTree(recvBodies, recvCells, bodies, cells, ncrit, nsurf, x0, r0);

  }
}
#endif
