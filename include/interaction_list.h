#ifndef interaction_list
#define interaction_list
#include "exafmm_t.h"

namespace exafmm_t {
  //! return x + 10y + 100z + 555
  int coord_hash(int* c) {
    const int n=5;
    return ( (c[2]+n) * (2*n) + (c[1]+n) ) *(2*n) + (c[0]+n);
  }

  //! swap x,y,z so that |z|>|y|>|x|, return hash of new coord
  int class_hash(int* c_) {
    int c[3]= {abs(c_[0]), abs(c_[1]), abs(c_[2])};
    if(c[1]>c[0] && c[1]>c[2]) {
      int tmp=c[0];
      c[0]=c[1];
      c[1]=tmp;
    }
    if(c[0]>c[2]) {
      int tmp=c[0];
      c[0]=c[2];
      c[2]=tmp;
    }
    if(c[0]>c[1]) {
      int tmp=c[0];
      c[0]=c[1];
      c[1]=tmp;
    }
    assert(c[0]<=c[1] && c[1]<=c[2]);
    return coord_hash(&c[0]);
  }

  void InitList(int max_r, int min_r, int step, Mat_Type t) {
    const int max_hash = 2000;
    int n1 = (max_r*2)/step+1;
    int n2 = (min_r*2)/step-1;
    int count=n1*n1*n1-(min_r>0?n2*n2*n2:0);
    std::vector<ivec3>& M=rel_coord[t];
    M.resize(count);
    hash_lut[t].assign(max_hash, -1);
    std::vector<int> class_size_hash(max_hash, 0);
    for(int k=-max_r; k<=max_r; k+=step)
      for(int j=-max_r; j<=max_r; j+=step)
        for(int i=-max_r; i<=max_r; i+=step)
          if(abs(i)>=min_r || abs(j)>=min_r || abs(k) >= min_r) {
            int c[3]= {i, j, k};
            // count the number of coords of the same class
            // ex. (-1,-1,2) is in the same class as (-2,-1,1)
            class_size_hash[class_hash(c)]++;
          }
    // class count -> class count displacement
    std::vector<int> class_disp_hash(max_hash, 0);
    for(int i=1; i<max_hash; i++) {
      class_disp_hash[i] = class_disp_hash[i-1] + class_size_hash[i-1];
    }

    int count_=0;
    for(int k=-max_r; k<=max_r; k+=step)
      for(int j=-max_r; j<=max_r; j+=step)
        for(int i=-max_r; i<=max_r; i+=step)
          if(abs(i)>=min_r || abs(j)>=min_r || abs(k) >= min_r) {
            int c[3]= {i, j, k};
            int& idx=class_disp_hash[class_hash(c)]; // idx is the displ of current class
            for(int l=0; l<3; l++) M[idx][l]=c[l]; // store the sorted coords
            hash_lut[t][coord_hash(c)]=idx;          // store mapping: hash -> index in rel_coord
            count_++;
            idx++;
          }
    assert(count_==count);
  }

  void InitAll() {
    rel_coord.resize(Type_Count);
    hash_lut.resize(Type_Count);
    InitList(0, 0, 1, M2M_V_Type);
    InitList(0, 0, 1, M2M_U_Type);
    InitList(0, 0, 1, L2L_V_Type);
    InitList(0, 0, 1, L2L_U_Type);
    InitList(1, 1, 2, M2M_Type); // count = 8, (+1 or -1)
    InitList(1, 1, 2, L2L_Type);
    InitList(3, 3, 2, P2P0_Type);  // count = 4^3-2^3 = 56
    InitList(1, 0, 1, P2P1_Type);
    InitList(3, 3, 2, P2P2_Type);
    InitList(3, 2, 1, M2L_Helper_Type);
    InitList(1, 1, 1, M2L_Type);
    InitList(5, 5, 2, M2P_Type);
    InitList(5, 5, 2, P2L_Type);
  }

  // Build t-type interaction list for node n
  void BuildList(Node* n, Mat_Type t) {
    const int n_child=8, n_collg=27;
    int c_hash, idx, rel_coord[3];
    int p2n = n->octant;       // octant
    Node* p = n->parent; // parent node
    std::vector<Node*>& interac_list = n->interac_list[t];
    switch (t) {
    case P2P0_Type:
      if(p == NULL || !n->IsLeaf()) return;
      for(int i=0; i<n_collg; i++) {
        Node* pc = p->colleague[i];
        if(pc!=NULL && pc->IsLeaf()) {
          rel_coord[0]=( i %3)*4-4-(p2n & 1?2:0)+1;
          rel_coord[1]=((i/3)%3)*4-4-(p2n & 2?2:0)+1;
          rel_coord[2]=((i/9)%3)*4-4-(p2n & 4?2:0)+1;
          c_hash = coord_hash(rel_coord);
          idx = hash_lut[t][c_hash];
          if(idx>=0) interac_list[idx] = pc;
        }
      }
      break;
    case P2P1_Type:
      if(!n->IsLeaf()) return;
      for(int i=0; i<n_collg; i++) {
        Node* col=(Node*)n->colleague[i];
        if(col!=NULL && col->IsLeaf()) {
          rel_coord[0]=( i %3)-1;
          rel_coord[1]=((i/3)%3)-1;
          rel_coord[2]=((i/9)%3)-1;
          c_hash = coord_hash(rel_coord);
          idx = hash_lut[t][c_hash];
          if(idx>=0) interac_list[idx] = col;
        }
      }
      break;
    case P2P2_Type:
      if(!n->IsLeaf()) return;
      for(int i=0; i<n_collg; i++) {
        Node* col=(Node*)n->colleague[i];
        if(col!=NULL && !col->IsLeaf()) {
          for(int j=0; j<n_child; j++) {
            rel_coord[0]=( i %3)*4-4+(j & 1?2:0)-1;
            rel_coord[1]=((i/3)%3)*4-4+(j & 2?2:0)-1;
            rel_coord[2]=((i/9)%3)*4-4+(j & 4?2:0)-1;
            c_hash = coord_hash(rel_coord);
            idx = hash_lut[t][c_hash];
            if(idx>=0) {
              assert(col->Child(j)->IsLeaf()); //2:1 balanced
              interac_list[idx] = (Node*)col->Child(j);
            }
          }
        }
      }
      break;
    case M2L_Type:
      if(n->IsLeaf()) return;
      for(int i=0; i<n_collg; i++) {
        Node* col=(Node*)n->colleague[i];
        if(col!=NULL && !col->IsLeaf()) {
          rel_coord[0]=( i %3)-1;
          rel_coord[1]=((i/3)%3)-1;
          rel_coord[2]=((i/9)%3)-1;
          c_hash = coord_hash(rel_coord);
          idx=hash_lut[t][c_hash];
          if(idx>=0) interac_list[idx]=col;
        }
      }
      break;
    case M2P_Type:
      if(!n->IsLeaf()) return;
      for(int i=0; i<n_collg; i++) {
        Node* col=(Node*)n->colleague[i];
        if(col!=NULL && !col->IsLeaf()) {
          for(int j=0; j<n_child; j++) {
            rel_coord[0]=( i %3)*4-4+(j & 1?2:0)-1;
            rel_coord[1]=((i/3)%3)*4-4+(j & 2?2:0)-1;
            rel_coord[2]=((i/9)%3)*4-4+(j & 4?2:0)-1;
            c_hash = coord_hash(rel_coord);
            idx=hash_lut[t][c_hash];
            if(idx>=0) interac_list[idx]=(Node*)col->Child(j);
          }
        }
      }
      break;
    case P2L_Type:
      if(p == NULL) return;
      for(int i=0; i<n_collg; i++) {
        Node* pc=(Node*)p->colleague[i];
        if(pc!=NULL && pc->IsLeaf()) {
          rel_coord[0]=( i %3)*4-4-(p2n & 1?2:0)+1;
          rel_coord[1]=((i/3)%3)*4-4-(p2n & 2?2:0)+1;
          rel_coord[2]=((i/9)%3)*4-4-(p2n & 4?2:0)+1;
          c_hash = coord_hash(rel_coord);
          idx=hash_lut[t][c_hash];
          if(idx>=0) interac_list[idx]=pc;
        }
      }
      break;
    default:
      abort();
    }
  }

  // Fill in interac_list of all nodes, assume sources == target for simplicity
  void BuildInteracLists(Nodes& nodes) {
    std::vector<Mat_Type> interactionTypes = {P2P0_Type, P2P1_Type, P2P2_Type,
                                              M2P_Type, P2L_Type, M2L_Type};
    for(int j=0; j<interactionTypes.size(); j++) {
      Mat_Type type = interactionTypes[j];
      int numRelCoord = rel_coord[type].size();  // num of possible relative positions
      #pragma omp parallel for
      for(size_t i=0; i<nodes.size(); i++) {
        Node* node = &nodes[i];
        node->interac_list[type].resize(numRelCoord, 0);
        BuildList(node, type);
      }
    }
  }

  void SetColleagues(Node* node=NULL) {
    Node* parent_node;
    Node* tmp_node1;
    Node* tmp_node2;
    for(int i=0; i<27; i++) node->colleague[i] = NULL;
    parent_node = node->parent;
    if(parent_node==NULL) return;
    int l=node->octant;         // l is octant
    for(int i=0; i<27; i++) {
      tmp_node1 = parent_node->colleague[i];  // loop over parent's colleagues
      if(tmp_node1!=NULL && !tmp_node1->IsLeaf()) {
        for(int j=0; j<8; j++) {
          tmp_node2=tmp_node1->Child(j);    // loop over parent's colleages child
          if(tmp_node2!=NULL) {
            bool flag=true;
            int a=1, b=1, new_indx=0;
            for(int k=0; k<3; k++) {
              int indx_diff=(((i/b)%3)-1)*2+((j/a)%2)-((l/a)%2);
              if(-1>indx_diff || indx_diff>1) flag=false;
              new_indx+=(indx_diff+1)*b;
              a*=2;
              b*=3;
            }
            if(flag)
              node->colleague[new_indx] = tmp_node2;
          }
        }
      }
    }
  }

  void SetColleagues(Nodes& nodes) {
    nodes[0].colleague[13] = &nodes[0];
    for(int i=1; i<nodes.size(); i++) {
      SetColleagues(&nodes[i]);
    }
  }
}
#endif
