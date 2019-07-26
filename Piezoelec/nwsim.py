

# coding: utf-8

# In[ ]:

import numpy as np

import random as rd

from numba import jit, autojit, double, float64, float32, void, int32

from scipy.sparse.linalg import spsolve

import networkx as nx

from scipy.sparse import*


from scipy.spatial import KDTree, cKDTree

class nanowire:

    def __init__(self, begin_pos,end_pos, rho,length,width):

        self.begin_pos = begin_pos

        self.end_pos = end_pos

        self.rho = rho

        self.length = length

        self.width = width

        

 

class Sample(object):

   
    

    def __init__(self,x,y,z,n_nanowires,poisson_ratio):

        self.x = x

        self.y = y

        self.z = z
        self.eps=0
        self.n_nanowires = n_nanowires

        self.poisson_ratio = poisson_ratio

        self.nanowire_list = []

        self.pos_list = np.zeros( (self.n_nanowires,6) )

    @jit

    def generate_network(self,rhos,lengths,widths,theta_range):

        

        for i in range(self.n_nanowires):

            if np.size(self.n_nanowires) > 1:

                r = rhos[i]

                w = widths[i]

                l = lengths[i]

            else:

                r = rhos

                w = widths

                l = lengths

            A = np.zeros( 3 )

            B = np.zeros( 3 )

            A[0] = self.x*rd.random()-self.x/2.
            A[1] = self.y*rd.random()-self.y/2.
            A[2] = self.z*rd.random()-self.z/2.

            theta = 2*theta_range*rd.random()-theta_range
            phi = 2*np.pi*rd.random()

            B[0] = A[0]+l*np.cos(theta)*np.cos(phi)
            B[1] = A[1]+l*np.cos(theta)*np.sin(phi)
            B[2] = A[2]+l*np.sin(theta)
            
            #here wires are oriented such that [x0,y0,z0,x1,y1,z1].
            #mostly z=0, but I wanted something general.
            #below, the wires are oriented such, that x0<x1 (so always left to right)
            if np.argmin( (A[0],B[0]) )==0: # which end is smaller, to orient all wires the same way.


                self.pos_list[i,0] = A[0]
                self.pos_list[i,1] = A[1]
                self.pos_list[i,2] = A[2]
                self.pos_list[i,3] = B[0]
                self.pos_list[i,4] = B[1]
                self.pos_list[i,5] = B[2]
                self.nanowire_list.append(nanowire(A,B,r,l,w))
            else:
                self.pos_list[i,3] = A[0]
                self.pos_list[i,4] = A[1]
                self.pos_list[i,5] = A[2]
                self.pos_list[i,0] = B[0]
                self.pos_list[i,1] = B[1]
                self.pos_list[i,2] = B[2]
                self.nanowire_list.append(nanowire(B,A,r,l,w))
        #generate the kd_tree
        wire_centers=0.5*(self.pos_list[:,[0,1,2]]+self.pos_list[:,[3,4,5]])
        self.ckd=cKDTree(wire_centers)
        self.pairz=self.ckd.query_pairs(r=np.max(lengths),output_type='ndarray')
    
    
    def calculate_resistance(self,c_res,is3d=False):
        lin_res=self.nanowire_list[0].rho*self.nanowire_list[0].length/( (self.nanowire_list[0].width/2.)**2*np.pi)
        x_l = -(self.x/2. *(1-self.electrode_width))
        x_r = -x_l

        wire_array = np.vstack((self.pos_list,np.array([x_l+self.x/1e6,-self.y/2.,0,x_l,self.y/2.,0]),np.array([x_r,-self.y/2.,0,x_r,self.y/2.,0]))) #we add the two rtrodes as not completely vertical lines (if they are fully vertical, they can cause issues in the intersection calculation (because they never intersect and the calculation creates a 1/0)

        #in the calculation of 'contacting pairs', the electrodes are missing. Here we check which wires could be eligible to contact them.
        left_w=np.where((np.min(wire_array[:,[0,3]],axis=1)<x_l)&(np.max(wire_array[:,[0,3]],axis=1)>x_l))[0]
        left_w=np.hstack( (self.n_nanowires*np.ones( (len(left_w),1)),left_w.reshape(len(left_w),1)) )

        right_w=np.where((np.min(wire_array[:,[0,3]],axis=1)<x_r)&(np.max(wire_array[:,[0,3]],axis=1)>x_r))[0] #wires aren't ordered left right
        right_w=np.hstack( ((self.n_nanowires+1)*np.ones( (len(right_w),1)),right_w.reshape(len(right_w),1)) )

        self.pairz=np.vstack( (self.pairz,left_w,right_w) ).astype(int)

        pairs_cuts_dist=np.asarray(compute_crossings(wire_array,self.pairz,is_electrode=True)) #computes crossings of all the wires, but self.pairz contains only the relevant pairs to look at (kDtree)
        #pairs_cuts_dist = wire_i, wire_j, s, t, distance, where s is the cut position on i, t the cut position on j, s and t in [0,1] along its length. Distance is only relevant for 3D.
        if is3d:
            pairs_cuts_dist=pairs_cuts_dist[(pairs_cuts_dist[:,0]>self.n_nanowires-1) | (pairs_cuts_dist[:,-1]<1.15*(self.nanowire_list[0].width*(1+self.eps)**(-self.poisson_ratio))**2),:]
        print(len(pairs_cuts_dist))
        G=nx.Graph()
        G.add_edges_from(map(tuple,pairs_cuts_dist[:,[0,1]])) #creates a graph of all the pairs that are connected.
        number_to_remove=1 #just a counter
        #here we clean all the dangling wires, which do not contribute to conductivity.
        while number_to_remove>0:
            k=np.asarray(dict(G.degree()).keys()) 
            val=np.asarray(dict(G.degree()).values())
            to_rem = k[np.where(val==1)[0]] #all the wires that have only 1 connecting partner are dangling wires, they can be deleted. no current will flow through them.
            G.remove_nodes_from( to_rem )
            number_to_remove=len(to_rem)

        
        
        #The graph checks all the connected wire clusters here.
        store=list(nx.connected_components(G))
        
        #check for percolation: it only percolates if there exists a cluster which contains both electrodes.
        truth_perco=[self.n_nanowires in el and self.n_nanowires+1 in el for el in store]
        if True in truth_perco:
            percolating_indices = list(store[np.where(np.asarray(truth_perco)==True)[0][0]])
            self.percolating_indices=percolating_indices #the only wires we care about are the percolating ones. Other disconnected clusters do not contribute.
        else:
            raise RuntimeError('No Percolation')

        self.wire_cluster=percolating_indices

       
        a=np.in1d(pairs_cuts_dist[:,0],percolating_indices) #boolean array
        b=np.in1d(pairs_cuts_dist[:,1],percolating_indices) #boolean array
        keep_percolated=a&b                                 #all the entries that contain wires that participate to the percolation.    
        pairs_cuts_dist=pairs_cuts_dist[keep_percolated,:]
        #At this point, we kept all the intersecting i,j wire pairs, and where they cut each other,
        #but only within the percolation cluster. Now we can start the "heavy-lifting".

        #Because of effiency, the pairs are calculated for i,j, j>i. But the system is symmetric
        #So for every wire i contacting wire j, there's a potential node on wire i at the cut position
        #but equally, for every wire j containg wire i, there's a potential node on wire j.
        overall=np.vstack( (pairs_cuts_dist,pairs_cuts_dist[:,[1,0,3,2,4]]))  #I just stack the list where I invert the positions of i,j, and s,t, to retrieve symmetry.
        print('overall'+str(len(overall)))
        total_nodes_in_principle=len(overall) #for every contact, there's a potential node 'u'.
        on_left_el = np.where(overall[:,0]==self.n_nanowires)[0] #all the wires containg the left electrode
        on_right_el = np.where(overall[:,0]==self.n_nanowires+1)[0] #all the wires contacting the right electrode

        total_nodes_in_principle=len(overall)-len(on_left_el)-len(on_right_el)+2 #we count the electrode as only 1 big potential node.

        #now we want to numerate all the nodes, so that they correspond to the m,n entry of a sparse matrix.
        labels=np.arange(len(overall))
        labels=np.ma.array(labels,mask=False)
        exclude=np.hstack( (on_left_el, on_right_el) )
        labels.mask[exclude]=True
        to_label_in_order=labels.data[labels.mask==False]
        label_vector = np.zeros(len(overall))
        label_vector[to_label_in_order]=np.arange(len(to_label_in_order)) #all the potential nodes that are not an electrode, 1,2,3,4.....etc
        label_vector[on_left_el]=total_nodes_in_principle-2 #all the potential nodes on the electrodes have the same number, so they're the same node. the electrode is equipotential
        label_vector[on_right_el]=total_nodes_in_principle-1 #all the potential nodes on the electrodes have the same number, so they're the same node. the electrode is equipotential
        
        
        overall=np.hstack( (overall,label_vector.reshape((len(label_vector),1)) ) ) #just for convenience I stick these arrays together.

        who_crossed_and_where_left=overall[on_left_el][:,[1,3]] #j, and t, i.e the contact wire's index, and where that wire is cut on the interval [0,1] along its length
        who_crossed_and_where_right=overall[on_right_el][:,[1,3]]

        num_contacts_on_el_left=np.zeros(len(who_crossed_and_where_left))
        num_contacts_on_el_right=np.zeros(len(who_crossed_and_where_right))
        
  
        el_cres_is_cres_over_N=True #We realized that the contact resistance onto the electrodes has to be computed 
        #c_res/N for each wire, where N is how many contacts the wire forms ON TOP of the electrode area.
        #If this is not the case, there's large artefacts and too much current drops over the electrode
        #This is especially visible in the fact that the sample doesn't scale (twice as long, not twice as resistive)
        #We can discuss if you want why we think this makes sense intuitively, beyond the cold fact that this allows the system
        #to scale correctly in width (R~1/width) and length (R~length).


        #Here we compute the conductivies between the nodes.
        if el_cres_is_cres_over_N==True:   
        #here we do consider that the contact resistance is effectively cres/N over the electrode area.
            #we first take care of the electrodes.
            #left elec
            for i in range(len(who_crossed_and_where_left)):
                wire=who_crossed_and_where_left[i,0] #the wires contacting the electrode
                the_crossing = who_crossed_and_where_left[i,1] #where along their own lengths they cut the electrode
                crossings = overall[overall[:,0]==wire][:,2] #all the cuts that those wires have (their own s)
                num_contacts_on_el_left[i]=np.sum(crossings<=the_crossing) #how many s cuts are happening over the electrode area (N in cres/N)
            #right elec      
            for i in range(len(who_crossed_and_where_right)):
                wire=who_crossed_and_where_right[i,0]
                the_crossing = who_crossed_and_where_right[i,1]
                crossings = overall[overall[:,0]==wire][:,2]
                num_contacts_on_el_right[i]=np.sum(crossings>=the_crossing) #oriented left-right


            total_conts=np.hstack((num_contacts_on_el_left,num_contacts_on_el_right))
            length=len(overall)
            
            #The pairs_cuts_.. was stacked on itself and permuted to get the symmetric i,j, and j,i cuts.
            #So, if there are 5000 cuts, when we stack, there's 10000.
            #Then, the cut between i and j on wire i, let's say, is node number 7, then the node 
            #on wire j is 5007. Which explains the 'reshape' below.
            g_contacts=np.vstack( (overall[:,5].reshape(2,length/2),1/c_res*np.ones(length/2).reshape(1,length/2)) ) #the  
            g_contacts[2,-len(total_conts):]=total_conts/c_res
            #all these nodes have a conductance of 1/c_res between them, these are the contacts. Except for the electrodes.
            #the g_contacts are matrix elements for the electrodes, and the conductances accross contacts. (upper triangular however. We have node n node m, but not m,n. Later.
        else:
            g_left=np.zeros(len(who_crossed_and_where_left))
            g_right=np.zeros(len(who_crossed_and_where_right))

            for i in range(len(who_crossed_and_where_left)):
                wire=who_crossed_and_where_left[i,0]
                the_crossing = who_crossed_and_where_left[i,1]
                crossings = overall[overall[:,0]==wire][:,2]
                crossings=crossings[crossings<=the_crossing]
                g_left[i]=np.sum(1/ (np.diff(np.sort(crossings))*lin_res+c_res) ) #oriented left-right

            for i in range(len(who_crossed_and_where_right)):
                wire=who_crossed_and_where_right[i,0]
                the_crossing = who_crossed_and_where_right[i,1]
                crossings = overall[overall[:,0]==wire][:,2]
                crossings=crossings[crossings>=the_crossing]
                g_right[i]=np.sum(1/ (np.diff(np.sort(crossings))*lin_res+c_res) ) #oriented left-right
                

            total_g=np.hstack((g_left,g_right))
            length=len(overall)
            g_contacts=np.vstack( (overall[:,5].reshape(2,length/2),1/c_res*np.ones(length/2).reshape(1,length/2)) )
            g_contacts[2,-len(total_g):]=total_g
            
        self.g_contacts=g_contacts #storing, if you want to retrieve during simulation
        self.overall=overall #storing, if you want to retrieve during simulation
        
        #Let's take time to breathe. So now we have constructed part of the matrix which links all the nodes accross the contacts.
        #But the nodes also receive a contribution from the neighboring nodes on the wires.
        #So every node receives an input from the node accross the contact, and its 1 or 2 neighboring wire nodes.
        #We compute the rest of these connections below.
        
        
        #We first sort all the wires by the wire index
        rearg = np.argsort(overall[:,0])
        overall=overall[rearg]
        
        #the strategy is: 
        #1. we check all the cuts on wire i
        #2. sort the cuts from left (s=0) to right (s=1)
        #3. We check which nodes these cuts correspond to
        #4. We give the segment between neighboring nodes the conductivty of the length of wire separating them.
        
        #a very efficient way to go through indices is to compute
        #a histogram of how many indices of each there is
        #and then cumulatively sum their number up
        #so that we can slice this vector which takes us through
        #all wire indices of 1, then 2, etc...
        
        #histogram
        vals,bins=np.histogram(overall[:,0],np.arange(self.n_nanowires+3))
        ids=np.cumsum(np.hstack( (0,vals)))
        #ids[0]:ids[1] gives all wires of index 1
        #ids[1]:ids[2] gives all the wires of index 2,etc (because we sorted the overall array)
        self.nodes_to_wire = ids #stocking if needed
        self.node_labels = overall[:,5]#stocking if needed
        self.s_crossings = overall[:,2]#stocking if needed

        on_wire=[]

        for i in range(len(ids)-3): #the last indices are elcetrodes, we computed already their resistances.
        
            #point 1. above
            s_chunk_on_wire = overall[ids[i]:ids[i+1],2]
            #point 3. above
            node_labels = overall[ids[i]:ids[i+1],5]
            #point 2. above
            rearg=np.argsort(s_chunk_on_wire)
            node_labels=node_labels[rearg]
            s_chunk_on_wire=s_chunk_on_wire[rearg]
            #point 4.above
            first_nodes=node_labels[0:-1]
            second_nodes=node_labels[1:]
            #point 4.above, calculating lengths
            cuts=np.diff(s_chunk_on_wire)
            cuts[cuts==0]=0.01 #sometimes rounding errors occur and makes length to 0..
            conductances = 1./(cuts*lin_res) #between neighboring nodes.
            on_wire.append(np.vstack( (first_nodes,second_nodes,conductances) )) #this the matrix elements for conductances along wires.
        
        on_wire=np.hstack(map(tuple,on_wire)) #manipulate numpy
        all_upper_tri_matrix_elements=np.hstack( (g_contacts,on_wire) )
        symetric_conductance_matrix=np.hstack( (all_upper_tri_matrix_elements,all_upper_tri_matrix_elements[[1,0,2]]) ) #up to now, we only constructed the upper triangular matrix, only the n,m entries, not m,n.
        #This is a 3xN vector, with 0,: the n entry of the matrix, 1,: the m entry, 2,: the conductance.
        self.ijrc=symetric_conductance_matrix #stocking for debug.
        self.tot_nodes=total_nodes_in_principle
        sparse_mat = csr_matrix((symetric_conductance_matrix[2], (symetric_conductance_matrix[0].astype(int),symetric_conductance_matrix[1].astype(int))), shape=(total_nodes_in_principle,total_nodes_in_principle))
        N=total_nodes_in_principle
        col_diag = np.arange(N)

        #construct the zero current equations from the conductance matrix. (equation S1)
        
        data_diag = np.reshape(np.asarray(sparse_mat.sum(1)),(N,)) 
        indptr_diag=np.arange(N+1)
        sparsified_diagonal_term = csr_matrix( (data_diag,col_diag,indptr_diag),shape=(N,N)) #the headache of how to construct csr matrices.. :)

        sparse_mat=sparsified_diagonal_term-sparse_mat #this is the matrix, that when multiplie with the unknown vector of vec(u), will give equation S1.
        current = np.zeros(N-1) #we will kill one entry later in the matrix.
        print(N)
        current[-1]=1 #the current on the left electrode is 1 (last node is right electrode, but the vector is N-1 long, so this would be the entry on the left electrode potential.)
        sol=spsolve(sparse_mat[:-1,:-1],current) #the second condition is that the right electrode has 0 potential, so we can delete its row and column from the matrix.
        #sol=spsolve(sparse_mat[:-1,:-1],current) #the reason for that, is that u_n=0 means the n-th column is always multiplied by 0, so we can leave it
        #and the last row is not an equation we do not need, because we have enough equations to solve the system.
        self.pots=sol #all the potential nodes

        self.res_system_wired_sparse=sol[-1] #because current=1, u_right=0, u_left is R.
        self.thesolvees=sparse_mat[:-1,:-1]
        
        
    
    def stretch(self, eps):

       
        sleeve=self.pos_list.copy()

        sleeve[:,0]*=(1+eps)

        sleeve[:,3]*=(1+eps)

        sleeve[:,1]*=(1+eps)**(-self.poisson_ratio)
        sleeve[:,2]*=(1+eps)**(-self.poisson_ratio)
        sleeve[:,4]*=(1+eps)**(-self.poisson_ratio)
        sleeve[:,5]*=(1+eps)**(-self.poisson_ratio)
        

        mid_point = (sleeve[:,0:3]+sleeve[:,3:6])*0.5

        for i in range(self.n_nanowires):

            bit_backward = sleeve[i][0:3]-mid_point[i]

            bit_forward = sleeve[i][3:6]-mid_point[i]

            l_half_sleeve = np.linalg.norm(bit_backward)

            l_half_wire = self.nanowire_list[i].length/2.

            

            

            self.pos_list[i][0:3]=mid_point[i] + bit_backward*l_half_wire/l_half_sleeve #(A+(B-A)*ratio_to_keep wire length const)

            self.pos_list[i][3:6]=mid_point[i] + bit_forward*l_half_wire/l_half_sleeve

            self.nanowire_list[i].begin_pos=self.pos_list[i][0:3]

            self.nanowire_list[i].end_pos=self.pos_list[i][3:6]

        

        self.x*=(1+eps)
        self.eps=eps
        self.y*=(1+eps)**(-self.poisson_ratio)
        self.z*=(1+eps)**(-self.poisson_ratio)


#@jit is a decorator from numba.
#what it does is 'precompile' the code and it will avoid
#all the 'type'-checks everytime the code enters a for-loop
#essentially making this c-code speed.It just needs a bit more nomenclature.
@jit(nopython=True)
def dot(a,b):

    res = a[0]*b[0]+a[1]*b[1]+a[2]*b[2]

    return res
    
 
#maths from http://www.cs.swan.ac.uk/~cssimon/line_intersection.html    
@jit(nopython=True)
def compute_crossings(array_to_compute,effective_kd_pairs,is_electrode=False):
    
    N = array_to_compute.shape[0]
    
    con_mat = [] #this will stock i,j wires if they intersect, the exact s,t where they intersect. (wire_i = p0+s*(p1-p0), wire_j = q0+t*(q1-q0), s,t in [0,1])
    
    
    
    p0 = np.zeros(3)    
    p1 = np.zeros(3)   
    q0 = np.zeros(3)    
    q1 = np.zeros(3)
       
    
    p0m1 = np.zeros(3)
    p1m0 = np.zeros(3) 
    q0m1 = np.zeros(3) 
    q1m0 = np.zeros(3) 
    p0mq0 = np.zeros(3)
    
    #we only loop on pairs that have the chance to intersect (whose centers are not further than a wire length apart)
    #this provides a massive acceleration because it only makes the necessary pair calculations.
        
    for z in range(len(effective_kd_pairs)):
        
        i = effective_kd_pairs[z,0]
        j = effective_kd_pairs[z,1]
        #unrolling this computation is more efficient for numba
        for k in xrange(3):



            p0[k] = array_to_compute[i,k] #x0,y0 or z0 for k=0,1,2 (wires are stored like [x1,y1,z1,x2,y2,z2])
            p1[k] = array_to_compute[i,k+3] #x1,y1 or z1 for k=0,1,2 +3  (wires are stored like [x1,y1,z1,x2,y2,z2])
            q0[k] = array_to_compute[j,k]
            q1[k] = array_to_compute[j,k+3]
            


            p0m1[k] = p0[k] - p1[k]
            p1m0[k] = -p0m1[k]
            q0m1[k] = q0[k] - q1[k]
            q1m0[k] = -q0m1[k]



            p0mq0[k] = p0[k] - q0[k]


        
        s = ( dot(p1m0, q1m0)*dot(q1m0, p0mq0) - dot(q1m0, q1m0)*dot(p1m0, p0mq0))/( dot(p1m0, p1m0)*dot(q1m0, q1m0) - dot(p1m0, q1m0)**2 )

        t = ( dot(p1m0, p1m0)*dot(q1m0, p0mq0) - dot(p1m0, q1m0)*dot(p1m0, p0mq0))/( dot(p1m0, p1m0)*dot(q1m0, q1m0) - dot(p1m0, q1m0)**2 )

        c=0
        if is_electrode:
            if i>=N-2:
                for k in xrange(3):

                    c += (p0[k]+(p1[k]-p0[k])*s-(q0[k]+(q1[k]-q0[k])*t))**2 #the square of the intersection distance. relevant only for the 3D case to define a 'contact' threshold.
                con_mat.append((i,j,s,t,c))
            else:
                if s>=0 and s<=1 and t>=0 and t<=1: #s in [0,1], t in [0,1] guarantees the intersection is within the wires, and not 'virtually' outside them.

                    for k in xrange(3):

                        c += (p0[k]+(p1[k]-p0[k])*s-(q0[k]+(q1[k]-q0[k])*t))**2 #the square of the intersection distance. relevant only for the 3D case to define a 'contact' threshold.
                    con_mat.append((i,j,s,t,c))
        else:
            if s>=0 and s<=1 and t>=0 and t<=1: #s in [0,1], t in [0,1] guarantees the intersection is within the wires, and not 'virtually' outside them.
                for k in xrange(3):

                    c += (p0[k]+(p1[k]-p0[k])*s-(q0[k]+(q1[k]-q0[k])*t))**2 #the square of the intersection distance. relevant only for the 3D case to define a 'contact' threshold.
                con_mat.append((i,j,s,t,c))
    return con_mat
    
