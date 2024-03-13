
from other.ecp import ecp
from other.multirank.dynkw import autoDynKWRupt
import numpy as np
from other.mnwbs_changepoints import mnwbs_changepoints
from other.ruptures import  kernseg_rbf,kernseg_linear
from changeforest import changeforest,Control
from time import perf_counter



class DetectionMethhod:
    def __init__(self):
        pass
    
    def ecp_method(self,X,minimal_relative_segment_length):
        return list(np.array(ecp(X,minimal_relative_segment_length))-1)
    def multi_rank_method(self,X,minimal_relative_segment_length):
        __, cpts = autoDynKWRupt(X.T, Kmax=int(1 / minimal_relative_segment_length))
        return np.append([0], cpts[cpts != 0] + 1)
    def mnwbs_method(self,X,minimal_relative_segment_length):
       return mnwbs_changepoints(X, minimal_relative_segment_length)
        
    def rbf_kcp_method(self, X, minimal_relative_segment_length):
       return kernseg_rbf(X, minimal_relative_segment_length=minimal_relative_segment_length)
   
    def linear_kcp_method(self, X, minimal_relative_segment_length):
        return kernseg_linear(X, minimal_relative_segment_length=minimal_relative_segment_length)
        
    def our_method(self,X,minimal_relative_segment_length,method,s_method):
        return ([0]+ changeforest(X, method, s_method, 
                                  Control(minimal_relative_segment_length=minimal_relative_segment_length)).split_points()+ [len(X)])
    
    def change_in_mean_method(self,X,minimal_relative_segment_length):
        return self.our_method(X,minimal_relative_segment_length,"change_in_mean", "bs")
    
    def change_knn_method(self,X,minimal_relative_segment_length):
        return self.our_method(X,minimal_relative_segment_length,"knn", "bs")
    
    def change_forest_method(self,X,minimal_relative_segment_length):
        return self.our_method(X,minimal_relative_segment_length,"random_forest", "bs")
    
    def run_all_methods(self,X,minimal_relative_segment_length,ecp=True,mnwbs=True):
        change_points = []
        time_taken = []
       
        names = ['Change in Mean','change KNN', 'ECP','KCP-rbf','KCP-linear',"MultiRank","MNWBS","Change Forest"] 
        methods = [self.change_in_mean_method,self.change_knn_method,self.ecp_method,
                   self.rbf_kcp_method,self.linear_kcp_method,self.multi_rank_method,self.mnwbs_method,self.change_forest_method]
        if not mnwbs:
            names.pop(-2)
            methods.pop(-2)
        if not ecp:
            names.pop(2)
            methods.pop(2)
        for method in methods:
            start = perf_counter()
            change_points.append(method(X,minimal_relative_segment_length))
            end = perf_counter()
            time_taken.append(end-start)
        return change_points,names,time_taken
            
        

        
        
    
    
    
        
            
            
        
        
    
    
            