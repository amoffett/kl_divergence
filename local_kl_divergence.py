import numpy as np
from random import shuffle
import math
import itertools as it
import mdtraj as md
import pickle as pkl

class local_kl_divergence:
    """
    Measures the local Kullback-Liebler divergence (KL divergence), as defined in [1], between two sets of simulations.
    
    KL_{res_{n}} = sum_{phi,psi,chi1}{sum_{i}^{nbins}{p_{i} * ln(p_{i} / p_{i}^{*})}}
    
    In words, the local KL divergence for residue n is the sum over all dihedral angles of residue n and all
    bins for each angle of the empirical probability of being in bin i in the test ensemble (p_{i}), times the natural
    logarithm of the probability of being in bin i in the test ensemble divided by the probability of being in 
    bin i in the reference ensemble (p_{i}^{*}).
    
    Notation from [1] is replicated as closely as possible using LaTeX format while still maintaining clear readability. 
    
    Inputs:
        __init__:
        ref ==> A list of mdtraj trajectories in the reference ensemble. Each trajectory should be stripped so that ONLY PROTEIN residues remain.
        test ==> A list of trajectories in the test ensemble. Each trajectory should be stripped so that ONLY PROTEIN residues remain.
        dihedrals ==> A list of dihedral angles to measure for each residue.
                        Choices are: ['phi','psi','chi1']
        
        featurize: Requires no input, except that provided in __init__.
        
        kl_div:
        nblocks ==> The number of 'blocks' to split the reference simulation set into for bootstrapping. nblocks 
                    must be less than or equal to the number of trajectories in the reference set. This will
                    randomly sort trajectories into nblocks different blocks as evenly as possible for use in the
                    bootstrapping method described in [1]. 
        bins ==> The number of bins to use in each histogram.
        binrange ==> The range of values to take each histogram over. The length of each bin is then: binrange / bins
        gamma ==> A pseudocount added to all counts to deal with zero counts in bins.
        alpha ==> Cutoff for p value from bootstrap distribution. If p < alpha, the returned value is KL_{res_{n}}
                    minus KL_{res_{n}}^{H0}, the mean of the bootstrap distribution.
                    
    Typical usage could resemble the following:
    
    >>> kl = local_kl_divergence(ref_trajs,test_trajs)
    >>> kl.featurize()
    >>> kl_div = kl.kl_div(nblocks = 2, bins = 20, binrange = [-np.pi, np.pi], gamma = .001, alpha = .05)
    >>> kl_div
    
    [[GLN1, 1.7464315375477832],
     [LEU2, 0.77049443999035305],
     [LYS3, 5.1969705269195927],
     [ARG4, 0],
     [PHE5, 1.06779220260472],
     [SER6, 6.5869069213542755],
     [LEU7, 3.7789105422773615],
     [ARG8, 4.8437410013039441],
     [GLU9, 4.3867046980773727],
     [LEU10, 11.494938678935977],
     [GLN11, 14.131194705172048],
     [VAL12, 9.3685133906337317],
     [ALA13, 9.3789445609694315],
     [SER14, 27.091639934261135],
     [ASP15, 23.831492435931217],
     ...
 
    References:
    [1] McClendon, C. L., Hua, L., Barreiro, G. and Jacobson, M P. J. Chem. Theory Comput. 2012, 8, 2115-2126. 
    """
    
    def __init__(self, ref, test, dihedrals = ['phi','psi','chi1']):
        self.ref = ref
        self.test = test
        self.dihedrals = dihedrals
        self.dih_ref = None
        self.dih_test = None
        
        for dihedral in self.dihedrals:
            if dihedral not in ['phi','psi','chi1']:
                print "Dihedral angle \'%s\' not supported." %dihedral
            
        def truth(val, ref):
            if val == ref:
                return True
            
        vtruth = np.vectorize(truth)
            
        def check_inputs():
            if self.ref[0].n_residues != self.test[0].n_residues:
                print "Different number of residues in the reference and test simulation sets."
                return None
            elif not np.all(vtruth([traj.n_residues for traj in self.ref],self.ref[0].n_residues)):
                print "Different number of residues in different reference trajectories."
                return None
            elif not np.all(vtruth([traj.n_residues for traj in self.test],self.test[0].n_residues)):
                print "Different number of dihedral angles in different test trajectories."
                return None
        
    def dihedral_featurizer(self, trajs):
        
        def phi_feat(traj, res):
            phi = traj.topology.select('(resid %i and name C) or (resid %i and (name N or name CA or name C))' %(res - 1, res))
            phi = phi.reshape([1,4])
            traj_phi = md.compute_dihedrals(traj, phi)
            return traj_phi
        
        def psi_feat(traj, res):
            psi = traj.topology.select('(resid %i and (name N or name CA or name C)) or (resid %i and name N)' %(res, res + 1))
            psi = psi.reshape([1,4])
            traj_psi = md.compute_dihedrals(traj, psi)
            return traj_psi
        
        def chi1_feat(traj, res):
            chi1 = traj.topology.select('resid %i and (name C or name CA or name CB or name CG or name SG or name CG1 or name OG or name OG1)' %res)
            if chi1.shape[0] != 4:
                return None
            chi1 = chi1.reshape([1,4])
            traj_chi1 = md.compute_dihedrals(traj, chi1)
            return traj_chi1
        
        residue_list = []
        
        for res in range(1, trajs[0].n_residues - 1):
            residue_n = []
            for traj in trajs:
                traj_k = []
                if 'phi' in self.dihedrals:
                    phi = phi_feat(traj, res)
                    traj_k.append(phi)
                if 'psi' in self.dihedrals:
                    psi = psi_feat(traj, res)
                    traj_k.append(psi)
                if 'chi1' in self.dihedrals:
                    chi1 = chi1_feat(traj, res)
                    if chi1 is not None:
                        traj_k.append(chi1)
                traj_k = np.hstack(traj_k)
                residue_n.append(traj_k)
            residue_n_name = trajs[0].topology.residue(res)
            residue_list.append([residue_n_name, residue_n])
        
        return residue_list 
    
    def prob(self, trajs, bins = 20, binrange = [-np.pi, np.pi], gamma = .001):
            all_p_dists = []
            for resname, res in trajs:
                res = np.vstack(res)
                p_dists = []
                for dihedral in range(res.shape[1]):
                    hist = np.histogram(res[:, dihedral],bins = bins, range = binrange)
                    p_dists.append(hist[0])
                p_dists = np.vstack(p_dists) + gamma
                normal = p_dists.sum(axis = 1).reshape([res.shape[1],1])
                p_dists = p_dists / normal.astype(float)
                all_p_dists.append([resname, p_dists])
            return all_p_dists
    
    def featurize(self, write_features = False, ref_set_name = 'ref_set_dihedrals.pkl', test_set_name = 'test_set_dihedrals.pkl'):
        self.dih_ref = self.dihedral_featurizer(self.ref)
        self.dih_test = self.dihedral_featurizer(self.test)
	if write_features == True:
		with open(ref_set_name, 'wb') as f:
			pkl.dump(self.dih_ref, f)
		with open(test_set_name, 'wb') as f:
			pkl.dump(self.dih_test, f)
    
    def load_features(self, ref, test):
	with open(ref, 'rb') as f:
		ref_set = pkl.load(f)
	with open(test, 'rb') as f:
		test_set = pkl.load(f)
	self.dih_ref = ref_set
	self.dih_test = test_set

    def kl_div_H0(self, nblocks = 10, bins = 20, binrange = [-np.pi, np.pi], gamma = .001):
        if (self.dih_ref == None) or (self.dih_test == None):
            print "Either run featurize or load features from a previous featurization first."
            return None
        
        def even_split(total, nblocks):
            if nblocks > total:
                print "Cannot have more blocks than simulations."
                return None
            elif nblocks % 2 != 0:
                print "\'nblocks\' must be an even number"
                return None
            integer_divisor = total / nblocks
            remainder = total % nblocks
            splits = [0]
            for i in range(1,remainder + 1):
                splits = splits + [splits[i - 1] + integer_divisor + 1]
            for i in range(remainder + 1, nblocks + 1):
                splits = splits + [splits[i - 1] + integer_divisor]
            return splits
        
        def kl_div(ref, test, bins = bins, binrange = binrange, gamma = gamma):
            kl_div_list = []
            for res in range(ref.shape[1]):
                p_ref = np.histogram(ref[:, res], bins = bins, range = binrange)[0] + gamma
                p_ref = p_ref / float(p_ref.sum())
                p_test = np.histogram(test[:, res], bins = bins, range = binrange)[0] + gamma
                p_test = p_test / float(p_test.sum())
                kl_div = np.sum(p_test * np.log(p_test / p_ref))
                kl_div_list.append(kl_div)
            residue_n_kl_div = np.vstack(kl_div_list).sum()
            return residue_n_kl_div
        
        dih_all_res_blocks = []
        for resname, res in self.dih_ref:
            shuffle(res)
            res_splits = even_split(len(res), nblocks)
            dih_ref_blocks = []
            for n in range(nblocks):
                block = res[res_splits[n]:res_splits[n+1]]
                block = np.vstack(block)
                dih_ref_blocks.append(block)
            dih_all_res_blocks.append(dih_ref_blocks)
        
        kl_div_H0 = []
        kl_div_bootstrap = []
        for res in range(len(self.dih_ref)):
            comb_kl_div = []
            block_set = set(range(len(dih_all_res_blocks[res])))
            for block in it.combinations(block_set, len(block_set) / 2):
                first_chunk_blocks = list(block)
                second_chunk_blocks = list(block_set.difference(set(block)))
                first_chunk = np.vstack([dih_all_res_blocks[res][traj] for traj in first_chunk_blocks])
                second_chunk = np.vstack([dih_all_res_blocks[res][traj] for traj in second_chunk_blocks])
                comb_kl_div.append(kl_div(first_chunk, second_chunk))
            
            kl_div_H0_n = np.sum(comb_kl_div)
            normal = float(math.factorial(nblocks) / (math.factorial(nblocks / 2) * math.factorial(nblocks / 2)))
            kl_div_H0_n = kl_div_H0_n / normal
            kl_div_bootstrap.append(comb_kl_div)
            kl_div_H0.append(kl_div_H0_n)
        return kl_div_H0, kl_div_bootstrap
    
    def kl_div(self, nblocks = 10, bins = 20, binrange = [-np.pi, np.pi], gamma = .001, alpha = .05):
        if (self.dih_ref == None) or (self.dih_test == None):
            print "Either run featurize or load features from a previous featurization first."
            return None
        kl_div_H0, kl_div_bootstrap = self.kl_div_H0(nblocks = nblocks, bins = bins, binrange = binrange, gamma = gamma)
        p_ref = self.prob(self.dih_ref, bins = bins, binrange = binrange, gamma = gamma)
        p_test = self.prob(self.dih_test, bins = bins, binrange = binrange, gamma = gamma)
        n_res = len(p_ref)
        kl_div_list = []
        for res in range(n_res):
            resname = p_test[res][0]
            kl_div = np.sum(p_test[res][1] * np.log(p_test[res][1] / p_ref[res][1]))
            kl_div_bootstrap_n = np.vstack(kl_div_bootstrap[res])
            p_value = np.sum(kl_div_bootstrap_n[kl_div_bootstrap_n[:] > kl_div])/float(np.sum(kl_div_bootstrap_n))
            if p_value < alpha:
                kl_div = kl_div - kl_div_H0[res]
            else:
                kl_div = 0
            kl_div_list.append([resname, kl_div])
        return kl_div_list
