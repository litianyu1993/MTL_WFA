import numpy as np
from sp2learn import Sample as Sample, Learning, Hankel as Hankel

import copy

from scipy.sparse.linalg import svds as sparse_svd
from scipy import sparse as sps



def flatten_k(H, k):
    return H.reshape((H.shape[k], -1))


class Q_WFA:
    def __init__(self,alpha=[], Omega=[], As=[]):
        '''
        Initialization of the class. Note all the parameters are a list of that correposding parameters,
        list elements' index corresponds to the task, i.e. to acess the parameters of task id 0, you can
        use alpha[0], Omega[0], As[0].
        :param alpha: List of initial vectors
        :param Omega: List of termination vectors
        :param As: List of transition matrices
        '''
        self.alpha = alpha
        self.Omega = Omega
        self.As = As

        self.alpha_vec = []
        self.As_vec = []
        self.Omega_vec = []


    def select_rows_columns(self, input_file, n_rows, n_cols, version):
        '''
        Selecting rows and columns of the Hankel matrix
        :param input_file: the address of the training file
        :param n_cols: desired number of rows
        :param version: desired number of colums
        :return: rows, cols and the sample instance
        '''
        pT = Sample(adr=input_file, version=version)
        rows = pT.select_rows(nb_rows_max=n_rows, version=version)
        cols = pT.select_columns(nb_columns_max=n_cols, version=version)
        #print(len(rows))
        return rows, cols, pT

    def Build_Hankel(self, pT, rows, cols, version='classic', sparse=False):
        '''
        Building the Hankel matrix
        :param pT: Sample instance
        :param rows: rows
        :param cols: cols
        :param version: version of the Hankel matrix, currently only support 'classic'
        :param sparse: Boolean, if the Hankel matrix is in DOX sparse matrix form
        :return: Hankel matrix
        '''
        H = np.array(Hankel(sample_instance=pT, lrows=rows, lcolumns=cols, version=version,
                                        partial=True, sparse=sparse).lhankel)
        return H


    def convert_meta_task_Q_WFA(self, alpha, As, Omega, n_tasks, P, P_vec, R_tasks_vec):
        '''
        Converting the meta WFA to its task-specific, projected version
        :param alpha: meta WFA's initial vector
        :param As: meta WFA's transition matrices
        :param Omega: meta WFA's termination vector
        :param n_tasks: number of tasks
        :param P: Left singular vectors of the meta Hankel matrix
        :param P_vec: Left singular vectors of each individual Hankel matrix corresponding to each task
        :param R_tasks_vec: Desired rank for each task-specific WFA
        :return:
        '''
        alpha_vec = []
        As_vec = []
        Omega_vec = []
        for i in range(n_tasks):
            P_task= P_vec[i][:, :R_tasks_vec[i]]
            #print(P_task)
            change_basis = np.linalg.pinv(P_task) @ P
            change_basis_inv = np.linalg.pinv(change_basis)
            alpha_temp = alpha[i]
            alpha_temp = alpha_temp.reshape(1, -1) @ change_basis_inv
            Omega_temp = Omega[i]
            Omega_temp = change_basis @ Omega_temp
            As_temp = As[i]
            temp_As = []
            for j in range(len(As_temp)):
                temp_As.append(change_basis @ As_temp[j] @ change_basis_inv)
            alpha_vec.append(alpha_temp)
            Omega_vec.append(Omega_temp)
            As_vec.append(temp_As)

        self.alpha = alpha_vec
        self.Omega = Omega_vec
        self.As = As_vec
        return

    def fit_WFA(self, H, R, task_id_vec=[0], R_tasks_vec=None,
                P_vec=None,  version='classic', sparse=False, return_P=False):
        '''
        General funtion for fitting the WFA
        :param H: Hankel matrix (can be either meta or task-specific one)
        :param R: Desired rank of the SVD of H
        :param task_id_vec: the vector of task ids
        :param R_tasks_vec: If executing multi_proj, set this parameter to desired task-specific rank
        :param P_vec: left singular vectors of each individual Hankel matrix corresponding to each task
        :param version: version of the Hankel matrix, currently only support 'classic'
        :param sparse: Boolean, if the Hankel matrix is in DOX sparse matrix form
        :param return_P: if you want to return the singular vectors of the meta Hankel matrix
        :return:
        '''
        # sparse = False
        try:
            if sparse:
                acc = []
                for task in range(len(H)):
                    acc.append(H[task][0])
                acc = sps.hstack(acc)
                U, D, VT = sparse_svd(acc, k=R)
            else:
                H = np.array(H)
                H = H.transpose((1, 2, 0, 3))
                H_1 = flatten_k(H[0], 0)
                U, D, VT = np.linalg.svd(H_1)
            P = U[:, :R].dot(np.diag(D[:R]))
            S = VT[:R, :]
            P_inv = np.linalg.pinv(P)
            S_inv = np.linalg.pinv(S)
            alpha = P[0]
            if sparse:
                acc = []
                for task in range(len(H)):
                    acc.append(H[task][0][:, 0])
                acc = sps.hstack(acc)
                Omega = P_inv @ acc
                As = []
                for sigma in range(1, len(H[0])):
                    acc = []
                    for task in range(len(H)):
                        acc.append(H[task][sigma])
                    acc = sps.hstack(acc)
                    A_sigma = P_inv @ acc @ S_inv
                    As.append(A_sigma)
                As = np.array(As)
            else:
                Omega = P_inv @ H[0, :, :, 0]
                As = []
                for sigma in range(1, H.shape[0]):
                    H_1_sigma = flatten_k(H[sigma], 0)
                    A_sigma = P_inv @ H_1_sigma @ S_inv
                    As.append(A_sigma)
                As = np.array(As)
            A = np.sum(As, axis=0)
            # print(R)
            for k in task_id_vec:
                self.alpha.append(alpha)
                self.As.append(As)
                self.Omega.append(Omega[:, k])
            if P_vec is not None:
                n_tasks = len(task_id_vec)
                alpha = copy.deepcopy(self.alpha)
                As = copy.deepcopy(self.As)
                Omega = copy.deepcopy(self.Omega)
                self.convert_meta_task_Q_WFA(alpha, As, Omega, n_tasks, P, P_vec, R_tasks_vec)

        except:
            #print(error)
            return 'Failed', []
        if return_P:
            return 'Success', P
        return 'Success', []
    def compute_value(self, x_vec, task_id):
        '''
        Compute the value output by the WFA
        :param x_vec: the input sequence
        :param task_id: which task you want to evaluate one
        :return: the value of the sequence computed by the corresponding WFA
        '''
        alpha = self.alpha[task_id].reshape(1, -1)
        Omega = self.Omega[task_id].reshape(-1, 1)
        temp = alpha
        As = self.As[task_id]
        for x in x_vec:
            #print(x, len(As))
            temp = np.dot(temp, As[x])
        temp = np.dot(temp, Omega)
        #print(alpha.shape)
        return abs(temp[0][0]) #abs() is to avoid nagative values when taking log (perplexity)
    def sum_to_one(self, y):
        '''
        make the sequence sum to one for computing perplexity
        :param y: The input sequence
        :return: A sequence that sum to one
        '''
        total = sum(y)
        for i in range(len(y)):
            y[i] /= total
        return y
    def compute_perplexity(self, y, y_hat):
        '''
        Compute the perplexity, following the equation from the Pautomac challenge:
        http://ai.cs.umbc.edu/icgi2012/challenge/Pautomac/download.php
        :param y: The actual probability of the testing input file
        :param y_hat: The estimated probability (sum to one) of the testing input file
        :return: Perplexity
        '''
        temp = 0.
        y_hat = self.sum_to_one(y_hat)
        for i in range(len(y)):
            temp += y[i]*np.log2(y_hat[i])
        perp = 2**(-temp)
        return perp


def fit_independent_WFA(H_P, q_wfa, R, version, sparse):
    '''
    Fit each tasks' WFA individually
    :param H_P: Hankel matrix
    :param q_wfa: a Q_WFA instance (class above)
    :param R: Desired rank
    :param version: version of the Hankel matrix, currently only support 'classic'
    :param sparse: Boolean, if the Hankel matrix is in DOX sparse matrix form
    :return: the status of the construction, the Q_WFA instance
    '''
    H = np.asarray([H_P])
    qwfa = q_wfa.fit_WFA(H, R, task_id_vec=[0],
                             P_vec=None, version=version, sparse=sparse)
    return qwfa, q_wfa


def fit_mtl_noproj_WFA(q_wfa, R, H_P, task_id_vec, version, sparse, return_P=False):
    '''
    Fit WFA with multi-tasking but without projection, i.e. meta-WFA
    :param q_wfa: a Q_WFA instance (class above)
    :param R: Desired rank
    :param H_P: Hankel matrix
    :param task_id_vec: task id vector
    :param version: version of the Hankel matrix, currently only support 'classic'
    :param sparse: Boolean, if the Hankel matrix is in DOX sparse matrix form
    :param return_P:
    :return: the status of the construction, the Q_WFA instance (meta-WFA)
    '''
    try:
        H_P = np.asarray(H_P)
    except:
        return 'Failed', q_wfa

    returns = q_wfa.fit_WFA(H_P, R, task_id_vec=task_id_vec,
                                P_vec=None, version=version, sparse=sparse)

    if returns[0] == 'Failed':
        return 'Failed', q_wfa
    else:
        if return_P:
            return 'Success', q_wfa, returns[-1]
        else:
            return 'Success', q_wfa




def fit_mtl_proj_WFA(q_wfa, R, R_task_vec, H_P,
                     task_id_vec, version, sparse):
    '''
    Fit WFA with multi-tasking and projection, i.e. task-specific WFAs
    :param q_wfa: a Q_WFA instance (class above)
    :param R: Desired rank
    :param H_P: Hankel matrix
    :param R_task_vec: Desired ranks for each task (in a vector form)
    :param task_id_vec: task ids vector
    :param version: version of the Hankel matrix, currently only support 'classic'
    :param sparse: Boolean, if the Hankel matrix is in DOX sparse matrix form
    :return: the status of the construction, the Q_WFA instance (task-specific WFAs)
    '''

    P_P = []
    for task_id in task_id_vec:
        H = H_P[task_id]
        if sparse:
            U, D, VT = sparse_svd(H[0], k=R)
        else:
            H_1 = flatten_k(H[0], 0)
            U, D, VT = np.linalg.svd(H_1)

        P = U[:, :R].dot(np.diag(D[:R]))
        P_P.append(P)

    H_P = np.asarray(H_P)
    qwfa = q_wfa.fit_WFA(H_P, R, task_id_vec=task_id_vec, R_tasks_vec=R_task_vec,
                             P_vec=P_P,version=version, sparse=sparse)
    return qwfa, q_wfa


def build_rows_cols_pTs(q_wfa, input_file_vec,
                        n_rows, n_cols, version, task_id_vec):
    '''
    Construct the rows and columns and the sample instance
    :param q_wfa: Q_WFA instance
    :param input_file_vec: A list of all the input training file addresses
    :param n_rows: Desired number of rows
    :param n_cols: Desired number of columns
    :param version: version of the Hankel matrix, currently only support 'classic'
    :param task_id_vec: Task id vector
    :return: the rows and columns and the sample instance
    '''
    rows_dic = {}
    cols_dic = {}
    pT_dic = {}
    for task_id in task_id_vec:
        input_file = input_file_vec[task_id]

        rows, cols, pT = q_wfa.select_rows_columns(input_file, n_rows, n_cols, version)
        rows_dic['%s, %s, %s' % (task_id, n_rows, n_cols)] = rows
        cols_dic['%s, %s, %s' % (task_id, n_rows, n_cols)] = cols
        pT_dic['%s, %s, %s' % (task_id, n_rows, n_cols)] = pT

    return rows_dic, cols_dic, pT_dic


def build_all_hankels_for_tasks(q_wfa, n_rows, n_cols, pT_dic, rows_dic, cols_dic,
                                version, sparse, task_id_vec):
    '''
    Construct Hankel matrices for all tasks
    :param q_wfa: Q_WFA instance
    :param n_rows: Desired number of rows
    :param n_cols: Desired number of columns
    :param pT_dic, rows_dic, cols_dic: parameters constructed by last function: build_rows_cols_pTs
    :param version: version: version of the Hankel matrix, currently only support 'classic'
    :param sparse: Boolean, if the Hankel matrix is in DOX sparse matrix form
    :param task_id_vec: task_id_vec: Task id vector
    :return: Hankel matrices
    '''
    H_P = []
    for task_id in task_id_vec:
        current_parameters = '%s, %s, %s' % (task_id, n_rows, n_cols)
        H = q_wfa.Build_Hankel(pT_dic[current_parameters], rows_dic[current_parameters],
                               cols_dic[current_parameters], version=version, sparse=sparse)
        H_P.append(H)
    return H_P

def learning(adr_vec, vali_adr_vec, vali_solution_adr_vec,
             n_rows, n_cols, meta_R,  task_id_vec, algorithm,  tasks_R= None,
             version = 'classic', sparse = 'False'):
    '''
    Learning algorithms for MTL_WFA as well as individual WFA
    :param adr_vec: A list of all training input file addresses
    :param vali_adr_vec: A list of all validation input file addresses
    :param vali_solution_adr_vec: A list of all validation output file addresses
    :param n_rows: Desired number of rows
    :param n_cols: Desired number of columns
    :param meta_R: The rank for meta-WFA
    :param tasks_R: The ranks for each task-specific WFAs/single task WFAs
    :param task_id_vec: Task id vecgtor
    :param algorithm: Algorithm you want to run: 'inde' for single task learning, 'multi-noproj' for multi-task
    with no projection (meta-WFA), 'multi-proj' for multi-task with projection
    :param version: version of the Hankel matrix, currently only support 'classic'
    :param sparse: Boolean, if the Hankel matrix is in DOX sparse matrix form
    :return: perplexity on the validation sets
    '''
    q_wfa = Q_WFA(alpha=[], As=[], Omega=[])
    rows_dic, cols_dic, pT_dic = build_rows_cols_pTs(q_wfa, adr_vec,
                        n_rows, n_cols, version, task_id_vec)
    H_P = build_all_hankels_for_tasks(q_wfa, n_rows, n_cols, pT_dic, rows_dic, cols_dic,
                                version, sparse, task_id_vec)
    q_wfa = Q_WFA(alpha=[], As=[], Omega=[])
    if algorithm == 'inde':
        for task_id in task_id_vec:
            status, q_wfa = fit_independent_WFA(H_P[task_id], q_wfa, tasks_R[task_id], version,
                                                sparse)
    elif algorithm == 'mtl_noproj':
        status, q_wfa = fit_mtl_noproj_WFA(q_wfa, meta_R, H_P, task_id_vec, version, sparse)
    else:
        status, q_wfa = fit_mtl_proj_WFA(q_wfa, meta_R, tasks_R, H_P,
                     task_id_vec, version, sparse)
    perplexity = []
    for task_id, vali_adr in enumerate(vali_adr_vec):
        values = []
        f = open(vali_adr, "r")
        line = f.readline()
        line = f.readline()
        while line:
            l = line.split()
            h = [int(x) for x in l]
            h = h[1:]
            values.append(q_wfa.compute_value(h, task_id))
            line = f.readline()
        solution = np.genfromtxt(vali_solution_adr_vec[task_id], delimiter=',')
        perp = q_wfa.compute_perplexity(solution, values)
        perplexity.append(perp)
    print('The perplexity for each tasks are (on validation sets): ', perplexity)
    return perplexity


'''
This is a toy example with Pautomac data, the 3rd dataset to be exact. 
You can access the data here: http://ai.cs.umbc.edu/icgi2012/challenge/Pautomac/download.php
The two tasks are identical, that's why there's no improvement on using multi-task learning. 
This is just a sanity check.
Note the true rank for this model is 24.
'''

adr_vec = ['./data/3.pautomac.train.task1', './data/3.pautomac.train.task1']
vali_adr_vec = ['./data/3.pautomac.test.task1', './data/3.pautomac.test.task1']
vali_solution_adr_vec = ['./data/3.pautomac_solution.task1.txt', './data/3.pautomac_solution.task1.txt']
n_rows = 1000
n_cols = 1000

meta_R = 24
tasks_R = [24, 24]
task_id_vec = [0, 1]
algorithms = ['inde', 'mtl_proj', 'mtl_noproj']
for algorithm in algorithms:
    print('Current algorithm is: ', algorithm)
    learning(adr_vec, vali_adr_vec, vali_solution_adr_vec,
         n_rows, n_cols, meta_R,  task_id_vec, algorithm,tasks_R = tasks_R,
         version='classic', sparse=False)



