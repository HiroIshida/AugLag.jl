import time
import numpy as np
import scipy.optimize
import json

def scipinize(fun):
    closure_member = {'jac_cache': None}

    def fun_scipinized(x):
        f, jac = fun(x)
        closure_member['jac_cache'] = jac
        return f

    def fun_scipinized_jac(x):
        return closure_member['jac_cache']
    return fun_scipinized, fun_scipinized_jac


with open("../data/mpc.json", "r") as io:
    jdata = json.load(io)

C_ineq1 = np.array(jdata["C_ineq1"])
C_ineq2 = np.array(jdata["C_ineq2"])
C_eq1 = np.array(jdata["C_eq1"])
C_eq2 = np.array(jdata["C_eq2"])
H = np.array(jdata["H"])
sol = np.array(jdata["sol"])

def objective(x):
    val = H.dot(x).dot(x)
    grad = 2 * H.dot(x)
    return val, grad

def eq_const(x):
    val = C_eq2 - C_eq1.dot(x)
    grad = -C_eq1
    return val, grad

def ineq_const(x):
    val = C_ineq2 - C_ineq1.dot(x)
    grad = -C_ineq1
    return val, grad

eq_const_scipy, eq_const_jac_scipy = scipinize(eq_const)
eq_dict = {'type': 'eq', 'fun': eq_const_scipy,
           'jac': eq_const_jac_scipy}
ineq_const_scipy, ineq_const_jac_scipy = scipinize(ineq_const)
ineq_dict = {'type': 'ineq', 'fun': ineq_const_scipy,
             'jac': ineq_const_jac_scipy}
f, jac = scipinize(objective)
dim = H.shape[0]
x_init = np.ones(dim)

slsqp_option = {'xtol': 1e-5, 'disp': True, 'maxiter': 100}

ts = time.time()
for i in range(1000):
    res = scipy.optimize.minimize(
        f, x_init, method='SLSQP', jac=jac,
        constraints=[eq_dict, ineq_dict],
        )
    print(res)
elapsed = time.time() - ts
print("average:", elapsed/1000.0)
