import tensorflow
import jax
jax.config.enable_omnistaging()
from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import ufl
from crikit import *
from crikit.fe import *
from crikit.fe_adjoint import *
import pyadjoint
from pyadjoint.tape import annotate_tape, get_working_tape, stop_annotating
from pyadjoint.overloaded_type import create_overloaded_object
from crikit.loss import SlicedWassersteinDistance
from crikit.observer import SubdomainObserver
import sys
import time
#to prevent stack overflows in JAX functions, which can have very deep recursion
sys.setrecursionlimit(100000)
import argparse
#from tqdm.auto import tqdm

class Experiment:
    def __init__(self, mesh, alpha=0, g=9.8, rho=1):
        self.mesh = mesh
        self.setDomainParameters(mesh,False)
        self.setF(alpha=alpha, g=g, rho=rho)

    def setF(self, alpha=None, g=None, rho=None):
        self.alpha = self.alpha if alpha is None else alpha
        self.g = self.g if g is None else g
        self.rho = self.rho if rho is None else rho
        #V = VectorFunctionSpace(mesh, "CG", 2)
        f = (-self.rho * self.g * sin(self.alpha), -self.rho * self.g * cos(self.alpha), 0)
        if self.dim == 2:
            f = f[:2]
        self.f = Constant(f, name='f')

    def setDomainParameters(self, smesh, rte=False):
        if rte:
            V = FiniteElement("RT",smesh.ufl_cell(),2)
            Q = FiniteElement("DG",smesh.ufl_cell(),1)
            self.dim = mesh.geometric_dimension()
            self.W = FunctionSpace(smesh,V * Q)
        else:
            V_e = VectorElement("CG", smesh.ufl_cell(), 2)
            Q_e = FiniteElement("CG", smesh.ufl_cell(), 1)
            VQ_e = V_e*Q_e
            self.W = FunctionSpace(smesh, VQ_e)
            self.dim = V_e.value_shape()[0]

        self.rte = rte

    def setBCs(self, dir_bcs, rob_bcs=None):
        """
        Sets Dirichlet and Robin boundary conditions. Each Robin condition is a
        tuple (a, b, j, ds) corresponding to the equation :math:`a u + b mu sym(grad(u)) n = j`
        on the boundary ds. Note that the Robin conditions are only applied
        to the velocity, not the pressure.

        Args:
            dir_bcs (DirichletBC or list[DirichletBC]): The Dirichlet boundary conditions.
            rob_bcs (tuple or list[tuple]): See description above.
        """
        self.bcs = Enlist(dir_bcs)
        self.h_bcs = homogenize_bcs(self.bcs)
        self.rob_bcs = Enlist(rob_bcs) if rob_bcs is not None else None

    def bc(self, u):
        """Applies Dirichlet boundary conditions to function or to matrix"""
        if hasattr(u, 'vector'):
            for bc in self.bcs:
                bc.apply(u.vector())
        else:
            for bc in self.bcs:
                bc.apply(u)
        return u

    def h_bc(self, u):
        """Applies homogenous Dirichlet boundary conditions to function or to matrix"""
        for bc in self.h_bcs:
            bc.apply(u.vector())
        return u

    def get_robin_terms(self, u, v):
        """Adds the Robin terms to the given Form"""
        if self.rob_bcs is None:
            return 0
        F = 0
        for a, b, j, ds in self.rob_bcs:
            F += inner(v, (a * u - j))/b * ds
        return F

    def run(self, cr, observer=None, initial_w=None, ufl=False, cback=None, solver_parameters=None, disp=True, quad_params=None):
        if solver_parameters is None:
            solver_parameters = {}

        # Define u.
        if initial_w is None:
            w = Function(self.W, name='w')
        else:
            w = initial_w

        if ufl:
            w = self._ufl_solve(cr, w, cback=cback, disp=disp)
        elif self.rte:
            w = self._rte_solve(cr, w, cback, disp)
        else:
            w = self._solve(cr, w, cback=cback, disp=disp, quad_params=quad_params)

        if observer is None:
            return w
        return observer(w)

    def _ufl_solve(self, cr, w, cback=None, disp=True):
        # Don't print out Newton iterations.
        if not disp:
            orig_log_level = get_log_level()
            set_log_level(LogLevel.CRITICAL)

        # Set up the weak form.
        u, p = split(w)
        sigma = cr(sym(grad(u)))
        # mu = cr(sym(grad(u)))
        # sigma = mu * sym(grad(u))
        F, u = self.get_form(w, sigma)
        params = {'nonlinear_solver' : 'snes', 'snes_solver' : {'line_search' : 'bt'}}
        solve(F == 0, w, self.bcs)#, solver_parameters=params)

        if not disp:
            set_log_level(orig_log_level)

        if cback is not None:
            cback(w)
        return w

    def _solve(self, cr, w, cback=None, disp=True, quad_params=None):
        # Set up the weak form.
        F, u, sigma = self.get_form_cr(cr, w)     
            
        with push_tape():
            residual = Function(self.W)
            assemble_with_cr(F, cr, sym(grad(u)), sigma, tensor=residual, quad_params=quad_params)

            wcontrol = Control(w)
            residual_rf = ReducedFunction(residual, wcontrol)
            
        reduced_equation = ReducedEquation(residual_rf, self.bc, self.h_bc)

        solver = SNESSolver(reduced_equation, {'jmat_type': 'assembled'})
        #get_working_tape().visualise_dot('tape.dot')
        w = solver.solve(wcontrol, disp=disp, cback=cback)
        return w


    def _rte_solve(self, cr, w, cback=None, disp=True):

        F, u, sigma = self.get_rte_form_cr(cr,w)

        with push_tape():
            res = Function(self.W)
            assemble_with_cr(F,cr,u,sigma,tensor=res,quad_params=quad_params)
            wcontrol = Control(w)
            res_rf = ReducedFunction(res,wcontrol)

        red_eq = ReducedEquation(res_rf,self.bc,self.h_bc)

        solver = SNESSolver(red_eq,{'jmat_type' : 'assembled'})
        w = solver.solve(wcontrol,disp=disp,cback=cback)
        return w

    def get_rte_form_cr(self, cr, w):
        target_shape = tuple(i for i in cr.target.shape() if i != -1)
        # mu = create_ufl_standins((target_shape,))[0]
        sigma = create_ufl_standins((target_shape,))[0]
        u, p = split(w)
        # sigma = mu * sym(grad(u))
        F, u = self.get_rte_form(w, sigma)
        return F, u, sigma

    def get_form_cr(self, cr, w):
        target_shape = tuple(i for i in cr.target.shape() if i != -1)
        # mu = create_ufl_standins((target_shape,))[0]
        sigma = create_ufl_standins((target_shape,))[0]
        u, p = split(w)
        # sigma = mu * sym(grad(u))
        F, u = self.get_form(w, sigma)
        return F, u, sigma


    def get_rte_form(self, w, sigma):
        u, p = split(w)
        w_test = TestFunction(self.W)
        v, q = split(w_test)
        a = (inner(dot(inv(sigma),u), v) -
             inner(p,div(v)) - inner(div(u),q)) * dx
        L = inner(self.f,v) * dx
        F = a - L + self.get_robin_terms(u,v)
        return F, u

    def get_form(self, w, sigma):
        u, p = split(w)
        w_test = TestFunction(self.W)
        v, q = split(w_test)

        lhs = (inner(sigma, grad(v)) * dx
               - inner(div(v), p) * dx
               - inner(div(u), q) * dx
              )
        rhs = inner(self.f, v) * dx
        F = lhs - rhs
        F = F + self.get_robin_terms(u, v)
        return F, u




logfile_name = 'form_invariant_opt.csv'#change this if desired
num_entries = -1 #changes to 0 when the file is opened
logfile = None
total_num_call = 0

def run_experiment(seed, std, loss):
    dims = 2 #must be 2 or 3
    grid_sizes = [5] * dims # N x N mesh in 2d, N x N x N mesh in 3d
    angle_rads = -np.pi / 6
    rho = 1
    g = 9.8
    p_true = 1.2 # true value of p. Must be > 1. Values in [1,2] require larger regularization than values in (2,\infty)
    epsilon = 0.1 # regularization -- if you get NaNs during the optimization process, this is probably too small
    fe_order = 2 # finite element order; must be >= 2
    robin_bcs = True #use Robin boundary conditions?
    robin_a = 1
    robin_b = 0.5
    cmap = 'magma'
    display_snes_iterations = False
    amesh = UnitSquareMesh(*grid_sizes)
    if dims == 2:
        amesh = UnitSquareMesh(*grid_sizes)
        inflow = Expression(("x[1]*(1-x[1])/2", "0"), degree=2)
        def side_boundary(x, on_boundary):
            return on_boundary and (near(x[1], 1, 1e-5) or near(x[1], 0, 1e-5))
    else:
        amesh = UnitCubeMesh(*grid_sizes)
        inflow = Expression(("x[1]*(1-x[1])/2 * x[2]*(1-x[2])/2", "0", "0"), degree=2)
        def side_boundary(x, on_boundary):
            return on_boundary and (near(x[1], 0, 1e-5) or near(x[1], 1, 1e-5)
                                or near(x[2], 0, 1e-5) or near(x[2], 1, 1e-5))
    
    def inflow_boundary(x, on_boundary):
        return on_boundary and near(x[0], 0, 1e-5)

    def outflow_boundary(x, on_boundary):
        return on_boundary and near(x[0], 1, 0.1)


    class OutflowBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return outflow_boundary(x, on_boundary)
    
        
    observer = SubdomainObserver(amesh, OutflowBoundary())
    noslip = ufl.zero(dims)
    eps2 = epsilon ** 2
    ex = Experiment(amesh, alpha=angle_rads, g=g)

    if robin_bcs:
        class InflowBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return inflow_boundary(x, on_boundary)
        
        class SideBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return side_boundary(x, on_boundary)
        
        markers = MeshFunction("size_t", amesh, amesh.topology().dim() - 1, 0)
        InflowBoundary().mark(markers, 1)
        SideBoundary().mark(markers, 2)
        dleft = ds(1, domain=amesh, subdomain_data=markers)
        dside = ds(2, domain=amesh, subdomain_data=markers)
        robin_a = Constant(robin_a)
        robin_b = Constant(robin_b)
        ex.setBCs([],
              [(robin_a, robin_b, inflow, dleft),
               (robin_a, robin_b, noslip, dside),
              ]
             )
        
    else:
        noslip = Expression(("0",)*dims, degree=0) #NOTE: do I actually need to do this?
        ex.setBCs([DirichletBC(ex.W.sub(0), noslip, side_boundary),
               DirichletBC(ex.W.sub(0), inflow, inflow_boundary)])
    
    quad_params = {'quadrature_degree' : fe_order + 1}
    domain = amesh.ufl_domain()
    #sets parameters for the crikit.covering module
    set_default_covering_params(domain=domain, quad_params=quad_params)
    
    cmap = cm.get_cmap(cmap)

    p = Constant(p_true, name='p')
    def true_cr(epsilon):
        scalar_invt = tr(dot(epsilon, epsilon))
        mu = (scalar_invt + eps2) ** ((p - 2) / 2)
        return mu * epsilon
    

    #generate the observations!
    with record_tape_block(name='Ground Truth Experiment'):
        w = ex.run(true_cr, ufl=True, observer=None, disp=display_snes_iterations)
        obs_w = observer(w)

        
    noise_maker = AdditiveRandomFunction(ex.W, std=std, seed=seed)
    noisy_w = noise_maker(w)

    theta = array([0.95, 1.15])

    def jax_form_invariant_cr_func(scalar_invts, theta):
        #a JAX-based CR that uses form invariants
        a, p = theta[0], theta[1]
        scale = a * (scalar_invts[1] + eps2) ** ((p - 2) / 2)
        return jnp.array([0, scale])

    def jax_energy_cr_func(scalar_invts, theta):
        #a JAX-based CR that calculates the strain energy and gets the stress by differentiating
        #it w.r.t. the input tensor
        a, p = theta[0], theta[1]
        return (a / p) * (scalar_invts[1] + eps2) ** (p / 2)

    input_types = (TensorType.make_symmetric(2, dims),)
    output_type = TensorType.make_symmetric(2,dims)#scalar()#
    form_invt_cr = CR(output_type, input_types,
                      jax_form_invariant_cr_func, params=(theta,))
    
    strain_energy_cr = CR(TensorType.make_scalar(), input_types, 
                      jax_energy_cr_func, params=(theta,),
                      strain_energy=True)

    with record_tape_block(name='JAX Form Invariant Experiment'):
        pred_w = ex.run(form_invt_cr, observer=None, ufl=False, disp=False, quad_params=quad_params)


    lname = loss
    loss = SlicedWassersteinDistance(ex.W, 30, jax.random.PRNGKey(seed), p=2) if loss == 'W2' else integral_loss

    err = loss(observer(noisy_w), observer(pred_w))
    print("Initial loss is ", err)
    Jhat = ReducedFunctional(err, Control(theta))
    
    def log_reset():
        global total_num_call, num_entries, logfile
        logfile.close()
        total_num_call = 0
        num_entries = -1

    def log_init(name):
        global num_entries, logfile
        if num_entries > -1:
            raise ValueError("Must set num_entries to -1 before initializing logging!")
    
        logfile = open(name, 'w+')
        num_entries = 0
        param_name_str = ','.join(['param_' + str(i) for i in range(theta.size)])
        grad_name_str = ','.join(['grad_param_' + str(i) for i in range(theta.size)])
        logfile.write('step,loss,' + param_name_str + ',' + grad_name_str + '\n')
    
    
    def log_entry(Jhat, params):
        global num_entries, logfile
        if num_entries < 0:
            raise Exception("Must initialize logging before logging an entry!")
        
        loss = Jhat.functional.block_variable.checkpoint
        param_str = ','.join(map(str, np.array(params.flatten())))
        grad = Jhat.controls[0].get_derivative()
        grad_str = ','.join(map(str, np.array(grad)))
        print(f"writing entry {str(num_entries) + ',' + str(loss) + ',' + param_str + ',' + grad_str}")
        logfile.write(str(num_entries) + ',' + str(loss) + ',' + param_str + ',' + grad_str + '\n')
        num_entries += 1

    def minimize_cb(x_k, state=None):
        log_entry(Jhat, x_k)
    
    
    def cb_pre(x):
        global total_num_call
        total_num_call += 1
        print("calls: ", total_num_call, " theta: ", x)
    
    h = 1e-3 * array(np.random.randn(*theta.shape))
    max_retry = 10
    retry = 0
    while pyadjoint.taylor_test(Jhat, theta, h) < 1.9 and retry < max_retry:
        #re-do the Taylor test if it fails, up to 10 times
        h = 1e-3 * array(np.random.randn(*theta.shape))
        retry += 1

    Jhat.eval_cb_pre = cb_pre
    bounds = [array([0.9, 1.05]), array([1.05, 1.5])]
    method = 'L-BFGS-B'
    log_init(lname + '_outflow' + method + '_std' + str(std) + '_' + 'seed_' + str(seed) + '_' + logfile_name)
    map_estimate = minimize(Jhat, method=method, callback=minimize_cb, bounds=bounds, options={'disp' : True})
    print("Total number of function evaluations: ", total_num_call)
    log_reset()
    print("MAP estimate is ", map_estimate)
    print(seed)


if __name__ == '__main__':

    seedlist = [1614458923, 1614459127, 1614459385, 1614459526, 1614459864,
                1614461636, 1614461791, 1614461944, 1614462881, 1614463230,
                1614463501, 1614463969, 1614464512, 1614464689, 1614464887]
    #subtracting "random" number generated by keyboard-mashing
    second_seedlist = [2 * x - 153245 for x in seedlist]

    parser = argparse.ArgumentParser()
    parser.add_argument('-stds', type=float, nargs='+', help='Noise stds to use')
    parser.add_argument('-second_list', action='store_true', default=False)
    losses = ['W2', 'L2']

    args = parser.parse_args()

    for std in args.stds:
        if args.second_list:
            for seed in second_seedlist:
                for loss in losses:
                    run_experiment(seed, std, loss)
        else:
            for seed in seedlist:
                for loss in losses:
                    run_experiment(seed, std, loss)
            
        
