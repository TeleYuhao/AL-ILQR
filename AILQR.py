# from DynamicCost import *
from CostFunction import *
from VehicleDynamic import VehicleDynamic
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import ReedsSheppPathPlanning as rs

class ConstrintFunc:
    def __init__(self,Constrint:list,State,Control):
        self.Constrint = Constrint
        self.State = State
        self.Control = Control
        self.mu = ca.SX.sym("mu",len(Constrint))
        self.I_mu = ca.diag(self.mu)
        self.Lambda = ca.SX.sym("lambda",len(Constrint))


        self.ConstraintFunction()
        self.ALMTerm()
    def ConstraintFunction(self):
        # transform Constraint cost into max(constraint term,0)
        for i in range(len(self.Constrint)):
            self.Constrint[i] = ca.mmax(ca.vertcat(self.Constrint[i],0))

        # the formular of constraint
        self.ConstrintCostFormular = ca.vertcat(*self.Constrint)

        # the calculate function of constraint
        self.ConstrintCalFunction = ca.Function("constraintFunction",[self.State,self.Control],[self.ConstrintCostFormular])

        Cx = ca.jacobian(self.ConstrintCostFormular,self.State)
        Cu = ca.jacobian(self.ConstrintCostFormular,self.Control)

        self.CxFunc = ca.Function("Cx",[self.State,self.Control],[Cx])
        self.CuFunc = ca.Function("Cu",[self.State,self.Control],[Cu])
    def ALMTerm(self):
        # get the Augment Langrange Term using second order Cost Term
        AugmentTerm = 0
        for i in range(len(self.Constrint)):
            AugmentTerm += 0.5 * self.Lambda[i] * ca.power(ca.mmax(ca.vertcat(self.mu[i] / self.Lambda[i] + self.Constrint[i], 0)),
                                                      2) - ca.power(self.mu[0] / self.Lambda[i], 2)
        cx = ca.jacobian(AugmentTerm,self.State)
        cu = ca.jacobian(AugmentTerm,self.Control)
        cxx = ca.jacobian(cx ,self.State)
        cux = ca.jacobian(cu ,self.State)
        cuu = ca.jacobian(cu ,self.Control)
        # Get the Augment Cost function
        self.AugmentFunction = ca.Function("ALM_Function",[self.State,self.Control,self.mu,self.Lambda],[AugmentTerm])
        # Get the jacobian function of Augment Cost
        self.CxFunc_ = ca.Function("Cx",[self.State,self.Control,self.mu,self.Lambda],[cx])
        self.CuFunc_ = ca.Function("Cx",[self.State,self.Control,self.mu,self.Lambda],[cu])
        # Get the Hessian function of Augment Cost
        self.CxxFunc_ = ca.Function("Cx",[self.State,self.Control,self.mu,self.Lambda],[cxx])
        self.CuxFunc_ = ca.Function("Cx",[self.State,self.Control,self.mu,self.Lambda],[cux])
        self.CuuFunc_ = ca.Function("Cx",[self.State,self.Control,self.mu,self.Lambda],[cuu])




class ALILQR:
    def __init__(self,VehicleDynamic:VehicleDynamic,CostFunc:CostFunc,ConstraintFunc:ConstrintFunc, ref_state):
        '''

        @param VehicleDynamic: the dynamic transfer function
        @param CostFunc:  the cost function of vehicle system
        @param ConstraintFunc: the constraint function of the function
        @param ref_state: the reference state of vehicle
        @param step:
        '''
        self.VehicleDynamic = VehicleDynamic
        self.CostFunc = CostFunc
        self.ConstraintFunc = ConstraintFunc
        self.ref_state = ref_state
        self.step = len(ref_state) - 1
        # initialize the multiplier
        self.Sigma = [ca.DM.ones(len(self.ConstraintFunc.Constrint)) for _ in range(self.step)]
        # initialize the penality term
        self.Mu     = [ca.DM.zeros(len(self.ConstraintFunc.Constrint)) for _ in range(self.step)]

        self.max_iter = 50
        self.line_search_beta_1 = -1e-4
        self.line_search_beta_2 = 10
        self.line_search_gamma = 0.5
        self.J_tolerance = 1e-2
        self.ConstraintTolerance = 1e-4
        # self.ConstraintTolerance = 1

        self.MuFactor = 1e1
        self.MuMax = 1e6

    def Evalueate(self,x,u):
        '''
        function : Evaluate the total trajectory
        @param x: the state of vehicle
        @param u: the control of vehicle
        @return:
        '''
        J = self.CostFunc.CalcCost(x,u)
        for i in range(self.step):
            # J += self.ConstraintFunc.AugmentPenelityFunction(x[i],u[i],self.I_mu[i],self.Lambda[i])
            J += self.ConstraintFunc.AugmentFunction(x[i],u[i],self.Mu[i],self.Sigma[i])
        return J

    def init_trajectory(self,x,u):
        '''
        function : roll out
        @param x: the state of vehicle
        @param u: the control input of vehicle
        @return:
        '''
        x_init = []
        x_init.append(x[0])
        for i in range(self.step):
            x_init.append(self.VehicleDynamic.RK3Function(x_init[-1],u[i]))
        return x_init
    def CalcConstraintViolation(self,x,u):
        '''
        @function: calculate the Constraint of Violation
        @param x: the state input
        @param u: the control input
        @return: the Constraint Violation
        '''
        Vio = 0
        for i in range(len(u)):
            Vio += self.ConstraintFunc.ConstrintCalFunction(x[i],u[i])
        return ca.sum1(Vio)
    def CalcMaxVio(self,x,u):
        '''
        function: update the constraint violation term
        @param x:
        @param u:
        @return:
        '''
        res = -1000
        for i in range(len(u)):
            Vio = self.ConstraintFunc.ConstrintCalFunction(x[i],u[i])
            maxvio = ca.mmax(Vio)
            res = max(res,maxvio)
        return res

    def UpdatePenalityParam(self,x,u):
        '''
        function: update the langrange multiplier and penality term
        @param x:the state of vehicle
        @param u:the control of vehicle
        @return:
        '''
        for i in range(self.step):
            Vio = self.ConstraintFunc.ConstrintCalFunction(x[i],u[i])
            for j in range(len(self.ConstraintFunc.Constrint)):
                # update mu : max(u_i_j + \sigma*C(x,u),0)
                self.Mu[i][j] = max(0,self.Mu[i][j]+Vio[j])
                # update lambda : \lambda
                self.Sigma[i][j] *= self.MuFactor


    def BackWardPass(self,x,u):
        '''
        the BackWard Process in ILQR
        @param x: the state input
        @param u: the control input
        @return: k,d,Qu_list,Quu_list
        '''
        p = self.CostFunc.p_fun(self.ref_state[-1], x[-1])
        P = self.CostFunc.P_fun(self.ref_state[-1], x[-1])

        k = [None] * self.step
        d = [None] * self.step

        Qu_list = [None] * self.step
        Quu_list = [None] * self.step
        for i in reversed(range(self.step)):
            # \frac{\partial f}{\partial  x}
            dfdx = self.VehicleDynamic.dfdx_func(x[i], u[i])
            # \frac{\partial f}{\partial u}
            dfdu = self.VehicleDynamic.dfdu_func(x[i], u[i])

            lx = self.CostFunc.lx_fun(self.ref_state[i], x[i], u[i])
            lu = self.CostFunc.lu_fun(self.ref_state[i], x[i], u[i])
            lxx = self.CostFunc.lxx_fun(self.ref_state[i], x[i], u[i])
            lux = self.CostFunc.lux_fun(self.ref_state[i], x[i], u[i])
            luu = self.CostFunc.luu_fun(self.ref_state[i], x[i], u[i])

            cx = self.ConstraintFunc.CxFunc_(x[i],u[i],self.Mu[i],self.Sigma[i])
            cu = self.ConstraintFunc.CuFunc_(x[i],u[i],self.Mu[i],self.Sigma[i])
            cxx = self.ConstraintFunc.CxxFunc_(x[i],u[i],self.Mu[i],self.Sigma[i])
            cux = self.ConstraintFunc.CuxFunc_(x[i],u[i],self.Mu[i],self.Sigma[i])
            cuu = self.ConstraintFunc.CuuFunc_(x[i],u[i],self.Mu[i],self.Sigma[i])

            Qx = lx + p @ dfdx + cx
            Qu = lu + p @ dfdu + cu
            Qxx = lxx + dfdx.T @ P @ dfdx + cxx
            Qux = lux + dfdu.T @ P @ dfdx + cux
            Quu = luu + dfdu.T @ P @ dfdu + cuu

            Quu_inverse = regularized_persudo_inverse(Quu)
            Quu_list[i] = Quu
            Qu_list[i]  = Qu

            k[i] = - Quu_inverse @ Qux
            d[i] = - Quu_inverse @ Qu.T

            p = Qx  + d[i].T @ Quu @ k[i] + d[i].T @ Qux + Qu @ k[i]
            P = Qxx + k[i].T @ Quu @ k[i] + Qux.T @ k[i] + k[i].T @ Qux
        return k, d, Qu_list, Quu_list
    def ForWardPass(self,x,u,k,d,alpha,Qu_list,Quu_list):
        '''
        the ForWard process in ILQR
        @param x: the state input
        @param u: the control input
        @param k: the back term
        @param d: the back term
        @param alpha: line search item
        @param Qu_list: the list of Qu
        @param Quu_list: the list of Quu
        @return:
        '''
        x_new = []
        u_new = []
        x_new.append(x[0])
        delta_J = 0.0
        for i in range(self.step):
            u_new.append(u[i] + k[i] @ (x_new[i]-x[i]) + alpha*d[i])
            x_new.append(self.VehicleDynamic.RK3Function(x_new[i],u_new[i]))
            # \delta J += \alpha \times (Q_{u}d+\frac{1}{2}\alpha ^2(d_i^TQ_{uu}d_i))
            delta_J += alpha * ( Qu_list[i] @ d[i]) + 0.5 * pow(alpha,2) * (d[i].T @ Quu_list[i] @ d[i])

        delta_x_terminal = x_new[-1] - x[-1]
        delta_J += (delta_x_terminal.T @ self.CostFunc.P_fun(self.ref_state[-1],x[-1]) @ delta_x_terminal +
                    self.CostFunc.p_fun(self.ref_state[-1],x[-1]) @delta_x_terminal)

        J = self.Evalueate(x_new,u_new)
        return x_new,u_new,J,delta_J

    def Solve(self,x_init,u_init):
        '''
        using this function to solve the nolinear planning problem
        @param x_init: the init vehicle state
        @param u_init: the init control input of vehicle
        @return:
        '''
        print("============== AL-ILQR starts ==============")
        # Init trajectory and control input
        u = u_init
        x = self.init_trajectory(x_init, u)

        x_hist = []
        u_hist = []
        ALILQR_iter = 0
        while True:
            print(
                "ALILQR: New al-ilqr iteration {0} starts ...".format(ALILQR_iter))
            if ALILQR_iter >= self.max_iter:
                print("ALILQR: Reach ilqr maximum iteration number")
                break
            J_opt = self.Evalueate(x, u)
            # ilqr Main loop
            ilqr_iter = 0
            converged = False
            while not converged:
                print(
                    "ALILQR: New ilqr iteration {0} starts ...".format(ilqr_iter))
                if ilqr_iter >= self.max_iter:
                    print("ALILQR: Reach ilqr maximum iteration number")
                    break
                # Backward pass
                K, k, Qu_list, Quu_list = self.BackWardPass(x, u)
                # Line search
                alpha = 1.0
                J_new = 0.0
                accept = False

                while not accept:
                    if alpha < 1e-6:
                        print("ALILQR: Line search fail to decrease cost function")
                    # Forward pass
                    x_new, u_new, J_new, delta_J = self.ForWardPass(
                        x, u, K, k, alpha, Qu_list, Quu_list)
                    z = (J_opt - J_new) / -delta_J
                    print("ALILQR: J_opt:{0} J_new:{1} delta_J:{2} z:{3}".format(
                        J_opt, J_new, delta_J, z))
                    if ((J_opt - J_new)/J_opt < self.J_tolerance and (J_opt - J_new)/J_opt > 0.0) or z > self.line_search_beta_1 and z < self.line_search_beta_2:
                        x = x_new
                        u = u_new
                        accept = True
                    alpha *= self.line_search_gamma
                if accept:
                    if abs(J_opt - J_new)/J_opt < self.J_tolerance:
                        converged = True
                    J_opt = J_new
                    x_hist.append(x)
                    u_hist.append(u)
                ilqr_iter += 1
            # ConstraintViolation = self.CalcConstraintViolation(x_hist[-1],u_hist[-1])
            ConstraintViolation = self.CalcMaxVio(x_hist[-1],u_hist[-1])
            print("ALILQR: New al-ilqr iteration {0} ends ... constraint violation: {1}".format(
                ALILQR_iter, ConstraintViolation))
            if ConstraintViolation < self.ConstraintTolerance:
                break
            self.UpdatePenalityParam(x,u)
            ALILQR_iter += 1

        res_dict = {'x_hist': x_hist, 'u_hist': u_hist}
        print("============== AL-ILQR ends ==============")
        return res_dict


import math
def CalculatePathCurvature(x, y):
    path_curvature = []
    for i in range(1, len(x) - 1):
        path_curvature.append(CalculateCurvature(
            x[i-1], x[i], x[i+1], y[i-1], y[i], y[i+1]))
    return path_curvature
def CalculateCurvature(xi_minus_1, xi, xi_plus_1, yi_minus_1, yi, yi_plus_1):
    # Menger curvature
    tri_area = 0.5 * ((xi - xi_minus_1) * (yi_plus_1 - yi_minus_1) -
                      (yi - yi_minus_1) * (xi_plus_1 - xi_minus_1))
    if abs(tri_area) < 1e-8:
        return 0.0

    kappa = (4.0 * tri_area) / (math.hypot(xi_minus_1 - xi, yi_minus_1 - yi)
                                * math.hypot(xi - xi_plus_1, yi - yi_plus_1)
                                * math.hypot(xi_plus_1 - xi_minus_1, yi_plus_1 - yi_minus_1))
    return kappa
def regularized_persudo_inverse(mat, reg=1e-5):
    u, s, v = np.linalg.svd(mat)
    for i in range(len(s)):
        if s[i] < 0:
            s.at[i].set(0.0)
            print("Warning: inverse operator singularity{0}".format(i))
    diag_s_inv = np.diag(1. / (s + reg))
    return ca.DM(v.dot(diag_s_inv).dot(u.T))
if __name__ == '__main__':
    start_pose = [30, 10, np.deg2rad(0.0)]
    goal_pose = [40, 7, np.deg2rad(0.0)]

    max_steer = 0.5
    wheel_base = 2.84
    # wheel_base = 1
    step_size = 0.1
    max_curvature = np.tan(max_steer) / wheel_base
    rs_paths = rs.calc_paths(start_pose[0], start_pose[1], start_pose[2], goal_pose[0],
                             goal_pose[1], goal_pose[2], max_curvature, step_size)
    best_rs_path = None
    best_rs_cost = None
    for path in rs_paths:
        cost = GetRsPathCost(path)
        if not best_rs_cost or cost < best_rs_cost:
            best_rs_cost = cost
            best_rs_path = path

    # x0 = [30, 10, np.deg2rad(0.0), 0.0] * (len(best_rs_path.x) - 1)

    u0 = [np.array([0.0, 0.0])] * (len(best_rs_path.x) - 1)
    x0 = [np.array([30, 10, 0.0 , 0.0])] * (len(best_rs_path.x) - 1)


    ref_path = np.vstack([best_rs_path.x,best_rs_path.y,best_rs_path.yaw,np.zeros(len(best_rs_path.x))]).T
    # plt.plot(best_rs_path.x, best_rs_path.y)

    Q = np.diag((1, 1, 1, 0))
    R = np.eye(2)
    Q_Terminal = np.diag((1, 1, 1, 0)) * 100
    Vehicle = VehicleDynamic(2.84)
    Cost = CostFunc(ref_path,Q,R,Q_Terminal)
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    theta = ca.SX.sym('theta')
    v = ca.SX.sym('v')
    steering = ca.SX.sym('steering')
    a = ca.SX.sym('a')
    t = ca.SX.sym('t')
    steering_rate = ca.SX.sym('steering_rate')

    state = ca.vertcat(x, y, theta, v)
    control = ca.vertcat(steering, a)

    max_steer = 0.5
    # max_steer = 5
    max_acc = 1.0
    # max_acc = 10
    constraint = []
    constraint.append(a - max_acc)
    constraint.append(-a - max_acc)
    constraint.append(steering - max_steer)
    constraint.append(-steering - max_steer)


    Constraint = ConstrintFunc(constraint,state,control)
    import  time
    time_start = time.time()
    solver = ALILQR(Vehicle,Cost,Constraint,ref_path)
    res = solver.Solve(x0,u0)
    x_opt = res['x_hist'][-1]
    u_opt = res['u_hist'][-1]
    print('success,using time(s):',time.time()-time_start)


    State = []
    Control = []
    for i in range(1,len(x_opt)):
        State.append(x_opt[i].full().flatten())
        Control.append(u_opt[i-1].full().flatten())
    State = np.array(State)
    Control = np.array(Control)

    fig, ax = plt.subplots()
    # ellipse = Ellipse(xy=(10,0), width=2.7, height=1.2, angle=0, edgecolor='b', facecolor='none')
    # ax.add_patch(ellipse)
    ax.plot(ref_path[:,0],ref_path[:,1])
    ax.plot(State[:,0],State[:,1])
    plt.show()

    plt.plot(CalculatePathCurvature(
        State[:,0], State[:,1]))
    plt.plot(CalculatePathCurvature(ref_path[:,0],ref_path[:,1]))
    plt.show()

    plt.figure(num=3)
    plt.subplot(411)
    # plt.plot(yaw_ilqr_list)
    plt.plot(State[:,2])
    plt.legend(["ILQR", "AL-ILQR"])
    plt.grid()
    plt.title("Yaw")

    plt.subplot(412)
    # plt.plot(v_ilqr_list)
    plt.plot(State[:,3])
    plt.legend(["ILQR", "AL-ILQR"])
    plt.grid()
    plt.title("Velocity")

    plt.subplot(413)
    # plt.plot(delta_ilqr_list)
    plt.plot(Control[:,0])
    plt.plot([max_steer] * len(Control), 'r')
    plt.plot([-max_steer] * len(Control), 'r')
    # plt.legend(["ILQR", "AL-ILQR", "Max Delta", "Min Delta"])
    plt.legend([ "AL-ILQR", "Max Delta", "Min Delta"])
    plt.grid()
    plt.title("Delta")

    plt.subplot(414)
    # plt.plot(acc_ilqr_list)
    plt.plot(Control[:,1])
    plt.plot([max_acc] * len(Control), 'r')
    plt.plot([-max_acc] * len(Control), 'r')
    plt.legend([ "AL-ILQR", "Max Acc", "Min Acc"])
    plt.grid()
    plt.title("Acceleration")

    plt.show()