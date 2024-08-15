from DynamicCost import *

class Ilqr:
    def __init__(self,VehicleDynamic:VehicleDynamic,CostFunc:CostFunc, ref_state):
        self.VehicleDynamic = VehicleDynamic
        self.CostFunc = CostFunc
        self.ref_state = ref_state
        self.step = len(ref_state) - 1

        self.max_iter = 50
        self.line_search_beta_1 = -1e-4
        self.line_search_beta_2 = 10
        self.line_search_gamma = 0.5
        self.J_tolerance = 1e-2
    def Evaluate(self,x,u):
        return self.CostFunc.CalcCost(x,u)
    def BackwardPass(self,x,u):
        p = self.CostFunc.p_fun(self.ref_state[-1],x[-1])
        P = self.CostFunc.P_fun(self.ref_state[-1],x[-1])

        k = [None] * self.step
        d = [None] * self.step

        Qu_list = [None] * self.step
        Quu_list = [None] * self.step

        for i in reversed(range(self.step)):
            dfdx = self.VehicleDynamic.dfdx_func(x[i],u[i])
            dfdu = self.VehicleDynamic.dfdu_func(x[i],u[i])
            lx   = self.CostFunc.lx_fun(self.ref_state[i],x[i],u[i])
            lu   = self.CostFunc.lu_fun(self.ref_state[i],x[i],u[i])
            lxx  = self.CostFunc.lxx_fun(self.ref_state[i],x[i],u[i])
            lux  = self.CostFunc.lux_fun(self.ref_state[i],x[i],u[i])
            luu  = self.CostFunc.luu_fun(self.ref_state[i],x[i],u[i])

            Qx   = lx + p @ dfdx
            Qu   = lu + p @ dfdu
            Qxx  = lxx + dfdx.T @ P @ dfdx
            Qux  = lux + dfdu.T @ P @ dfdx
            Quu  = luu + dfdu.T @ P @ dfdu

            Quu_inverse = regularized_persudo_inverse(Quu)
            # Quu_inverse = np.linalg.inv(Quu)
            Quu_list[i] = Quu
            Qu_list[i]  = Qu

            k[i] = - Quu_inverse @ Qux
            d[i] = - Quu_inverse @ Qu.T

            p = Qx  + d[i].T @ Quu @ k[i] + d[i].T @ Qux + Qu @ k[i]
            P = Qxx + k[i].T @ Quu @ k[i] + Qux.T @ k[i] + k[i].T @ Qux

        return k,d,Qu_list,Quu_list
    def ForWardPass(self,x,u,k,d,alpha,Qu_list,Quu_list):
        x_new = []
        u_new = []
        x_new.append(x[0])
        delta_J = 0.0
        for i in range(self.step):
            u_new.append(u[i] + k[i] @ (x_new[i]-x[i]) + alpha*d[i])
            x_new.append(self.VehicleDynamic.RK3Function(x_new[i],u_new[i]))

            delta_J += alpha * ( Qu_list[i] @ d[i]) + 0.5 * pow(alpha,2) * (d[i].T @ Quu_list[i] @ d[i])
        delta_x_terminal = x_new[-1] - x[-1]
        delta_J += (delta_x_terminal.T @ self.CostFunc.P_fun(self.ref_state[-1],x[-1]) @ delta_x_terminal +
                    self.CostFunc.p_fun(self.ref_state[-1],x[-1]) @delta_x_terminal)
        J = self.Evaluate(x_new,u_new)
        return x_new,u_new,J,delta_J
    def init_trajectory(self,x,u):
        x_init = []
        x_init.append(x[0])
        for i in range(self.step):
            x_init.append(self.VehicleDynamic.RK3Function(x_init[-1],u[i]))
        return x_init

    def Solve(self,x_init,u_init):
        u = u_init
        x = self.init_trajectory(x_init,u)
        J_opt = self.Evaluate(x,u)
        J_hist = [J_opt]
        x_hist = [x]
        u_hist = [u]
        iter = 0
        coveraged = False

        while not coveraged:
            print("ILQR: New iteration {0} starts ...".format(iter))
            if iter >= 500:
                print("ILQR: Reach the maximum iteration number")
                break
            k,d,Qu_list,Quu_list = self.BackwardPass(x,u)
            alpha = 1
            J_new = 0
            accept = False
            while not accept:
                # if alpha < 1e-6:
                #     print("ILQR:Line search fail to decrease cost function")
                x_new, u_new, J_new, delta_J = self.ForWardPass(
                                    x, u, k, d, alpha, Qu_list, Quu_list)
                z = (J_opt - J_new) / -delta_J
                print("ILQR: J_opt:{0} J_new:{1} delta_J:{2} z:{3}".format(
                    J_opt, J_new, delta_J, z))
                if z > self.line_search_beta_1 and z < self.line_search_beta_2:
                    x = x_new
                    u = u_new
                    accept = True
                alpha *= self.line_search_gamma
            iter += 1
            J_hist.append(J_opt)
            x_hist.append(x)
            u_hist.append(u)
            if accept:
                if abs(J_opt - J_new)/J_opt < self.J_tolerance:
                    coveraged = True
                    print('ILQR:Converged at iteration {0}; J={1};'.format(iter, J_opt))
                J_opt = J_new

        for i in range(1,len(x_hist[-1])):
            x_hist[-1][i] = x_hist[-1][i].full().T.flatten()
        res_dict = {'x_hist': x_hist, 'u_hist': u_hist, 'J_hist': J_hist}
        print("============== ILQR ends ==============")
        return res_dict







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
    import time
    start_time = time.time()
    Solver = Ilqr(Vehicle,Cost,ref_path)
    res = Solver.Solve(x0,u0)
    print("Ilqr solve time ",time.time() - start_time)
    # k, d, Qu_list, Quu_list = Solver.BackwardPass(x0,u0)
    # Solver.ForWardPass(x0,u0,k,d,1,Qu_list,Quu_list)
    x_opt = res['x_hist'][-1]
    u_opt = res['u_hist'][-1]
    x = []
    y = []
    u = []
    for i in range(len(x_opt)):
        x.append(x_opt[i][0])
        y.append(x_opt[i][1])
    for i in range(len(x_opt)-1):
        u.append(u_opt[i].full().flatten())
    u = np.array(u)
    C = Cost.CalcCost(x0,u0)


    plt.plot(ref_path[:,0], ref_path[:,1])
    plt.plot(x,y)
    plt.show()

    x = [0, 100]
    y = [-0.5, -0.5]
    u = np.array(u)
    plt.plot(u[:, 0])
    plt.plot(u[:, 1])
    plt.plot(x, y)
    plt.show()
