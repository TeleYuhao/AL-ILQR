import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import ReedsSheppPathPlanning as rs


class VehicleDynamic:
    def __init__(self, wheel_base):
        self.wheel_base = wheel_base
        self.make_variable()
        self.BicycleModelRk3()
        self.Jacobian()
        self.Hessian()
    def make_variable(self):
        '''
        Build the Variable to be used and generate the init Dynamic Function
        @return:
        '''
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        theta = ca.SX.sym('theta')
        v = ca.SX.sym('v')
        steering = ca.SX.sym('steering')
        a = ca.SX.sym('a')
        self.t = 0.1
        steering_rate = ca.SX.sym('steering_rate')
        # state:[x,y,v,\theta, steering]
        self.state = ca.vertcat(x, y, theta, v)
        # control:[acc, delta_f]
        self.control = ca.vertcat(steering, a)
        beta = ca.atan(2 * ca.tan(steering) / self.wheel_base)
        rhs = ca.vertcat(v * ca.cos(theta), v * ca.sin(theta), v / self.wheel_base * ca.tan(steering), a)
        self.Dynamic_Func = ca.Function('f', [self.state, self.control], [rhs])

    def BicycleModelRk3(self):
        '''
        use the Runge-Kutta Method to generate the Vehicle Model
        @return:
        '''
        k1 = self.Dynamic_Func(self.state, self.control) * self.t
        k2 = self.Dynamic_Func(self.state + 0.5 * k1, self.control) * self.t
        k3 = self.Dynamic_Func(self.state - k1 + 2 * k2, self.control) * self.t
        self.RK3 = self.state + (k1 + 4 * k2 + k3) / 6
        self.RK3Function = ca.Function('rk3', [self.state, self.control], [self.RK3])

    def Jacobian(self):
        '''
        Calculate the jacobian of vehicle Dynamic
        @return:
        '''
        self.dfdx = ca.jacobian(self.RK3, self.state)
        self.dfdx_func = ca.Function("dfdx",[self.state,self.control],[self.dfdx])
        self.dfdu = ca.jacobian(self.RK3, self.control)
        self.dfdu_func = ca.Function("dfdx", [self.state, self.control], [self.dfdu])

    def Hessian(self):
        '''
        Calculate the Hessian of Vehicle Dynamic
        @return:
        '''
        self.dfddx = ca.jacobian(self.dfdx, self.state)
        self.dfdxdu = ca.jacobian(self.dfdx, self.control)
        self.dfddu = ca.jacobian(self.dfdu, self.control)



class CostFunc:
    def __init__(self, ref_path, Q , R, Q_Terminal,SoftConstrain = []):
        self.ref_path = ref_path
        self.Q = Q
        self.R = R
        self.Q_Terminal = Q_Terminal

        self.Constrain = SoftConstrain
        self.SoftConstrainFormular()
        self.StateCostFunc()
        self.CalcJacobian()
        self.CalcHessian()
        self.CalcTerminalFunc()
    def SoftConstrainFormular(self):
        vio = ca.SX.sym("vio")
        self.SoftConstrain = ca.Function( "Constrain",[vio],[0.01 *ca.exp(-10 * vio)])
    def StateCostFunc(self):
        '''
        use to generate the State Cost Function and it's Terminal Cost Function
        @return:
        '''
        x_ref = ca.SX.sym('x_ref')
        y_ref = ca.SX.sym('y_ref')
        yaw_ref = ca.SX.sym('yaw_ref')
        v_ref = ca.SX.sym('v_ref')
        self.state_ref = ca.vertcat(x_ref, y_ref, yaw_ref,v_ref)

        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        theta = ca.SX.sym('theta')
        v = ca.SX.sym('v')
        steering = ca.SX.sym('steering')
        a = ca.SX.sym('a')
        self.state = ca.vertcat(x,y,theta,v)
        self.control = ca.vertcat(steering,a)

        state_diff = self.state - self.state_ref

        self.StageCost = state_diff.T @ self.Q @ state_diff
        self.StageCost += self.control.T @ self.R @ self.control
        for con in self.Constrain:
            self.StageCost += self.SoftConstrain(con)

        self.TerminalCost = state_diff.T @ self.Q_Terminal @ state_diff

        self.StageCostFunc = ca.Function("StageCost",[self.state_ref,self.state,self.control],[self.StageCost])
        self.TerminalCostFunc = ca.Function("TerminalCost",[self.state_ref,self.state],[self.TerminalCost])
    def CalcCost(self,State,Control):
        '''
        function : use to calculate the trajectory Cost
        @param State: the vehicle state ,the format is [x ,y ,yaw ,velocity ]
        @param Control:the control input of vehicle the format is [steering,a]
        @return:
        '''
        Cost = 0
        self.StageCostFunction = 0
        for i in range(len(self.ref_path) - 1):
            Cost += self.StageCostFunc(self.ref_path[i],State[i], Control[i])
            self.StageCostFunction += self.StageCostFunc(self.ref_path[i],self.state,self.control)
            # print(Cost,i)
        Cost += self.TerminalCostFunc(self.ref_path[-1],State[-1])
        # print(Cost)
        return Cost
    def CalcJacobian(self):
        '''
        Calculate the Jacobian of Cost Function
        @return: None
        '''
        self.lx = ca.jacobian(self.StageCost,self.state)
        self.lu = ca.jacobian(self.StageCost,self.control)
        self.lx_fun = ca.Function("lx",[self.state_ref,self.state,self.control],[self.lx])
        self.lu_fun = ca.Function("lu",[self.state_ref,self.state,self.control],[self.lu])
    def CalcHessian(self):
        '''
        Calculate the Hessian of Cost Function
        @return:
        '''
        self.lxx_fun = ca.Function("lxx",[self.state_ref,self.state,self.control],[ca.jacobian(self.lx,self.state)])
        self.lux_fun = ca.Function("lux",[self.state_ref,self.state,self.control],[ca.jacobian(self.lu,self.state)])
        self.luu_fun = ca.Function("luu",[self.state_ref,self.state,self.control],[ca.jacobian(self.lu,self.control)])

    def CalcTerminalFunc(self):
        '''
        Calculate the Jacobian and Hessian of Terminal Cost Function
        @return:
        '''
        p = ca.jacobian(self.TerminalCost,self.state)
        self.p_fun = ca.Function("p",[self.state_ref,self.state],[p])
        self.P_fun = ca.Function("P",[self.state_ref,self.state],[ca.jacobian(p,self.state)])








def GetRsPathCost(path):
    path_cost = 0.0
    for length in path.lengths:
        if length >= 0.0:
            path_cost += length
        else:
            path_cost += abs(length)
    return path_cost


# This is for a atest
# V = VehicleDynamic(2.84)
# start_pose = np.array([30, 10, np.deg2rad(0.0), 0.0])
# start_control = np.array([1,0.4])
# res = V.RK3Function(start_pose,start_control)
# print(res)

if __name__ == '__main__':
    Vehicle = VehicleDynamic(2.84)
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
    Q = np.diag((1,1,1,0))
    R = np.eye(2)
    Q_Terminal = np.diag((1,1,1,0)) * 100
    Cost = CostFunc(ref_path,Q,R,Q_Terminal)
    C = Cost.CalcCost(x0,u0)
    print(C)
    plt.plot(ref_path[:,0], ref_path[:,1])
    plt.show()
