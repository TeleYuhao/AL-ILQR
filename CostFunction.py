'''
@Project :own
@Author : YuhaoDu
@Date : 2024/8/15 
'''
import casadi as ca

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