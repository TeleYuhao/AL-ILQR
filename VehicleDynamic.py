'''
@Project :Ilqr
@Author : YuhaoDu
@Date : 2024/8/15 
'''
import casadi as ca
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
