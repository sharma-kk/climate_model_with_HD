import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMPI_MCA_btl_openib_warn_no_device_params_found"] = "0"
import time
import math
from functions_barycentric_mesh import *
N = 50
mesh = UnitSquareMesh(N,N)
bmh = BaryMeshHierarchy(mesh, 0)
M = bmh[-1]

V_1 = VectorFunctionSpace(M, "CG", 2)
V_1_out = VectorFunctionSpace(M, "CG", 1)
V_2 = FunctionSpace(M, "CG", 1)
V_3 = FunctionSpace(M, "DG", 1)
V_4 = FunctionSpace(M, 'CG', 2)

Z = V_1*V_2*V_1*V_3*V_2*V_4

w = Function(Z)
ua,Ta,uo,p,To,q = split(w)
va,phi_a,vo,eta,phi_o,psi = TestFunctions(Z)
ua_ = Function(V_1)
uo_ = Function(V_1)

q_ = Function(V_4)
psi_ = TestFunction(V_4)

Dt = 0.1*(1/N) # CFL condition
half = Constant(0.5)

x,y = SpatialCoordinate(M)


# dimensionless constants for atmosphere
Ro_a = Constant(1) # Rossby number
Re_a = Constant(10) # Reynolds number
Pe_a = Constant(10) # Peclet number
C_a = Constant(1/3000)

# dimensionless constants for ocean
Ro_o = Constant(1) # Rossby number
Re_o = Constant(100) # Reynolds number
Pe_o = Constant(1000) # Peclet number

#########################specifying initial conditions
print("option 1: Wind blowing over the ocean from left to right in square patch of the domain \n")
print("option 2: Circular velocity field of the atmosphere like a cyclone rotating anti-clockwise \n")
opt = input("Please choose from the above options (type 1 or 2) for atm vel initial condition: ")

bell = 0.5*(1+cos(math.pi*min_value(sqrt(pow(x-0.5, 2) + pow(y-0.5, 2))/0.25, 1.0)))
circ = conditional(sqrt(pow(x-0.5, 2) + pow(y-0.5,2)) < 0.25, 1.0, 0.0)
sq = conditional(And(And(x> 0.25, x < 0.75), And(y > 0.4, y < 0.6)), 1.0, 0.0)
i_uo = project(as_vector([Constant(0),Constant(0)]), V_1)
To_ = Function(V_2).interpolate(Constant(1))
Ta_ = Function(V_2).interpolate(Constant(1))
p_= Function(V_3).interpolate(Constant(0))

if opt=="1":
    i_ua = project(as_vector([sq,0]), V_1)
    o_file = "test_case_wind.pvd"
elif opt=="2":
    i_ua = project(as_vector([-3*circ*(y-0.5), 3*circ*(x-0.5)]), V_1)
    o_file = "test_case_cyclone.pvd"
else:
    print("wrong choice ! program won't run successfully !")
#############################

gamma = -Constant(1.0)
sigma = -Constant(1.0)
ua_.assign(i_ua)
uo_.assign(i_uo)

F_1 = (-inner(grad(q_),grad(psi_)))* dx - div(ua_)*psi_*dx

nullspace_1 = VectorSpaceBasis(constant=True)
solve (F_1 == 0, q_, nullspace = nullspace_1)

F = ( inner(ua - ua_, va)
        + Dt*half*(inner(dot(ua, nabla_grad(ua)), va) + inner(dot(ua_, nabla_grad(ua_)), va))
        + Dt*half*(1/Ro_a)*(-(ua[1]+ua_[1])*va[0] +(ua[0]+ua_[0])*va[1]) 
        + Dt*half*(1/C_a)*inner((grad(Ta)+grad(Ta_)),va)
        + Dt *half *(1/Re_a)*inner((nabla_grad(ua)+nabla_grad(ua_)), nabla_grad(va))
        + (Ta -Ta_)*phi_a + Dt*half*(inner(ua_,grad(Ta_)) + inner(ua,grad(Ta)))*phi_a
        - Dt*gamma*half*(Ta - To + Ta_ - To_)* phi_a
        + Dt*half*(1/Pe_a)*inner((grad(Ta)+grad(Ta_)),grad(phi_a))
        + inner(uo-uo_,vo)
        + Dt*half*(inner(dot(uo, nabla_grad(uo)), vo) + inner(dot(uo_, nabla_grad(uo_)), vo))
        + Dt*half*(1/Ro_o)*(-(uo[1]+uo_[1])*vo[0] +(uo[0]+uo_[0])*vo[1])
        - Dt*(1/Ro_o)*p*div(vo) 
        - Dt*half*sigma*inner((uo - (ua - grad(q)) + uo_ - (ua_ - grad(q_))),vo)
        + Dt *half *(1/Re_o)*inner((nabla_grad(uo)+nabla_grad(uo_)), nabla_grad(vo))
        + Dt*div(uo)*eta
        + (To -To_)*phi_o + Dt*half*(inner(uo_,grad(To_))+inner(uo,grad(To)))*phi_o
        + Dt*half*(1/Pe_o)*inner((grad(To)+grad(To_)),grad(phi_o))
        - inner(grad(q), grad(psi)) - div(ua)*psi)* dx



bound_cond = [DirichletBC(Z.sub(0), Constant((0, 0)), (1,2,3,4)),DirichletBC(Z.sub(2), Constant((0, 0)), (1, 2, 3,4))]

nullspace_2 = MixedVectorSpaceBasis(Z, [Z.sub(0), Z.sub(1), Z.sub(2),VectorSpaceBasis(constant=True), Z.sub(4), VectorSpaceBasis(constant=True)])



ua_ns = Function(V_1)
ua_ns.interpolate(grad(q_))
ua_s = Function(V_1)
ua_s.assign(ua_ - ua_ns)
ua_s.rename("ua_sol")


p_.rename("ocean_pressure")
To_.rename("ocean_temperature")
Ta_.rename("atm_temperature")
ua_.rename("atm_velocity")
uo_.rename("ocean_velocity")

outfile = File("./results/"+ o_file)
outfile.write(ua_, uo_, Ta_, To_, p_, ua_s)

t = Dt
iter = 1
if opt=="1":
    end = 0.2
elif opt=="2":
    end = 0.2
else:
    print("wrong choice ! program won't run successfully !")

freq = 5 # printing results after every freq solves
t_step = freq*Dt  # printing time step
current_time = time.strftime("%H:%M:%S", time.localtime())
print("Local time at the start of simulation:",current_time)
start_time = time.time()

while (round(t,4)<=end):

    solve(F==0, w, bcs= bound_cond, nullspace=nullspace_2)

    ua,Ta,uo,p,To,q= w.split()
    if iter%freq==0:
        if iter==freq:
            end_time = time.time()
            execution_time = (end_time-start_time)/60 # running time for one time step (t_step)
            print("Approx. running time for one t_step: %.2f minutes" %execution_time)
            total_execution_time = (end/t_step)*execution_time
            print("Approx. total running time: %.2f minutes:" %total_execution_time)
            print("Approx total running time: %.2f hours:"%(total_execution_time/60))
        print("t=", round(t,4))
        p.rename("ocean_pressure")
        To.rename("ocean_temperature")
        Ta.rename("atm_temperature")
        ua.rename("atm_velocity")
        uo.rename("ocean_velocity")
        ua_ns = Function(V_1)
        ua_ns.interpolate(grad(q))
        ua_s.assign(ua- ua_ns)
        outfile.write(ua, uo, Ta, To, p, ua_s)
    ua_.assign(ua)
    Ta_.assign(Ta)
    uo_.assign(uo)
    To_.assign(To)
    q_.assign(q)

    t += Dt
    iter +=1