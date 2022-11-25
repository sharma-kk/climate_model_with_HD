import time
from functions_barycentric_mesh import *

N = 50
bmh = PeriodicUnitSquareBaryMeshHierarchy(N,0)
M =  bmh[-1]

V_1 = VectorFunctionSpace(M, "CG", 2)
V_1_out = VectorFunctionSpace(M, "CG", 1)
V_2 = FunctionSpace(M, "CG", 1)
V_3 = FunctionSpace(M, "DG", 1)

Z = V_1*V_2*V_1*V_3*V_2

w = Function(Z)
ua,Ta,uo,p,To = split(w)
va,phi,vo,q,psi = TestFunctions(Z)
ua_ = Function(V_1)
uo_ = Function(V_1)

Dt = 0.1*(1/N) # CFL condition
half = Constant(0.5)

x,y = SpatialCoordinate(M)

# dimensionless constants for atmosphere
Ro_a = Constant(10) # Rossby number
Re_a = Constant(100) # Reynolds number
Pe_a = Constant(100) # Peclet number


# dimensionless constants for ocean
Ro_o = Constant(10) # Rossby number
Re_o = Constant(100) # Reynolds number
Pe_o = Constant(100) # Peclet number


bell = 0.5*(1+cos(math.pi*min_value(sqrt(pow(x-0.5, 2) + pow(y-0.5, 2))/0.25, 1.0)))
i_uo = project(as_vector([Constant(0),Constant(0)]), V_1)
To_ = Function(V_2).interpolate(3000 + 50*bell)
p_= Function(V_3).interpolate(Constant(0))

# i_ua = project(as_vector([(sin(pi*y))**2, 0]), V_1)
i_ua = project(as_vector([Constant(0),Constant(0)]), V_1)
Ta_ = Function(V_2).interpolate(Constant(3000))

gamma = -Constant(1.0)
sigma = -Constant(1.0)
ua_.assign(i_ua)
uo_.assign(i_uo)


F = ( inner(ua-ua_,va)
    + Dt*half*(1/Ro_a)*(-(ua[1]+ua_[1])*va[0] +(ua[0]+ua_[0])*va[1])
    + Dt *half *(1/Re_a)*inner((nabla_grad(ua)+nabla_grad(ua_)), nabla_grad(va))
    + Dt*half*(inner(dot(ua, nabla_grad(ua)), va) + inner(dot(ua_, nabla_grad(ua_)), va))
    + Dt*half*(1/Ro_a)*inner((grad(Ta)+grad(Ta_)),va)
    - Dt*gamma*half*(Ta - To + Ta_ - To_)* phi
    + (Ta -Ta_)*phi + Dt*half*(inner(ua_,grad(Ta_)) + inner(ua,grad(Ta)))*phi
    + Dt*half*(1/Pe_a)*inner((grad(Ta)+grad(Ta_)),grad(phi))
    + inner(uo-uo_,vo)
    + Dt*half*(1/Ro_o)*(-(uo[1]+uo_[1])*vo[0] +(uo[0]+uo_[0])*vo[1])
    + Dt *half *(1/Re_o)*inner((nabla_grad(uo)+nabla_grad(uo_)), nabla_grad(vo))
    + Dt*half*(inner(dot(uo, nabla_grad(uo)), vo) + inner(dot(uo_, nabla_grad(uo_)), vo))
    - Dt*(1/Ro_o)*p*div(vo) + Dt*div(uo)*q
    - Dt*half*sigma*inner((uo_ - ua_ + uo -ua),vo)
    - Dt*sigma*inner(project(as_vector([Constant(assemble(ua_[0]*dx)),Constant(assemble(ua_[1]*dx))]), V_1),vo)
    + (To -To_)*psi + Dt*half*(inner(uo_,grad(To_))+inner(uo,grad(To)))*psi
    + Dt*half*(1/Pe_o)*inner((grad(To)+grad(To_)),grad(psi)) )*dx

nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), Z.sub(1), Z.sub(2),VectorSpaceBasis(constant=True), Z.sub(4)])

p_.rename("Ocean_pressure")
To_.rename("Ocean_temperature")
Ta_.rename("Atm_temperature")

outfile = File(./results/"clim_modeL_with_pbc_CC_1.pvd")
outfile.write(project(i_ua,V_1_out, name= "atm_velocity"),Ta_,project(i_uo,V_1_out, name= "ocean_velocity"),p_,To_)

t = Dt
iter = 1
end = 0.1
freq = 1 # printing results after every freq solves
t_step = freq*Dt  # printing time step

current_time = time.strftime("%H:%M:%S", time.localtime())
print("Local time at the start of simulation:",current_time)
start_time = time.time()

while (round(t,4)<=end):

    solve(F==0, w, nullspace= nullspace)

    ua,Ta,uo,p,To= w.split()

    if iter%freq==0:
        if iter==freq:
            end_time = time.time()
            execution_time = (end_time-start_time)/60 # running time for one time step (t_step)
            print("Approx. running time for one t_step: %.2f minutes" %execution_time)
            total_execution_time = (end/t_step)*execution_time
            print("Approx. total running time: %.2f minutes:" %total_execution_time)
            print("Approx total running time: %.2f hours:"%(total_execution_time/60))
        print("t=", round(t,4))
        p.rename("Ocean_pressure")
        To.rename("Ocean_temperature")
        Ta.rename("Atm_temperature")
        outfile.write(project(ua,V_1_out, name= "atm_velocity"),Ta,project(uo,V_1_out, name= "ocean_velocity"),p,To)

    ua_.assign(ua)
    Ta_.assign(Ta)
    uo_.assign(uo)
    To_.assign(To)

    t += Dt
    iter +=1

