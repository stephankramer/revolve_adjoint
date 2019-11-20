from firedrake import *
from revolve_adjoint import *
#from firedrake_adjoint import *

tr = TimestepRegister()

mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, "CG", 1)
u0 = Function(V, name='u0')
u1 = Function(V, name='u1')
c = Control(u0)
ur = Function(V, name='ur')
x,y = SpatialCoordinate(mesh)
ur.interpolate(x)
#ur.assign(1.0)
alfa = Function(V, name='alfa')
alfa.interpolate(y)
#alfa.assign(1.0)
tst = TestFunction(V)

print(str(u0), str(u1), str(ur), str(alfa))

source = alfa*ur
dt = 1

F = tst*(u1-u0 - dt*(source-alfa*u1))*dx

functional = 0
for i in range(100):
    solve(F==0, u1)
    u0.assign(u1)
    f = assemble(u1*dx)
    functional += f
    ur.assign(i*alfa)
    tr.mark_end_of_timestep()

print(functional)
rf = RevolveReducedFunctional(functional, c, tr, 10)
#rf = ReducedFunctional(functional, c)
u1.assign(0.0)
print(rf(u1))
print(rf.derivative().dat.data)
u1.assign(1.0)
print(rf(u1))
print(rf.derivative().dat.data)
u1.assign(0.0)
print(rf(u1))
print(rf.derivative().dat.data)
