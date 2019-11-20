import pytest
pytest.importorskip("firedrake")

from firedrake import *
from revolve_adjoint import *

from numpy.random import rand

def test_function():
    tr = TimestepRegister()
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "Lagrange", 1)

    c = Constant(1)
    f = Function(V)
    f.vector()[:] = 1

    u = Function(V)
    v = TestFunction(V)
    bc = DirichletBC(V, Constant(1), "on_boundary")

    F = inner(grad(u), grad(v))*dx - f**2*v*dx
    solve(F == 0, u, bc)

    J = assemble(c**2*u*dx)
    tr.mark_end_of_timestep()
    Jhat = RevolveReducedFunctional(J, Control(f), tr, 1)
    
    h = Function(V)
    h.vector()[:] = rand(V.dof_dset.size)
    assert taylor_test(Jhat, f, h) > 1.9


@pytest.mark.parametrize("control", ["dirichlet", "neumann"])
def test_wrt_function_dirichlet_boundary(control):
    tr = TimestepRegister()
    mesh = UnitSquareMesh(10,10)

    V = FunctionSpace(mesh,"CG",1)
    u = TrialFunction(V)
    u_ = Function(V)
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)
    bc_func = project(sin(y), V)
    bc1 = DirichletBC(V, bc_func, 1)
    bc2 = DirichletBC(V, 2, 2)
    bc = [bc1,bc2]

    g1 = Constant(2)
    g2 = Constant(1)
    f = Function(V)
    f.vector()[:] = 10

    a = inner(grad(u), grad(v))*dx
    L = inner(f,v)*dx + inner(g1,v)*ds(4) + inner(g2,v)*ds(3)

    solve(a==L,u_,bc)

    J = assemble(u_**2*dx)
    tr.mark_end_of_timestep()

    if control == "dirichlet":
        Jhat = RevolveReducedFunctional(J, Control(bc_func), tr, 1)
        g = bc_func
        h = Function(V)
        h.vector()[:] = 1
    else:
        Jhat = RevolveReducedFunctional(J, Control(g1), tr, 1)
        g = g1
        h = Constant(1)

    assert taylor_test(Jhat, g, h) > 1.9


def test_time_dependent():
    tr = TimestepRegister()
    # Defining the domain, 100 points from 0 to 1
    mesh = IntervalMesh(100, 0, 1)

    # Defining function space, test and trial functions
    V = FunctionSpace(mesh,"CG",1)
    u = TrialFunction(V)
    u_ = Function(V)
    v = TestFunction(V)

    # Dirichlet boundary conditions
    bc_left = DirichletBC(V, 1, 1)
    bc_right = DirichletBC(V, 2, 2)
    bc = [bc_left, bc_right]

    # Some variables
    T = 1.5
    dt = 0.1
    f = Function(V)
    f.vector()[:] = 1

    u_1 = Function(V)
    u_1.vector()[:] = 1
    control = Control(u_1)

    a = u_1*u*v*dx + dt*f*inner(grad(u),grad(v))*dx
    L = u_1*v*dx

    # Time loop
    t = dt
    while t <= T:
        solve(a == L, u_, bc)
        f.assign(t*u_)
        t += dt
        tr.mark_end_of_timestep()

    J = assemble(u_1**2*dx)
    tr.mark_end_of_timestep()

    Jhat = RevolveReducedFunctional(J, control, tr, 2)

    h = Function(V)
    h.vector()[:] = 1
    assert taylor_test(Jhat, control.data(), h) > 1.9


def test_mixed_boundary():
    tr = TimestepRegister()
    mesh = UnitSquareMesh(10,10)

    V = FunctionSpace(mesh,"CG",1)
    u = TrialFunction(V)
    u_ = Function(V)
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)
    bc1 = DirichletBC(V, y**2, 1)
    bc2 = DirichletBC(V, 2, 2)
    bc = [bc1,bc2]
    g1 = Constant(2)
    g2 = Constant(1)
    f = Function(V)
    f.vector()[:] = 10

    a = f*inner(grad(u), grad(v))*dx
    L = inner(f,v)*dx + inner(g1,v)*ds(4) + inner(g2,v)*ds(3)

    solve(a==L,u_,bc)

    J = assemble(u_**2*dx)
    tr.mark_end_of_timestep()

    Jhat = RevolveReducedFunctional(J, Control(f), tr, 1)
    h = Function(V)
    h.vector()[:] = 1
    assert taylor_test(Jhat, f, h) > 1.9


@pytest.mark.xfail(reason="Constant annotation not yet quite right")
def test_assemble_recompute():
    tr = TimestepRegister()
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)

    v = TestFunction(V)
    u = Function(V)
    u.vector()[:] = 1

    bc = DirichletBC(V, Constant(1), "on_boundary")
    f = Function(V)
    f.vector()[:] = 2
    expr = Constant(assemble(f**2*dx))
    J = assemble(expr**2*dx(domain=mesh))
    tr.mark_end_of_timestep()
    Jhat = RevolveReducedFunctional(J, Control(f), tr, 1)

    h = Function(V)
    h.vector()[:] = 1
    assert taylor_test(Jhat, f, h) > 1.9
