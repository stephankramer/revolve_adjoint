import pytest
pytest.importorskip("firedrake")

from firedrake import *
from revolve_adjoint import *

from numpy.random import rand


def test_assign_linear_combination():
    tr = TimestepRegister()
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "CG", 1)

    x, = SpatialCoordinate(mesh)
    f = interpolate(x, V)
    g = interpolate(sin(x), V)
    u = Function(V)

    u.assign(3*f + g)
    tr.mark_end_of_timestep()

    J = assemble(u**2*dx)
    tr.mark_end_of_timestep()
    rf = RevolveReducedFunctional(J, Control(f), tr, 1)

    h = Function(V)
    h.vector()[:] = rand(V.dim())
    assert taylor_test(rf, f, h) > 1.9


def test_assign_vector_valued():
    tr = TimestepRegister()
    mesh = UnitSquareMesh(10, 10)
    element = VectorElement("CG", mesh.ufl_cell(), degree=1, dim=2)
    V = FunctionSpace(mesh, element)

    x = SpatialCoordinate(mesh)
    f = interpolate(as_vector((x[0]*x[1], x[0]+x[1])), V)
    g = interpolate(as_vector((sin(x[1])+x[0], cos(x[0])*x[1])), V)
    u = Function(V)

    u.assign(f - 0.5*g)

    J = assemble(inner(f, g)*u**2*dx)
    tr.mark_end_of_timestep()
    rf = RevolveReducedFunctional(J, Control(f), tr, 1)

    h = Function(V)
    h.vector()[:] = 1
    assert taylor_test(rf, f, h) > 1.9



def test_assign_nonlincom():
    tr = TimestepRegister()
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "CG", 1)

    x = SpatialCoordinate(mesh)
    f = interpolate(x[0], V)
    g = interpolate(sin(x[0]), V)
    u = Function(V)

    u.assign(f*g)

    J = assemble(u ** 2 * dx)
    tr.mark_end_of_timestep()
    rf = RevolveReducedFunctional(J, Control(f), tr, 1)

    h = Function(V)
    h.vector()[:] = rand(V.dim())
    assert taylor_test(rf, f, h) > 1.9


def test_assign_nonlin_changing():
    tr = TimestepRegister()
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "CG", 1)

    x = SpatialCoordinate(mesh)
    f = interpolate(x[0], V)
    g = interpolate(sin(x[0]), V)
    control = Control(g)

    test = TestFunction(V)
    trial = TrialFunction(V)
    a = inner(grad(trial), grad(test))*dx
    L = inner(g, test)*dx

    bc = DirichletBC(V, g, "on_boundary")
    sol = Function(V)
    solve(a == L, sol, bc)

    u = Function(V)

    u.assign(f*sol*g)

    J = assemble(u ** 2 * dx)
    tr.mark_end_of_timestep()
    rf = RevolveReducedFunctional(J, control, tr, 1)

    g = Function(V)
    g.vector()[:] = rand(V.dim())

    h = Function(V)
    h.vector()[:] = rand(V.dim())
    assert taylor_test(rf, g, h) > 1.9
