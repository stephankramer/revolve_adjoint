import pytest
import numpy as np
from firedrake import *
from revolve_adjoint import *
from pyadjoint import taylor_to_dict
from numpy.testing import assert_approx_equal


def test_sin_weak_spatial():
    tr = TimestepRegister()
    mesh = UnitOctahedralSphereMesh(2)
    x = SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)
    S = VectorFunctionSpace(mesh, "CG", 1)
    s = Function(S)
    mesh.coordinates.assign(mesh.coordinates + s)
    
    J = assemble(sin(x[0]) * dx)
    tr.mark_end_of_timestep()
    Jhat = RevolveReducedFunctional(J, Control(s), tr, 1)
    assert_approx_equal(J, Jhat(s))
    computed = Jhat.derivative().vector().get_local()
    
    V = TestFunction(S)
    dJV = div(V)*sin(x[0])*dx + V[0]*cos(x[0])*dx
    actual = assemble(dJV).vector().get_local()
    assert np.allclose(computed, actual, rtol=1e-14)


def test_multiple_assignments():
    tape = get_working_tape()
    tape.clear_tape()
    tr = TimestepRegister()

    mesh = UnitSquareMesh(5, 5)
    S = VectorFunctionSpace(mesh, "CG", 1)
    s = Function(S)

    mesh.coordinates.assign(mesh.coordinates + s)
    mesh.coordinates.assign(mesh.coordinates + s)

    V = FunctionSpace(mesh, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    x, y = SpatialCoordinate(mesh)
    f = (x - 0.2) ** 2 + y ** 2 - 1
    a = dot(grad(u), grad(v)) * dx + u * v * dx
    l = f * v * dx

    u = Function(V)
    solve(a == l, u)
    J = assemble(u * dx)

    tr.mark_end_of_timestep()
    Jhat = RevolveReducedFunctional(J, Control(s), tr, 1)
    assert_approx_equal(J, Jhat(s))
    dJdm = Jhat.derivative()

    pert = as_vector((x * y, sin(x)))
    pert = interpolate(pert, S)
    taylor_test(Jhat, s, pert)

    tape = get_working_tape()
    tape.clear_tape()
    tr = TimestepRegister()

    mesh = UnitSquareMesh(5, 5)
    S = VectorFunctionSpace(mesh, "CG", 1)
    s = Function(S)
    mesh.coordinates.assign(mesh.coordinates + 2*s)

    V = FunctionSpace(mesh, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    x, y = SpatialCoordinate(mesh)
    f = (x - 0.2) ** 2 + y ** 2 - 1
    a = dot(grad(u), grad(v)) * dx + u * v * dx
    l = f * v * dx

    u = Function(V)
    solve(a == l, u)
    J = assemble(u * dx)

    tr.mark_end_of_timestep()
    Jhat = RevolveReducedFunctional(J, Control(s), tr, 1)
    assert_approx_equal(J, Jhat(s))
    assert np.allclose(Jhat.derivative().vector().get_local(),
                       dJdm.vector().get_local())

    pert = as_vector((x * y, sin(x)))
    pert = interpolate(pert, S)
    taylor_test(Jhat, s, pert)
