from firedrake_adjoint import *
import firedrake_adjoint
from pyadjoint.enlisting import Enlist
import pyrevolve
from pyrevolve.schedulers import Action

class TimestepRegister:
    def __init__(self, tape=None):
        if tape is None:
            self.tape = get_working_tape()
        else:
            self.tape = tape
        self.block_index = [len(self.tape._blocks)]

    def mark_end_of_timestep(self):
        self.block_index.append(len(self.tape._blocks))

    def n_timesteps(self):
        return len(self.block_index)-1


class Function(firedrake_adjoint.Function):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _ad_create_checkpoint(self):
        if self.original_block_variable is self.block_variable:
            return super()._ad_create_checkpoint()
        else:
            return None


class RevolveReducedFunctional(ReducedFunctional):
    def __init__(self, functional, controls, timestep_register, n_checkpoints, _verbose=False, **kwargs):
        super().__init__(functional, controls, **kwargs)
        self.timestep_register = timestep_register
        self.n_checkpoints = n_checkpoints
        self._checkpoints = []
        self._checkpoint_from_last_forward = None
        self._verbose = _verbose

    def __call__(self, values):
        values = Enlist(values)
        if len(values) != len(self.controls):
            raise ValueError("values should be a list of same length as controls.")

        # Call callback.
        self.eval_cb_pre(self.controls.delist(values))
        for i, value in enumerate(values):
            if isinstance(value, Function):
                self.controls[i].block_variable.checkpoint = firedrake.Function(value)
            else:
                self.controls[i].update(value)

        self.tape.reset_blocks()
        self._mark_block_variable_lifespan()
        blocks = self.tape.get_blocks()


        n_timesteps = self.timestep_register.n_timesteps()
        self._revolve = pyrevolve.Revolve(self.n_checkpoints, n_timesteps)
        with self.marked_controls(), stop_annotating():
            while True:
                action = self._revolve.next()
                if action.type == Action.ADVANCE:
                    self._revolve_forward_action()
                elif action.type ==Action.TAKESHOT:
                    self._revolve_takeshot_action()
                elif action.type == Action.RESTORE:
                    self._revolve_restore_action()
                elif action.type == Action.LASTFW:
                    self._revolve_forward_one_timestep()
                    break
                else:
                    print(action)
                    raise SyntaxError("Unknown revolve action type in forward mode")

        func_value = self.scale * self.functional.block_variable.checkpoint

        # Call callback
        self.eval_cb_post(func_value, self.controls.delist(values))

        return func_value

    def derivative(self, options={}):

        values = [c.data() for c in self.controls]
        self.derivative_cb_pre(self.controls.delist(values))

        options = {} if options is None else options
        self.tape.reset_variables()
        self.functional.adj_value = self.scale
        m = self.controls

        with stop_annotating(), self.marked_controls(), self.tape.marked_nodes(m):
            self._revolve_backward_one_timestep()
            while True:
                action = self._revolve.next()
                if action.type == Action.ADVANCE:
                    self._revolve_forward_action()
                elif action.type ==Action.TAKESHOT:
                    self._revolve_takeshot_action()
                elif action.type == Action.RESTORE:
                    self._revolve_restore_action()
                elif action.type == Action.REVERSE:
                    self._revolve_forward_one_timestep()
                    self._revolve_backward_one_timestep()
                elif action.type == Action.TERMINATE:
                    break
                else:
                    print(action)
                    raise SyntaxError("Unknown revolve action type in backward mode")

        derivatives = m.delist([i.get_derivative(options=options) for i in m])

        # Call callback
        self.derivative_cb_post(self.functional.block_variable.checkpoint,
                                derivatives,
                                self.controls.delist(values))

        return derivatives

    def _mark_block_variable_lifespan(self):

        blocks = self.tape.get_blocks()

        # reset block variables (only block variables that are output in the graph are touched)
        for block in blocks:
            for bv in block.get_outputs():
                bv._final_block_index = -1

        # working backwards find the last block in which a block variable is used
        for i in range(len(blocks)-1, -1, -1):
            blocks[i]._final_dependencies = []
            for bv in blocks[i].get_dependencies():
                if hasattr(bv, '_final_block_index') and bv._final_block_index==-1:
                    bv._final_block_index = i
                    blocks[i]._final_dependencies.append(bv)
            for bv in blocks[i].get_outputs():
                if bv._final_block_index == -1:
                    bv._final_block_index = i
                    blocks[i]._final_dependencies.append(bv)

        # ensure that the block_variable ass. with the functional survives
        bv = self.functional.block_variable
        blocks[bv._final_block_index]._final_dependencies.remove(bv)
        bv._final_block_index = len(blocks) + 1

    def _revolve_forward_action(self, construct_checkpoint=True):
        blocks = self.tape.get_blocks()
        i0 = self.timestep_register.block_index[self._revolve.old_capo]
        i1 = self.timestep_register.block_index[self._revolve.capo]
        self.log("FORWARD:", self._revolve.old_capo, self._revolve.capo, i0, i1)
        assert self._checkpoint_from_last_forward is None, "Have not used last constructed checkpoint"
        self._checkpoint_from_last_forward = {}
        for i in range(i0, i1):
            blocks[i].recompute()
            for bv in blocks[i]._final_dependencies:
                assert bv._final_block_index == i
                bv.checkpoint = None
            if construct_checkpoint:
                for output in blocks[i].get_outputs():
                    if hasattr(output, '_final_block_index') and output._final_block_index >= i1:
                        self._checkpoint_from_last_forward[output] = output.checkpoint

    def _revolve_forward_one_timestep(self):
        self._checkpoint_from_last_forward = None
        blocks = self.tape.get_blocks()
        i0 = self.timestep_register.block_index[self._revolve.capo]
        i1 = self.timestep_register.block_index[self._revolve.capo+1]
        self.log("FORWARD ONE:", self._revolve.capo, i0, i1)
        for i in range(i0, i1):
            blocks[i].recompute()

    def _revolve_backward_one_timestep(self):
        blocks = self.tape.get_blocks()
        i0 = self.timestep_register.block_index[self._revolve.capo]
        i1 = self.timestep_register.block_index[self._revolve.capo+1]
        self.log("BACKWARD ONE:", self._revolve.capo, i0, i1)
        for i in range(i1-1, i0-1, -1):
            self.log(i)
            blocks[i].evaluate_adj(markings=True)
            for bv in blocks[i].get_outputs():
                if not bv.is_control:
                    bv.reset_variables(types=("adjoint"))
                    if isinstance(bv.output, firedrake.Function):
                        bv._checkpoint = None

    def _revolve_takeshot_action(self):
        cp = self._revolve.cp_pointer
        self.log("TAKESHOT STORE AT:", cp)
        if cp == 0:
            pass
        elif cp <= len(self._checkpoints):
            self._checkpoints[cp-1] = self._checkpoint_from_last_forward
        elif cp-1 == len(self._checkpoints):
            self._checkpoints.append(self._checkpoint_from_last_forward)
        else:
            raise IndexError("Checkpoint index larger than number of stored checkpoints.")
        self._checkpoint_from_last_forward = None

    def _revolve_restore_action(self):
        cp = self._revolve.cp_pointer
        self.log("RESTORE FROM:", cp)
        if cp == 0:
            return
        elif cp > len(self._checkpoints):
            raise IndexError("Checkpoint index larger than number of stored checkpoints.")
        for bv in self._checkpoints[cp-1]:
            bv.checkpoint = self._checkpoints[cp-1][bv]

    def log(self, *args):
        if self._verbose:
            print(*args)
