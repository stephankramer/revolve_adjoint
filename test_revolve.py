import pyrevolve
from pyrevolve.schedulers import Action

revolve = pyrevolve.Revolve(10, 100)
while True:
    action = revolve.next()
    if action.type in [1,2]:
        print(action.type, revolve.capo, 'at:', revolve.cp_pointer)
    else:
        print(action.type, revolve.old_capo, revolve.capo)
    if action.type == Action.TERMINATE:
        break
