
import lammps

i = 1
def callback(caller, ntimestep, nlocal, tag, x, fext):
    global i
    fext.fill(i * 1.0)
    L.set_variable("gnn_energy", str(i*10.0))
    i += 1
    #L.fix_external_set_energy_global("2", 10.0)


L = lammps.lammps()
L.file("in")
L.set_fix_external_callback("gnn", callback, None)
L.command("run 1")