

import matplotlib.pyplot as plt
from math import sqrt

with open("ev_gnn.txt", "r") as f:
	ev_gnn_str = f.read()
with open("ev_eam.txt", "r") as f:
	ev_eam_str = f.read()
with open("f_gnn.txt", "r") as f:
	f_gnn_str = f.read()
with open("f_eam.txt", "r") as f:
	f_eam_str = f.read()


f_gnn_x, f_gnn_y = [], []
for line in f_gnn_str.split("\n"):
	if line == "":
		break
	i = line.find(" ")
	f_gnn_x.append(float(line[:i]))
	f_gnn_y.append(eval(line[i+1:]))

f_eam_x, f_eam_y = [], []
for line in f_eam_str.split("\n"):
	if line == "":
		break
	i = line.find(" ")
	f_eam_x.append(float(line[:i]))
	f_eam_y.append(eval(line[i+1:]))

ev_gnn_x, ev_gnn_y = [], []
for line in ev_gnn_str.split("\n"):
	if line == "":
		break
	elif line[0] != "#":
		i = line.find(" ")
		ev_gnn_x.append(float(line[:i]))
		ev_gnn_y.append(float(line[i+1:]))

ev_eam_x, ev_eam_y = [], []
for line in ev_eam_str.split("\n"):
	if line == "":
		break
	elif line[0] != "#":
		i = line.find(" ")
		ev_eam_x.append(float(line[:i]))
		ev_eam_y.append(float(line[i+1:]))

f_eam_x_p, f_eam_y_p, f_gnn_x_p, f_gnn_y_p = [], [], [], []
f_eam_x_s = list(set(f_eam_x))
for x_s in f_eam_x_s:
	for x, y in zip(f_eam_x, f_eam_y):
		if x == x_s:
			f_eam_x_p.append(x)
			f_eam_y_p.append(y)
			break
f_gnn_x_s = list(set(f_gnn_x))
for x_s in f_gnn_x_s:
	for x, y in zip(f_gnn_x, f_gnn_y):
		if x == x_s:
			f_gnn_x_p.append(x)
			f_gnn_y_p.append(y)
			break

def norm(v):
	return sqrt(v[0]**2 + v[1]**2 + v[2]**2)

f_eam_y_n = [max([norm(v[0]), norm(v[1]), norm(v[2]), norm(v[3])]) for v in f_eam_y_p]
f_gnn_y_n = [max([norm(v[0]), norm(v[1]), norm(v[2]), norm(v[3])]) for v in f_gnn_y_p]

print(f"f_eam_x_p: {f_eam_x_p}\n\n")
print(f"f_eam_y_n: {f_eam_y_n}\n\n")
print(f"f_gnn_x_p: {f_gnn_x_p}\n\n")
print(f"f_gnn_y_n: {f_gnn_y_n}\n\n")

fig, axs = plt.subplots(2, 2)
axs[0][0].plot(ev_eam_x, ev_eam_y, color="blue", label="eam")
axs[0][1].plot(ev_gnn_x, ev_gnn_y, color="orange", label="gnn")
axs[1][0].bar(f_eam_x_p, f_eam_y_n, label="eam f norm")
axs[1][1].bar(f_gnn_x_p, f_gnn_y_n, label="gnn f norm")
fig.tight_layout(pad=3.0)
[ax.legend() for ax in [axs[0][0], axs[0][1], axs[1][0], axs[1][1]]]
plt.show()