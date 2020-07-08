from subprocess import call
import time

for p in range(1, 9):
	for d in [2100, 5040, 7560, 10080, 12600]:
		call(["mpirun",  "-np",  str(p), "mpi.out",  str(d), "100"])
		print("P=" + str(p) + "   D=" + str(d) + "\n")
		print("\n\n\n\n")
		time.sleep(2)