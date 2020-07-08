from subprocess import call
import time


for t in range(2, 9):
	for d in [1050, 2100, 5040, 7560, 10080, 12600]:
		call(["./openmp.out",  str(d), "100", str(t)])
		print("T=" + str(t) + "   D=" + str(d) + "\n")
		print("\n\n\n\n")
		time.sleep(2)