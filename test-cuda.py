from subprocess import call
import time

for b in [128]:
	for d in [1050, 2100, 5040, 7560, 10080, 12600]:
		call(["./cuda.out",  str(d), "100", str(b)])
		print("B=" + str(b) + "   D=" + str(d) + "\n")
		print("\n\n\n\n")
		time.sleep(2)