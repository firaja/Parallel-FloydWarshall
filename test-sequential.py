from subprocess import call
import time

for d in [1050, 2100, 5040, 7560, 10080, 12600]:
	call(["./sequential.out", str(d), "100"])
	print("D=" + str(d) + "\n")
	print("\n\n\n\n")
	time.sleep(2)