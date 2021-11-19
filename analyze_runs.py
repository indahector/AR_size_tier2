#!/python

from ARsize_cmip56_library import *
import time

t0 = time.time()
for run in runs[11:]:

    tt0 = time.time()
    print(run)
    analyze_run(run, verbose=True)
    seconds = time.time()-tt0
    if seconds<=60:
        print("{:.2f} seconds ellapsed.\n".format(seconds))
    else:
        minutes = int(seconds/60)
        seconds = int(seconds%60)
        print("{}m {}s ellapsed.\n".format(minutes, seconds))

print("Done")
seconds = time.time()-t0
if seconds<=60:
    print("TOTAL {:.2f} seconds ellapsed.\n".format(seconds))
else:
    minutes = int(seconds/60)
    seconds = int(seconds%60)
    print("TOTAL {}m {}s ellapsed.\n".format(minutes, seconds))
