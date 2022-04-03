import itertools
import subprocess
import os, sys

dirname = sys.argv[1]
os.mkdir(dirname)

px = [4,8,16,32]
pya = [4,8,16,32]
pyb = [4,8,16,32]
rxa = [.1, .3, .5]
rxb = [.1, .3, .5]
exa_mix = [.5, .7, .9]
n_per_ftrs = [4, 8, 16, 32, 64, 128, 256]
n_per_ftr2s = [4, 8, 16, 32, 64, 128, 256]
ax = [-.5, -1]
aya = [-.5, -1]
ayb = [-.5, -1]

#needs to be in same order
# chose n_per_ftrs and exa_mix to be outermost params, since those 
# appeared to be the ones slowing down the program most. So separate
# jobs with large n_per_ftrs and small exa_mix in case they fail. 
outer_params = [n_per_ftrs, n_per_ftr2s,exa_mix]
outer_param_names = ["n_per_ftrs", "n_per_ftr2s", "exa_mix"]
inner_params = [
    {'name': 'rxa', 'values': rxa},
    {'name': 'rxb', 'values': rxb},
    {'name': 'ax', 'values': ax},
    {'name': 'aya', 'values': aya},
    {'name': 'ayb', 'values': ayb},
    {'name': 'px', 'values': px},
    {'name': 'pya', 'values': pya},
    {'name': 'pyb', 'values': pyb}
]

#get all possible combinations of parameters
#this is ok because # of combinations is small enough to hold in memory at the same time
combinations = list(itertools.product(*outer_params))
j = 0
for combination in combinations:
    #construct CLI args to pass fixed parameters to job
    cli_args = ["python", "analyze_model_job.py", "-iteration", str(j), "-dirname", dirname]
    i = 0
    for param in outer_param_names:
        cli_args.append("-" + param)
        cli_args.append(str(combination[i]))
        i = i + 1
    for param in inner_params:
        cli_args.append("-" + param['name'])
        [cli_args.append(str(value)) for value in param['values']]
    j = j+1
    #print(" ".join(cli_args))
    subprocess.run(["sbatch","--time=1-", "--wrap", " ".join(cli_args)])

