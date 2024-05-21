counter = 0
numbers = [465, 514, 720]
for n in numbers:
    for j in range(20):
        with open("jobs/"+str(counter)+".sh", "w") as f:
            counter = counter + 1
            f.write("#!/bin/bash\n")
            f.write("#SBATCH --ntasks=24\n")
            f.write("#SBATCH --job-name=std_"+str(counter)+"\n")
            f.write("#SBATCH --output=/cluster/project/krause/mmutny/std_"+str(counter)+".out\n")
            f.write("#SBATCH --error=/cluster/project/krause/mmutny/std_"+str(counter)+".err\n")
            f.write("#SBATCH --mem-per-cpu=4000\n")
            f.write("#SBATCH --time=03:59:00\n")
            f.write("\n")
            f.write("cd ../ && python explore_simulation.py --seed="+str(j)+" --n="+str(n)+" \n")
