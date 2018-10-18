names = ["Assault", "Atlantis", "BeamRider","CrazyClimber", "DemonAttack", "Gopher", "Kangaroo", "KungFuMaster", "Pong", "RoadRunner", "SpaceInvaders", "StarGunner", "VideoPinball"]
sizes = [1000,10000,100000]

output = ""
for name in names:
    for size in sizes:
        output += "python3 validate.py "+name+" "+str(size)+"\n"

with open("validate.sh", 'a') as myfile:
    myfile.write(output)