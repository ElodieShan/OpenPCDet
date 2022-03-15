from basic import *

def get_round(data, rn=2):
    return round(float(data),rn)

def get_3d_bev(f):
    bbox = f.readline()
    bev = f.readline()
    threed = f.readline()
    aos = f.readline()
    threed = threed.strip().replace("3d   AP:",'').split(', ')
    bev = bev.strip().replace('bev  AP:','').split(', ')
    for idx in range(len(threed)):
        threed[idx] = get_round(threed[idx], rn=2)
        bev[idx] = get_round(bev[idx], rn=2)

    return f, threed, bev

with open("./result.txt") as f:
    for line in f:
        if line.find(Car_)!=-1:
            f, threed, bev = get_3d_bev(f)
            for idx, difficult in enumerate(['easy', 'mod', 'hard']):
                res['CAR'][difficult].append(threed[idx])
                res['CAR'][difficult].append(bev[idx])

        if line.find(Pedestrian_)!=-1:
            f, threed, bev = get_3d_bev(f)
            for idx, difficult in enumerate(['easy', 'mod', 'hard']):
                res['PED'][difficult].append(threed[idx])
                res['PED'][difficult].append(bev[idx])
        if line.find(Cyclist_)!=-1:
            f, threed, bev = get_3d_bev(f)
            for idx, difficult in enumerate(['easy', 'mod', 'hard']):
                res['CYC'][difficult].append(threed[idx])
                res['CYC'][difficult].append(bev[idx])
print("result:\n")


for class_name in ['CAR', 'PED', 'CYC']:
    for difficult in ['easy', 'mod', 'hard']: 
        output_str = str(res[class_name][difficult][0]) + '/' + str(res[class_name][difficult][1])
        print(output_str, end='\t')

print("\n******Done******\n")
