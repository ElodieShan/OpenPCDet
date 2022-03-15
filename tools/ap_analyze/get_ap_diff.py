from basic import *
tsp_baseline = [87.86, 91.88, 76.24, 86.88, 73.34, 84.52, 48.95, 55.53, 44.97, 51.10, 41.09, 47.4, 74.79, 78.77, 57.18, 61.91, 53.72, 57.94]


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

with open("./ap_analyze/result.txt") as f:
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
print("3d:\n")

idx = 0
output_str = ""
output_diff_str = ""

for class_name in ['CAR', 'PED', 'CYC']:
    for difficult in ['easy', 'mod', 'hard']: 
        res_3d = res[class_name][difficult][0]
        res_3d_diff = res_3d - tsp_baseline[idx]
        idx += 1
        res_bev = res[class_name][difficult][1]
        res_bev_diff = res_bev - tsp_baseline[idx]
        idx += 1
        output_str += str(res_3d) + '/' + str(res_bev) + '\t'
        
        output_diff_str += "%+.2f"%res_3d_diff +  '/' +"%+.2f"%res_bev_diff +'\t'

print(output_str)
print(output_diff_str)


print("\n******Done******\n")
