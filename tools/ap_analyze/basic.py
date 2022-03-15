ap_type="AP_R40"
Car_ = "Car " + ap_type + "@0.70, 0.70, 0.70:"
Pedestrian_ = "Pedestrian " + ap_type + "@0.50, 0.50, 0.50:"
Cyclist_ = "Cyclist " + ap_type + "@0.50, 0.50, 0.50:"


res = {
    'CAR':{
        'easy':[], # 3d & bev
        'mod':[],
        'hard':[],
    },
    'PED':{
        'easy':[],
        'mod':[],
        'hard':[],
    },
    'CYC':{
        'easy':[],
        'mod':[],
        'hard':[],
    },
}