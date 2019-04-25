import torch


def get_class_weight():
    """
    class weight for CrossEntropy in Kinetics
    class IDs are based on csv files made with "utils/build_dataset.py"

    if you calculate your dataset, please try this code:


    ```
    df_train = pd.read_csv('./utils/train.csv')
    df_val = pd.read_csv('./utils/val.csv')
    df = pd.concat([df_train, df_val])

    nums = {}
    for i in range(400):
        nums[i] = 0;

    for i in range(len(df)):
        nums[df.iloc[i, 2]] += 1

    class_num = []
    for val in nums.values():
        class_num.append(val)

    class_num = torch.tensor(class_num)

    total = class_num.sum().item()

    frequency = class_num.float() / total
    median = torch.median(frequency)

    class_weight = median / frequency

    ```
    """

    class_weight = torch.tensor([
        1.0395, 1.2895, 1.1287, 2.4025, 0.6356, 1.9495, 0.9682, 0.5914, 1.5000,
        0.5914, 1.7080, 1.0414, 1.0395, 0.9369, 0.9539, 1.6783, 1.8208, 1.1914,
        1.5078, 1.7029, 0.6266, 1.0702, 0.9507, 1.0176, 0.8565, 1.3129, 0.6610,
        1.9627, 0.9931, 1.7815, 1.3986, 0.6082, 0.5994, 0.5975, 0.6056, 2.3347,
        0.6893, 1.0140, 0.8307, 0.5796, 1.1050, 1.9627, 1.7232, 1.2505, 1.1673,
        2.0753, 0.7157, 0.8403, 0.8759, 0.9309, 1.8799, 1.2670, 1.4658, 2.2885,
        1.3624, 0.8248, 1.3100, 0.5796, 1.7284, 0.9309, 0.8019, 0.7888, 1.4367,
        0.9220, 0.8098, 0.8087, 1.0158, 0.7510, 1.0883, 2.1932, 0.8178, 1.3465,
        1.4808, 0.9492, 0.8826, 2.0174, 1.0742, 1.4548, 1.6173, 1.1557, 1.0489,
        0.6012, 1.6128, 0.6959, 1.3189, 1.8498, 0.6280, 1.5777, 1.0782, 0.8098,
        0.6780, 1.1178, 0.7878, 1.8439, 0.6780, 0.9234, 1.1938, 2.0532, 1.4156,
        0.6602, 0.7846, 1.5078, 1.0000, 1.0742, 0.6733, 2.0104, 1.3341, 0.6694,
        0.6219, 0.8908, 0.5773, 0.7588, 0.6314, 1.0527, 0.6186, 0.7471, 0.8213,
        1.6880, 1.9829, 1.4548, 1.8984, 0.8565, 1.8439, 1.4808, 0.8773, 1.8799,
        0.7932, 1.6218, 1.6402, 0.7720, 1.3952, 0.7539, 1.7870, 0.6031, 1.2614,
        1.7440, 0.6212, 0.7639, 1.2505, 0.9983, 0.9033, 0.7588, 1.6979, 0.5914,
        0.5969, 0.7569, 1.3986, 0.9161, 0.6663, 0.6246, 1.4658, 0.6648, 2.0605,
        0.6031, 1.1092, 0.8894, 0.6542, 0.7376, 0.7628, 1.6173, 0.7649, 0.7348,
        1.7870, 0.7166, 0.6219, 1.6783, 1.2398, 0.9570, 1.6880, 2.5174, 0.9264,
        0.9430, 0.8867, 0.6019, 0.8343, 0.6127, 1.9430, 1.1309, 0.9234, 0.6462,
        1.1443, 1.0925, 0.7311, 0.7910, 1.2319, 0.7320, 0.6146, 0.5938, 1.5950,
        0.7357, 0.7835, 0.7329, 1.2670, 0.8178, 0.6565, 1.7387, 2.4850, 0.7423,
        1.1178, 0.7964, 0.5750, 1.7080, 0.8155, 1.1816, 0.7376, 0.7953, 1.3189,
        0.6595, 1.6638, 1.0230, 1.5117, 1.4658, 1.1914, 0.5831, 0.9747, 0.9539,
        2.1287, 0.6038, 2.3441, 0.6000, 1.6083, 1.7815, 1.0722, 1.4846, 1.1673,
        0.9966, 0.9586, 1.6039, 1.1534, 0.7175, 2.2099, 0.5920, 1.1265, 0.8053,
        2.0035, 1.1580, 0.6108, 0.6572, 1.2982, 1.2345, 0.6820, 0.9847, 1.8381,
        0.8813, 0.5872, 1.5277, 0.6050, 0.5819, 0.9586, 1.0017, 0.6671, 1.3885,
        0.6280, 0.8733, 0.6686, 1.7981, 2.0316, 0.6678, 1.1221, 1.5481, 1.6686,
        2.0387, 0.9476, 0.7598, 1.2319, 1.2138, 0.6542, 0.7148, 0.6239, 0.7210,
        1.2345, 0.5778, 0.5831, 0.6602, 2.1365, 1.0762, 1.9829, 0.8616, 1.0212,
        0.6266, 0.9047, 0.8707, 0.8629, 1.6356, 2.1131, 0.6321, 1.0945, 1.3372,
        0.6836, 1.2697, 1.0585, 1.7335, 1.7080, 1.8094, 1.3786, 0.8746, 1.9236,
        2.0459, 1.0643, 0.6405, 1.3986, 2.0316, 0.7490, 0.7166, 0.5932, 1.2241,
        0.6909, 0.7720, 0.8098, 1.3280, 2.1209, 0.7772, 1.0087, 0.7997, 0.7964,
        0.7052, 1.9236, 1.0358, 0.6108, 1.0052, 0.8379, 1.6880, 1.4923, 0.8921,
        0.7588, 1.2138, 1.0194, 0.7549, 0.7070, 0.6153, 0.6426, 0.7010, 0.6287,
        0.7452, 0.8166, 1.3434, 1.5481, 1.3497, 1.0302, 1.3372, 1.3852, 0.9279,
        2.0605, 1.0683, 1.5039, 0.7131, 0.6153, 0.8166, 0.6193, 0.5872, 0.6528,
        1.7652, 1.8860, 2.0532, 1.0105, 0.7175, 1.7815, 1.6083, 1.2697, 0.5975,
        0.8295, 1.7440, 0.5738, 0.6477, 0.8143, 0.6114, 1.2697, 0.6440, 0.9399,
        0.9731, 1.5863, 0.9983, 1.9046, 0.8477, 1.0742, 1.4548, 1.8208, 0.5902,
        2.1055, 2.0532, 1.1050, 0.6239, 0.7856, 0.8840, 2.0679, 1.9046, 0.5796,
        0.6610, 1.2895, 2.2099, 0.7061, 1.3011, 1.4019, 0.8921, 1.9829, 1.0604,
        1.2953, 1.3100, 0.7669, 0.7751
    ])

    return class_weight
