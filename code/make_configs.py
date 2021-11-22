import json
import os

# num_stacks=[6,5,4,3,2,1]
# num_stacks=[2,3]
num_stacks=[2, 3, 4, 5]
lr=[0.001, 0.01]
data = ['ecg_3', 'ecg_4', 'ecg_7', 'nyc_taxi', 'TEK16', 'ecg_2', 'power_demand', 'ecg_1', 'ecg_5', 'TEK17', 'ecg_8', 'ecg_9', 'gesture', 'TEK14', 'ecg_6', 'nprs43', 'nprs44']
data_dir = ["NAB-known-anomaly"]
# lr=[0.01, 0.005, 0.003, 0.002, 0.001]
# data=["scada2", "scada1"]
d_ff = [128]
# auto_dims = [(200, 50), (200, 50), (200, 180), (200, 160), (200, 140), (200, 120), (200, 80), (200, 60)]
# auto_dims = [(50, 25), (100, 80), (100, 50)]
auto_dims = [(100, 80)]
batch_size = [64]
# auto_dims = [(120, 40), (100, 40)]
# auto_dims.extend([(120, 60), (140, 70), ( 160, 80),  (180, 90)])

for dd in data_dir: 
    for d in data:
        for num in num_stacks:
            for l in lr:
                for ff in d_ff:
                    for auto in auto_dims:
                        for b in batch_size:
                            filename = "{}_CEN_AUTO_{}stacks_{}dff_{}auto_{}trans_{}lr_{}batch".format(d.upper(), num, ff, auto[0], auto[1], l, b).replace(".", "_")
                            print(filename)
                            d_model = 1
                            if 'ecg' in d or 'gesture' in d:
                                d_model = 2
                            config_path = os.path.join("./configs/to_be_run", "{}.json".format(filename))
                            configs = {
                                "experiment": filename,
                                "data_dir": "{}".format(dd),
                                "auto_dataset": "{}".format(d),
                                "num_stacks": num,
                                "d_model": d_model,
                                "d_ff": ff,
                                "num_heads": 1,
                                "dropout": 0.1,
                                "autoencoder_dims": auto[0],
                                "l_win": auto[1],
                                "pre_mask": int(auto[1] / 5 * 2),
                                "post_mask": int(auto[1] / 5 * 3),
                                "batch_size": b,
                                "shuffle": 1,
                                "dataloader_num_workers": 4,
                                "auto_num_epoch": 50,
                                "trans_num_epoch": 50,
                                "lr": l,
                                "load_dir": "default"
                            }
                            with open(config_path, 'w', encoding='utf-8') as f:
                                json.dump(configs, f, indent=4)

